"""Detect ring markers in a board image.

Usage:
    python detect_ringgrid.py path/to/ringgrid.png [--output result.png]

The script uses the `radsym` Python bindings to:
1. load a grayscale image,
2. compute an outer-radius radial symmetry response,
3. extract center proposals,
4. optionally fit outer and inner ellipses around each proposal,
5. render overlays on the source image and response heatmap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import radsym
from _ringgrid_common import (
    build_radius_band,
    render_heatmap,
    render_overlay,
    render_performance,
    render_summary,
    show_or_save,
    suppress_near_duplicates,
)
from _surf_hole_common import timed_call

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ring markers in a board image using RSD/FRST proposals and local ellipse fitting."
    )
    parser.add_argument("image", type=Path, help="Input grayscale-compatible image path.")
    parser.add_argument("--output", type=Path, help="Output overlay PNG path.")
    parser.add_argument("--heatmap-output", type=Path, help="Output heatmap PNG path.")
    parser.add_argument(
        "--detector", default="rsd", choices=("rsd", "frst"),
        help="Proposal generator (default: rsd).",
    )
    parser.add_argument(
        "--polarity", default="dark", choices=("bright", "dark", "both"),
        help="Proposal polarity mode (default: dark).",
    )
    parser.add_argument("--outer-radius", type=float, help="Outer radius prior in px.")
    parser.add_argument("--inner-ratio", type=float, default=0.47, help="Inner/outer ratio.")
    parser.add_argument("--min-center-distance", type=float)
    parser.add_argument("--max-detections", type=int, default=96)
    parser.add_argument("--fit-ellipses", action="store_true")
    parser.add_argument("--min-combined-support", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    overlay_path = args.output.expanduser().resolve() if args.output else None
    heatmap_path = args.heatmap_output.expanduser().resolve() if args.heatmap_output else None

    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

    image = timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(image_path))
    t0 = perf_counter()
    height, width = image.shape
    outer_radius = args.outer_radius or max(10.0, min(height, width) * 0.042)
    inner_ratio = max(0.1, min(0.9, args.inner_ratio))
    radii = build_radius_band(outer_radius)
    min_dist = args.min_center_distance if args.min_center_distance is not None else 1.25 * outer_radius

    gradient = timed_call(metrics, "sobel_gradient", radsym.sobel_gradient, image)
    if args.detector == "rsd":
        response = timed_call(
            metrics, "rsd_response", radsym.rsd_response, gradient,
            radsym.RsdConfig(radii=radii, gradient_threshold=2.0, polarity=args.polarity, smoothing_factor=0.5),
        )
    else:
        response = timed_call(
            metrics, "frst_response", radsym.frst_response, gradient,
            radsym.FrstConfig(radii=radii, alpha=2.0, gradient_threshold=2.0, polarity=args.polarity, smoothing_factor=0.5),
        )

    proposals = timed_call(
        metrics, "extract_proposals", radsym.extract_proposals, response,
        radsym.NmsConfig(radius=max(6, int(round(0.55 * outer_radius))), threshold=0.01, max_detections=256),
        polarity=args.polarity,
    )
    if not proposals:
        raise RuntimeError("no proposals found")

    raw_count = len(proposals)
    proposals = suppress_near_duplicates(proposals, min_dist)
    proposals = proposals[: args.max_detections]

    centers: list[tuple[float, float]]
    ellipses_overlay: list[tuple[radsym.Ellipse, str, float]] = []

    if args.fit_ellipses:
        outer_cfg = radsym.EllipseRefineConfig(
            max_iterations=5, convergence_tol=0.05, annulus_margin=0.12, ray_count=96,
            radial_search_inner=0.60, radial_search_outer=1.45,
            normal_search_half_width=6.0, min_inlier_coverage=0.60,
            max_center_shift_fraction=0.40, max_axis_ratio=1.80,
        )
        inner_cfg = radsym.EllipseRefineConfig(
            max_iterations=5, convergence_tol=0.05, annulus_margin=0.10, ray_count=96,
            radial_search_inner=0.75, radial_search_outer=1.20,
            normal_search_half_width=4.0, min_inlier_coverage=0.55,
            max_center_shift_fraction=0.25, max_axis_ratio=1.80,
        )
        scoring_cfg = radsym.ScoringConfig(annulus_margin=0.12, min_samples=32, weight_ringness=0.60, weight_coverage=0.40)

        detections = []
        for proposal in proposals:
            outer_seed = radsym.Ellipse(proposal.position, outer_radius, outer_radius, 0.0)
            outer_result = timed_call(metrics, "refine_ellipse (outer)", radsym.refine_ellipse, gradient, outer_seed, outer_cfg)
            outer_ellipse = outer_result.hypothesis
            outer_support = timed_call(metrics, "score_ellipse_support (outer)", radsym.score_ellipse_support, gradient, outer_ellipse, scoring_cfg)

            inner_seed = radsym.Ellipse(outer_ellipse.center, outer_ellipse.semi_major * inner_ratio, outer_ellipse.semi_minor * inner_ratio, outer_ellipse.angle)
            inner_result = timed_call(metrics, "refine_ellipse (inner)", radsym.refine_ellipse, gradient, inner_seed, inner_cfg)
            inner_ellipse = inner_result.hypothesis
            inner_support = timed_call(metrics, "score_ellipse_support (inner)", radsym.score_ellipse_support, gradient, inner_ellipse, scoring_cfg)

            combined = 0.65 * outer_support.total + 0.35 * inner_support.total
            cx, cy = outer_ellipse.center
            if combined < args.min_combined_support or cx < 0 or cy < 0 or cx >= width or cy >= height:
                continue
            detections.append({"center": outer_ellipse.center, "outer": outer_ellipse, "inner": inner_ellipse, "combined": combined, "proposal": proposal})

        if not detections:
            raise RuntimeError("no refined detections passed the support threshold")
        detections.sort(key=lambda d: d["combined"], reverse=True)

        centers = [d["center"] for d in detections]
        for d in detections:
            ellipses_overlay.append((d["outer"], "#ff8c1a", 1.8))
            ellipses_overlay.append((d["inner"], "#6aff6a", 1.3))
    else:
        centers = [p.position for p in proposals]

    processing_ms = (perf_counter() - t0) * 1000.0

    overlay_fig = render_overlay(image, centers, args.detector, ellipses=ellipses_overlay or None)
    heatmap_fig = render_heatmap(response, centers, args.detector, ellipses=[(e, c, lw) for e, c, lw in ellipses_overlay if c == "#ff8c1a"] or None)

    show_or_save([(overlay_fig, overlay_path), (heatmap_fig, heatmap_path)])

    render_summary(str(image_path), width, height, args.detector.upper(), radii, raw_count, len(centers), processing_ms, title="Ringgrid Summary")
    render_performance(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
