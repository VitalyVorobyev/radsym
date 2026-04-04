"""Detect ring markers using FRST (per-radius) proposals.

Usage:
    python detect_ringgrid_frst.py path/to/ringgrid.png [--output result.png]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ring markers using per-radius FRST proposals."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="Output overlay PNG path.")
    parser.add_argument("--heatmap-output", type=Path, help="Output heatmap PNG path.")
    parser.add_argument("--polarity", default="dark", choices=("bright", "dark", "both"))
    parser.add_argument("--outer-radius", type=float)
    parser.add_argument("--max-detections", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

    image = timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(args.image))
    t0 = perf_counter()
    height, width = image.shape
    outer_radius = args.outer_radius or max(10.0, min(height, width) * 0.042)
    radii = build_radius_band(outer_radius)

    gradient = timed_call(metrics, "sobel_gradient", radsym.sobel_gradient, image)
    config = radsym.FrstConfig(
        radii=radii, alpha=2.0, gradient_threshold=2.0,
        polarity=args.polarity, smoothing_factor=0.5,
    )
    response = timed_call(metrics, "frst_response", radsym.frst_response, gradient, config)

    nms = radsym.NmsConfig(
        radius=max(6, int(round(0.55 * outer_radius))),
        threshold=0.01, max_detections=256,
    )
    proposals = timed_call(
        metrics, "extract_proposals", radsym.extract_proposals,
        response, nms, polarity=args.polarity,
    )
    raw_count = len(proposals)
    proposals = suppress_near_duplicates(proposals, 1.25 * outer_radius)
    proposals = proposals[: args.max_detections]
    centers = [p.position for p in proposals]
    processing_ms = (perf_counter() - t0) * 1000.0

    overlay_path = args.output.expanduser().resolve() if args.output else None
    heatmap_path = args.heatmap_output.expanduser().resolve() if args.heatmap_output else None
    overlay_fig = render_overlay(image, centers, "FRST (per-radius)")
    heatmap_fig = render_heatmap(response, centers, "FRST (per-radius)")
    show_or_save([(overlay_fig, overlay_path), (heatmap_fig, heatmap_path)])

    render_summary(
        str(args.image), width, height, "FRST (per-radius)",
        radii, raw_count, len(proposals), processing_ms,
        title="FRST Detection",
    )
    render_performance(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
