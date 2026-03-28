"""Detect ring markers in a board image.

Usage:
    python detect_ringgrid.py path/to/ringgrid.png [--output result.png]

The script uses the `radsym` Python bindings to:
1. load a grayscale image,
2. compute an outer-radius radial symmetry response,
3. extract center proposals,
4. fit outer and inner ellipses around each proposal,
5. render overlays on the source image and response heatmap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse as MplEllipse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import radsym
from _surf_hole_common import timed_call

console = Console()


def build_radius_band(
    base_radius: float,
    start_scale: float = 0.8,
    stop_scale: float = 1.16,
    steps: int = 5,
) -> list[int]:
    start = max(4, int(round(base_radius * start_scale)))
    stop = max(start + 1, int(round(base_radius * stop_scale)))
    if steps <= 1 or start == stop:
        return [start]

    radii = {
        int(round(start + (stop - start) * index / (steps - 1)))
        for index in range(steps)
    }
    return sorted(radius for radius in radii if radius > 0)


def suppress_near_duplicates(
    proposals: list[radsym.Proposal],
    min_distance: float,
) -> list[radsym.Proposal]:
    kept: list[radsym.Proposal] = []
    min_distance = max(0.0, min_distance)
    for proposal in proposals:
        px, py = proposal.position
        duplicate = False
        for other in kept:
            ox, oy = other.position
            if math.hypot(px - ox, py - oy) < min_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
    return kept


def ellipse_patch(
    ellipse: radsym.Ellipse,
    edgecolor: str,
    linewidth: float,
    alpha: float = 1.0,
):
    return MplEllipse(
        xy=ellipse.center,
        width=2.0 * ellipse.semi_major,
        height=2.0 * ellipse.semi_minor,
        angle=math.degrees(ellipse.angle),
        edgecolor=edgecolor,
        facecolor="none",
        linewidth=linewidth,
        alpha=alpha,
    )


def render_overlay(
    image,
    detections: list[dict],
    detector_name: str,
    fit_ellipses: bool,
):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    if fit_ellipses:
        for detection in detections:
            ax.add_patch(ellipse_patch(detection["outer_ellipse"], "#ff8c1a", 1.8))
            ax.add_patch(
                ellipse_patch(detection["inner_ellipse"], "#6aff6a", 1.3, alpha=0.9)
            )

    centers_x = [detection["center"][0] for detection in detections]
    centers_y = [detection["center"][1] for detection in detections]
    if centers_x:
        ax.scatter(
            centers_x,
            centers_y,
            s=28,
            c="#43e6ff",
            edgecolors="black",
            linewidths=0.3,
            label="Detected centers",
        )

    mode_label = "ellipses" if fit_ellipses else "proposals"
    ax.set_title(f"Ringgrid {mode_label} ({detector_name.upper()})")
    ax.set_axis_off()
    if centers_x:
        ax.legend(loc="lower right", frameon=True)
    return fig


def render_heatmap(
    response,
    detections: list[dict],
    detector_name: str,
    fit_ellipses: bool,
):
    heatmap = response.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image_artist = ax.imshow(heatmap, cmap="hot")
    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04, label="Response")

    if fit_ellipses:
        for detection in detections:
            ax.add_patch(ellipse_patch(detection["outer_ellipse"], "white", 1.5, alpha=0.95))

    centers_x = [detection["center"][0] for detection in detections]
    centers_y = [detection["center"][1] for detection in detections]
    if centers_x:
        ax.scatter(
            centers_x,
            centers_y,
            s=24,
            c="#6aff6a",
            edgecolors="black",
            linewidths=0.3,
            label="Detected centers",
        )

    mode_label = "ellipse seeds" if fit_ellipses else "proposal centers"
    ax.set_title(f"Ringgrid response ({detector_name.upper()}, {mode_label})")
    ax.set_axis_off()
    if centers_x:
        ax.legend(loc="lower right", frameon=True)
    return fig


def render_summary(
    image_path: Path,
    width: int,
    height: int,
    detector_name: str,
    fit_ellipses: bool,
    outer_radius: float,
    inner_ratio: float,
    radii: list[int],
    raw_proposal_count: int,
    deduplicated_count: int,
    final_count: int,
    detections: list[dict],
) -> None:
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Image", str(image_path))
    summary.add_row("Image size", f"{width}x{height}")
    summary.add_row("Detector", detector_name.upper())
    summary.add_row("Mode", "ellipse refinement" if fit_ellipses else "proposals only")
    summary.add_row("Outer radius prior", f"{outer_radius:.2f} px")
    summary.add_row("Inner ratio prior", f"{inner_ratio:.3f}")
    summary.add_row("Radius band", str(radii))
    summary.add_row("Raw proposals", str(raw_proposal_count))
    summary.add_row("Deduplicated proposals", str(deduplicated_count))
    summary.add_row("Final detections", str(final_count))
    if fit_ellipses and detections:
        best_support = detections[0]["combined_support"]
        worst_support = detections[-1]["combined_support"]
        summary.add_row("Combined support", f"{best_support:.3f} .. {worst_support:.3f}")
    elif detections:
        best_score = detections[0]["proposal"].score
        worst_score = detections[-1]["proposal"].score
        summary.add_row("Proposal score", f"{best_score:.3f} .. {worst_score:.3f}")

    console.print(Panel(summary, title="Ringgrid Summary", expand=False))


def render_performance(metrics: dict) -> None:
    table = Table(title="radsym Performance", header_style="bold magenta")
    table.add_column("Call", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Total ms", justify="right")
    table.add_column("Avg ms", justify="right")

    for name, stat in sorted(metrics.items(), key=lambda item: item[1]["total_ms"], reverse=True):
        count = stat["count"]
        total_ms = stat["total_ms"]
        avg_ms = total_ms / max(1, count)
        table.add_row(name, str(count), f"{total_ms:.2f}", f"{avg_ms:.2f}")

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ring markers in a board image using RSD/FRST proposals and local ellipse fitting."
    )
    parser.add_argument("image", type=Path, help="Input grayscale-compatible image path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the grayscale detection overlay PNG.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        help="Output path for the response heatmap overlay PNG.",
    )
    parser.add_argument(
        "--detector",
        default="rsd",
        choices=("rsd", "frst"),
        help="Proposal generator. Default is rsd because it recovers all ringgrid centers on the provided board image.",
    )
    parser.add_argument(
        "--polarity",
        default="dark",
        choices=("bright", "dark", "both"),
        help="Proposal polarity mode. Default is dark for the provided ringgrid image.",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        help="Outer ellipse radius prior in pixels. Defaults to 4.2%% of the minimum image dimension.",
    )
    parser.add_argument(
        "--inner-ratio",
        type=float,
        default=0.47,
        help="Inner/outer radius ratio prior. Default: 0.47.",
    )
    parser.add_argument(
        "--min-center-distance",
        type=float,
        help="Minimum distance between retained proposal centers. Defaults to 1.25 * outer radius prior.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=96,
        help="Maximum number of deduplicated detections to keep. Default: 96.",
    )
    parser.add_argument(
        "--fit-ellipses",
        action="store_true",
        help="Run local outer and inner ellipse refinement for each proposal. Default is proposals-only.",
    )
    parser.add_argument(
        "--min-combined-support",
        type=float,
        default=0.0,
        help="Minimum weighted outer/inner support score to keep a refined detection. Only used with --fit-ellipses.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    overlay_path = args.output.expanduser().resolve() if args.output is not None else None
    heatmap_path = (
        args.heatmap_output.expanduser().resolve()
        if args.heatmap_output is not None
        else None
    )

    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

    image = timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(image_path))
    height, width = image.shape
    min_dim = min(height, width)
    outer_radius = args.outer_radius or max(10.0, min_dim * 0.042)
    inner_ratio = max(0.1, min(0.9, args.inner_ratio))
    inner_radius = outer_radius * inner_ratio
    radii = build_radius_band(outer_radius)
    min_center_distance = (
        args.min_center_distance
        if args.min_center_distance is not None
        else 1.25 * outer_radius
    )

    gradient = timed_call(metrics, "sobel_gradient", radsym.sobel_gradient, image)
    if args.detector == "rsd":
        response = timed_call(
            metrics,
            "rsd_response",
            radsym.rsd_response,
            gradient,
            radsym.RsdConfig(
                radii=radii,
                gradient_threshold=2.0,
                polarity=args.polarity,
                smoothing_factor=0.5,
            ),
        )
    else:
        response = timed_call(
            metrics,
            "frst_response",
            radsym.frst_response,
            gradient,
            radsym.FrstConfig(
                radii=radii,
                alpha=2.0,
                gradient_threshold=2.0,
                polarity=args.polarity,
                smoothing_factor=0.5,
            ),
        )

    proposals = timed_call(
        metrics,
        "extract_proposals",
        radsym.extract_proposals,
        response,
        radsym.NmsConfig(
            radius=max(6, int(round(0.55 * outer_radius))),
            threshold=0.01,
            max_detections=256,
        ),
        polarity=args.polarity,
    )
    if not proposals:
        raise RuntimeError("no proposals found")

    raw_proposal_count = len(proposals)
    proposals = suppress_near_duplicates(proposals, min_center_distance)
    proposals = proposals[: args.max_detections]
    detections: list[dict]
    if args.fit_ellipses:
        outer_refine_config = radsym.EllipseRefineConfig(
            max_iterations=5,
            convergence_tol=0.05,
            annulus_margin=0.12,
            ray_count=96,
            radial_search_inner=0.60,
            radial_search_outer=1.45,
            normal_search_half_width=6.0,
            min_inlier_coverage=0.60,
            max_center_shift_fraction=0.40,
            max_axis_ratio=1.80,
        )
        inner_refine_config = radsym.EllipseRefineConfig(
            max_iterations=5,
            convergence_tol=0.05,
            annulus_margin=0.10,
            ray_count=96,
            radial_search_inner=0.75,
            radial_search_outer=1.20,
            normal_search_half_width=4.0,
            min_inlier_coverage=0.55,
            max_center_shift_fraction=0.25,
            max_axis_ratio=1.80,
        )
        scoring_config = radsym.ScoringConfig(
            annulus_margin=0.12,
            min_samples=32,
            weight_ringness=0.60,
            weight_coverage=0.40,
        )

        detections = []
        for proposal in proposals:
            outer_seed = radsym.Ellipse(proposal.position, outer_radius, outer_radius, 0.0)
            outer_result = timed_call(
                metrics,
                "refine_ellipse (outer)",
                radsym.refine_ellipse,
                gradient,
                outer_seed,
                outer_refine_config,
            )
            outer_ellipse = outer_result.hypothesis
            outer_support = timed_call(
                metrics,
                "score_ellipse_support (outer)",
                radsym.score_ellipse_support,
                gradient,
                outer_ellipse,
                scoring_config,
            )

            inner_seed = radsym.Ellipse(
                outer_ellipse.center,
                outer_ellipse.semi_major * inner_ratio,
                outer_ellipse.semi_minor * inner_ratio,
                outer_ellipse.angle,
            )
            inner_result = timed_call(
                metrics,
                "refine_ellipse (inner)",
                radsym.refine_ellipse,
                gradient,
                inner_seed,
                inner_refine_config,
            )
            inner_ellipse = inner_result.hypothesis
            inner_support = timed_call(
                metrics,
                "score_ellipse_support (inner)",
                radsym.score_ellipse_support,
                gradient,
                inner_ellipse,
                scoring_config,
            )

            combined_support = 0.65 * outer_support.total + 0.35 * inner_support.total
            if (
                combined_support < args.min_combined_support
                or outer_ellipse.center[0] < 0.0
                or outer_ellipse.center[1] < 0.0
                or outer_ellipse.center[0] >= width
                or outer_ellipse.center[1] >= height
            ):
                continue

            detections.append(
                {
                    "proposal": proposal,
                    "center": outer_ellipse.center,
                    "outer_result": outer_result,
                    "inner_result": inner_result,
                    "outer_ellipse": outer_ellipse,
                    "inner_ellipse": inner_ellipse,
                    "outer_support": outer_support,
                    "inner_support": inner_support,
                    "combined_support": combined_support,
                }
            )

        if not detections:
            raise RuntimeError("no refined detections passed the support threshold")

        detections.sort(key=lambda item: item["combined_support"], reverse=True)
    else:
        detections = [
            {
                "proposal": proposal,
                "center": proposal.position,
            }
            for proposal in proposals
        ]

    overlay_figure = render_overlay(image, detections, args.detector, args.fit_ellipses)
    heatmap_figure = render_heatmap(response, detections, args.detector, args.fit_ellipses)

    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_figure.savefig(overlay_path, dpi=180, bbox_inches="tight")
    if heatmap_path is not None:
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_figure.savefig(heatmap_path, dpi=180, bbox_inches="tight")

    if overlay_path is None and heatmap_path is None:
        plt.show()
    else:
        plt.close(overlay_figure)
        plt.close(heatmap_figure)

    render_summary(
        image_path=image_path,
        width=width,
        height=height,
        detector_name=args.detector,
        fit_ellipses=args.fit_ellipses,
        outer_radius=outer_radius,
        inner_ratio=inner_ratio,
        radii=radii,
        raw_proposal_count=raw_proposal_count,
        deduplicated_count=len(proposals),
        final_count=len(detections),
        detections=detections,
    )
    render_performance(metrics)
    if overlay_path is not None:
        console.print(f"[bold]overlay:[/bold] {overlay_path}")
    if heatmap_path is not None:
        console.print(f"[bold]heatmap:[/bold] {heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
