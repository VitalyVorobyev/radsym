"""Detect a single central hole-like structure in a surf image.

Usage:
    python detect_surf_hole.py path/to/image.png [--output result.png]

The script uses the `radsym` Python bindings to:
1. load a grayscale image,
2. compute an FRST response map,
3. extract center proposals,
4. refine the best central candidate as an ellipse,
5. render matplotlib overlays on the source image and FRST heatmap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import radsym
from _surf_hole_common import (
    detect_working_image,
    downscale_to_level,
    timed_call,
)

console = Console()

ELLIPSE_FIT_MIN_ALIGNMENT = 0.3
ELLIPSE_FIT_ANGULAR_SAMPLES = 64
ELLIPSE_FIT_RADIAL_SAMPLES = 9

def render_overlay(image, ranked: list[dict], downscale: int, detection: dict):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse as MplEllipse

    full_resolution_ellipse = detection["full_resolution_ellipse"]

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    proposals_x = [item["image_proposal"].position[0] for item in ranked[:5]]
    proposals_y = [item["image_proposal"].position[1] for item in ranked[:5]]
    if proposals_x:
        ax.scatter(
            proposals_x,
            proposals_y,
            s=80,
            c="#3ad7ff",
            marker="+",
            linewidths=1.8,
            label="Top proposals",
        )

    ax.add_patch(
        MplEllipse(
            xy=full_resolution_ellipse.center,
            width=2.0 * full_resolution_ellipse.semi_major,
            height=2.0 * full_resolution_ellipse.semi_minor,
            angle=math.degrees(full_resolution_ellipse.angle),
            edgecolor="#ff7f0e",
            facecolor="none",
            linewidth=2.5,
            label="Refined ellipse",
        )
    )
    ax.scatter(
        [full_resolution_ellipse.center[0]],
        [full_resolution_ellipse.center[1]],
        s=55,
        c="#ffee58",
        edgecolors="black",
        linewidths=0.6,
        label="Refined center",
    )
    ax.set_title(f"Surf hole detection (full resolution, {downscale}x working downscale)")
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True)
    return fig


def render_heatmap(response, ranked: list[dict], downscale: int, detection: dict):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse as MplEllipse

    ellipse = detection["working_ellipse"]

    heatmap = response.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image_artist = ax.imshow(heatmap, cmap="magma")
    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04, label="FRST response")

    proposals_x = [item["proposal"].position[0] for item in ranked[:5]]
    proposals_y = [item["proposal"].position[1] for item in ranked[:5]]
    if proposals_x:
        ax.scatter(
            proposals_x,
            proposals_y,
            s=80,
            c="#9efeff",
            marker="+",
            linewidths=1.8,
            label="Top proposals",
        )

    ax.add_patch(
        MplEllipse(
            xy=ellipse.center,
            width=2.0 * ellipse.semi_major,
            height=2.0 * ellipse.semi_minor,
            angle=math.degrees(ellipse.angle),
            edgecolor="white",
            facecolor="none",
            linewidth=2.2,
            label="Refined ellipse",
        )
    )
    ax.scatter(
        [detection["best"]["proposal"].position[0]],
        [detection["best"]["proposal"].position[1]],
        s=65,
        c="#ff66c4",
        marker="x",
        linewidths=1.8,
        label="Pre-ellipse center",
    )
    ax.scatter(
        [ellipse.center[0]],
        [ellipse.center[1]],
        s=55,
        c="#72ff72",
        edgecolors="black",
        linewidths=0.6,
        label="Refined center",
    )
    ax.set_title(f"Surf FRST heatmap ({downscale}x downscaled working image)")
    ax.set_axis_off()
    ax.legend(loc="lower right", frameon=True)
    return fig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the main central hole-like structure in a surf image."
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
        help="Output path for the FRST heatmap overlay PNG.",
    )
    parser.add_argument(
        "--polarity",
        default="bright",
        choices=("bright", "dark", "both"),
        help="FRST polarity mode. Default is bright for the provided surf images.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=8,
        help="Detection downscale factor. Detection runs on the downscaled working image.",
    )
    parser.add_argument(
        "--working-radius-hint",
        type=float,
        help="Optional radius hint in working-image pixels. Overrides the automatic size-based hint.",
    )
    return parser.parse_args()


def render_summary(
    detection: dict,
    processing_ms: float,
) -> None:
    ellipse = detection["full_resolution_ellipse"]
    support = detection["best"]["support"]
    ellipse_seed = detection["best"]["ellipse_seed"]
    width, height = detection["image_size"]
    working_width, working_height = detection["working_size"]
    downscale = detection["downscale"]
    working_radius_hint = detection["working_radius_hint"]

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Image size", f"{width}x{height}")
    summary.add_row("Working size", f"{working_width}x{working_height}")
    summary.add_row("Downscale", f"{downscale}x")
    summary.add_row("Working radius hint", f"{working_radius_hint:.2f} px")
    summary.add_row("Center", f"({ellipse.center[0]:.2f}, {ellipse.center[1]:.2f})")
    summary.add_row("Semi-major", f"{ellipse.semi_major:.2f} px")
    summary.add_row("Semi-minor", f"{ellipse.semi_minor:.2f} px")
    summary.add_row(
        "Axis ratio",
        f"{ellipse.semi_major / max(1e-6, ellipse.semi_minor):.3f}",
    )
    summary.add_row("Angle", f"{math.degrees(ellipse.angle):.2f} deg")
    summary.add_row(
        "Swept circle radius",
        f"{detection['best']['radius_sweep']['radius'] * downscale:.2f} px",
    )
    summary.add_row(
        "Seed ellipse",
        f"circle @ {ellipse_seed['ellipse'].semi_major * downscale:.2f} px",
    )
    summary.add_row("Support", f"{support.total:.3f}")
    summary.add_row("Coverage", f"{support.angular_coverage:.3f}")
    summary.add_row("Processing time", f"{processing_ms:.2f} ms (excl. load)")
    console.print(Panel(summary, title="Detection Summary", expand=False))


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


def render_frst_cost(frst_config: radsym.FrstConfig, width: int, height: int) -> None:
    sigmas = [frst_config.smoothing_factor * radius for radius in frst_config.radii]
    kernel_widths = [
        1 if sigma <= 0.5 else 2 * int(math.ceil(3.0 * sigma)) + 1 for sigma in sigmas
    ]
    estimated_blur_taps = width * height * 2 * sum(kernel_widths)

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Radii", str(frst_config.radii))
    summary.add_row("Smoothing factor", f"{frst_config.smoothing_factor:.2f}")
    summary.add_row("Sigmas", "[" + ", ".join(f"{sigma:.1f}" for sigma in sigmas) + "]")
    summary.add_row("Blur kernels", "[" + ", ".join(str(width) for width in kernel_widths) + "]")
    summary.add_row("Estimated blur taps", f"{estimated_blur_taps:,}")
    summary.add_row("Build note", "Use `maturin develop --release` for realistic timings")
    console.print(Panel(summary, title="FRST Cost", expand=False))


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    level = downscale_to_level(args.downscale)
    overlay_path = args.output.expanduser().resolve() if args.output is not None else None
    heatmap_path = (
        args.heatmap_output.expanduser().resolve()
        if args.heatmap_output is not None
        else None
    )
    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    if args.working_radius_hint is not None and args.working_radius_hint <= 0.0:
        raise ValueError("--working-radius-hint must be > 0")

    image = timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(image_path))
    processing_start = perf_counter()
    pyramid = timed_call(
        metrics,
        "pyramid_level_image",
        radsym.pyramid_level_image,
        image,
        level,
    )
    working_image = pyramid.to_numpy()
    working = detect_working_image(
        working_image,
        args.polarity,
        metrics,
        working_radius_hint=args.working_radius_hint,
    )
    response = working["response"]
    best = working["best"]
    full_resolution_ellipse = pyramid.map_ellipse(best["ellipse"])
    ranked = []
    for item in working["ranked"]:
        ranked_item = dict(item)
        ranked_item["image_proposal"] = pyramid.map_proposal(item["proposal"])
        ranked_item["full_resolution_ellipse"] = pyramid.map_ellipse(item["ellipse"])
        ranked.append(ranked_item)
    detection = {
        "image_size": (image.shape[1], image.shape[0]),
        "working_size": (working_image.shape[1], working_image.shape[0]),
        "downscale": args.downscale,
        "level": level,
        "radii": working["frst_config"].radii,
        "working_ellipse": best["ellipse"],
        "full_resolution_ellipse": full_resolution_ellipse,
        "working_radius_hint": working["radius_hint"],
        "best": best,
        "ranked": ranked,
        "pyramid": pyramid,
    }
    processing_ms = (perf_counter() - processing_start) * 1000.0
    working_height, working_width = working_image.shape

    overlay_figure = render_overlay(image, ranked, args.downscale, detection)
    heatmap_figure = render_heatmap(response, ranked, args.downscale, detection)

    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_figure.savefig(overlay_path, dpi=180, bbox_inches="tight")
    if heatmap_path is not None:
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        heatmap_figure.savefig(heatmap_path, dpi=180, bbox_inches="tight")

    if overlay_path is None and heatmap_path is None:
        from matplotlib import pyplot as plt

        plt.show()
    else:
        from matplotlib import pyplot as plt

        plt.close(overlay_figure)
        plt.close(heatmap_figure)

    console.print(f"[bold]image:[/bold] {image_path}")
    console.print(f"[bold]radii:[/bold] {detection['radii']}")
    console.print(f"[bold]proposals:[/bold] {len(detection['ranked'])}")
    render_summary(detection, processing_ms)
    render_frst_cost(working["frst_config"], working_width, working_height)
    render_performance(metrics)
    if overlay_path is not None:
        console.print(f"[bold]overlay:[/bold] {overlay_path}")
    if heatmap_path is not None:
        console.print(f"[bold]heatmap:[/bold] {heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
