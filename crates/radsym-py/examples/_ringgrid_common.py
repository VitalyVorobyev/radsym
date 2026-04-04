"""Shared utilities for ringgrid detection examples."""

from __future__ import annotations

import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse as MplEllipse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import radsym

console = Console()


# ---------------------------------------------------------------------------
# Radius helpers
# ---------------------------------------------------------------------------


def build_radius_band(
    base_radius: float,
    start_scale: float = 0.8,
    stop_scale: float = 1.16,
    steps: int = 5,
) -> list[int]:
    """Build a set of geometrically spaced integer radii around a base radius."""
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
    """Greedily suppress proposals closer than min_distance."""
    kept: list[radsym.Proposal] = []
    min_distance = max(0.0, min_distance)
    for proposal in proposals:
        px, py = proposal.position
        if not any(
            math.hypot(px - ox, py - oy) < min_distance
            for ox, oy in (o.position for o in kept)
        ):
            kept.append(proposal)
    return kept


# ---------------------------------------------------------------------------
# Matplotlib rendering
# ---------------------------------------------------------------------------


def ellipse_patch(
    ellipse: radsym.Ellipse,
    edgecolor: str,
    linewidth: float,
    alpha: float = 1.0,
) -> MplEllipse:
    """Create a matplotlib Ellipse patch from a radsym Ellipse."""
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
    centers: list[tuple[float, float]],
    detector_name: str,
    *,
    ellipses: list[tuple[radsym.Ellipse, str, float]] | None = None,
) -> plt.Figure:
    """Render detected centers (and optional ellipses) over a grayscale image.

    Args:
        image: 2D numpy array (grayscale).
        centers: list of (x, y) center positions.
        detector_name: label for the title.
        ellipses: optional list of (ellipse, color, linewidth) tuples to draw.
    """
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    if ellipses:
        for ellipse, color, lw in ellipses:
            ax.add_patch(ellipse_patch(ellipse, color, lw))

    if centers:
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        ax.scatter(
            xs, ys, s=28, c="#43e6ff", edgecolors="black",
            linewidths=0.3, label="Detected centers",
        )
        ax.legend(loc="lower right", frameon=True)

    ax.set_title(f"Ringgrid ({detector_name.upper()})")
    ax.set_axis_off()
    return fig


def render_heatmap(
    response: radsym.ResponseMap,
    centers: list[tuple[float, float]],
    detector_name: str,
    *,
    ellipses: list[tuple[radsym.Ellipse, str, float]] | None = None,
) -> plt.Figure:
    """Render a response heatmap with detected centers overlaid.

    Args:
        response: ResponseMap from a proposal generator.
        centers: list of (x, y) center positions.
        detector_name: label for the title.
        ellipses: optional list of (ellipse, color, linewidth) tuples to draw.
    """
    heatmap = response.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(heatmap, cmap="hot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Response")

    if ellipses:
        for ellipse, color, lw in ellipses:
            ax.add_patch(ellipse_patch(ellipse, color, lw, alpha=0.95))

    if centers:
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        ax.scatter(
            xs, ys, s=24, c="#6aff6a", edgecolors="black",
            linewidths=0.3, label="Detected centers",
        )
        ax.legend(loc="lower right", frameon=True)

    ax.set_title(f"Response heatmap ({detector_name.upper()})")
    ax.set_axis_off()
    return fig


def show_or_save(
    figures: list[tuple[plt.Figure, Path | None]],
) -> None:
    """Save figures to paths or show interactively if no paths given."""
    any_saved = False
    for fig, path in figures:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            console.print(f"[bold]Saved:[/bold] {path}")
            any_saved = True
    if not any_saved:
        plt.show()
    else:
        for fig, path in figures:
            if path is None:
                plt.close(fig)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def render_summary(
    image_path: str,
    width: int,
    height: int,
    detector_name: str,
    radii: list[int],
    raw_count: int,
    final_count: int,
    processing_ms: float,
    title: str = "Detection Summary",
) -> None:
    """Print a summary panel to the console."""
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Image", str(image_path))
    summary.add_row("Image size", f"{width}x{height}")
    summary.add_row("Detector", detector_name)
    summary.add_row("Radius band", str(radii))
    summary.add_row("Raw proposals", str(raw_count))
    summary.add_row("Final detections", str(final_count))
    summary.add_row("Processing time", f"{processing_ms:.2f} ms")
    console.print(Panel(summary, title=title, expand=False))


def render_performance(metrics: dict) -> None:
    """Print a performance breakdown table to the console."""
    perf = Table(title="radsym Performance", header_style="bold magenta")
    perf.add_column("Call", style="cyan")
    perf.add_column("Count", justify="right")
    perf.add_column("Total ms", justify="right")
    perf.add_column("Avg ms", justify="right")
    for name, stat in sorted(
        metrics.items(), key=lambda kv: kv[1]["total_ms"], reverse=True
    ):
        count = stat["count"]
        total = stat["total_ms"]
        perf.add_row(name, str(count), f"{total:.2f}", f"{total / max(1, count):.2f}")
    console.print(perf)
