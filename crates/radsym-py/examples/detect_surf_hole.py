"""Detect a single central hole-like structure in a surf image.

Usage:
    python detect_surf_hole.py path/to/surf1.png [--output result.png]

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
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse as MplEllipse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import radsym

console = Console()

ELLIPSE_FIT_MIN_ALIGNMENT = 0.3
ELLIPSE_FIT_ANGULAR_SAMPLES = 64
ELLIPSE_FIT_RADIAL_SAMPLES = 9


def build_radius_band(base_radius: float, steps: int = 5) -> list[int]:
    """Return a compact radius search band around a nominal radius."""
    start = max(6, int(round(base_radius * 0.65)))
    stop = max(start + 1, int(round(base_radius * 1.35)))
    if steps <= 1 or start == stop:
        return [start]

    radii = {
        int(round(start + (stop - start) * index / (steps - 1)))
        for index in range(steps)
    }
    return sorted(r for r in radii if r > 0)


def downscale_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Downscale a grayscale image by block averaging."""
    if factor <= 1:
        return image

    height, width = image.shape
    trimmed_height = height - (height % factor)
    trimmed_width = width - (width % factor)
    if trimmed_height == 0 or trimmed_width == 0:
        raise ValueError(f"downscale factor {factor} is too large for image shape {image.shape}")

    trimmed = image[:trimmed_height, :trimmed_width]
    reduced = trimmed.reshape(
        trimmed_height // factor,
        factor,
        trimmed_width // factor,
        factor,
    ).mean(axis=(1, 3))
    return reduced.astype(np.uint8)


def upscale_point(point: tuple[float, float], factor: int) -> tuple[float, float]:
    if factor <= 1:
        return point
    offset = 0.5 * (factor - 1)
    return (point[0] * factor + offset, point[1] * factor + offset)


def upscale_ellipse(ellipse: radsym.Ellipse, factor: int) -> radsym.Ellipse:
    if factor <= 1:
        return ellipse
    return radsym.Ellipse(
        upscale_point(ellipse.center, factor),
        ellipse.semi_major * factor,
        ellipse.semi_minor * factor,
        ellipse.angle,
    )


def timed_call(metrics: dict, name: str, fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    metrics[name]["count"] += 1
    metrics[name]["total_ms"] += (perf_counter() - start) * 1000.0
    return result


def sweep_radius_at_center(
    gradient: radsym.GradientField,
    center: tuple[float, float],
    radius_hint: float,
    metrics: dict,
) -> dict:
    """Estimate radius at a fixed center via a 1D support sweep."""
    scoring_config = radsym.ScoringConfig(
        annulus_margin=0.10,
        min_samples=24,
        weight_ringness=0.75,
        weight_coverage=0.25,
    )
    radius_min = max(6.0, radius_hint * 0.35)
    radius_max = max(radius_min + 6.0, radius_hint * 1.05)

    best = None
    start = perf_counter()
    eval_count = 0
    for radius in np.arange(radius_min, radius_max + 0.5, 1.0):
        circle = radsym.Circle(center, float(radius))
        score = radsym.score_circle_support(gradient, circle, scoring_config)
        eval_count += 1
        if score.is_degenerate:
            continue

        candidate = {
            "radius": float(radius),
            "circle": circle,
            "score": score,
        }
        if best is None:
            best = candidate
            continue

        score_delta = score.total - best["score"].total
        # Prefer smaller radii when the score plateau is effectively tied.
        if score_delta > 1e-4 or (abs(score_delta) <= 1e-4 and radius < best["radius"]):
            best = candidate

    metrics["score_circle_support (radius sweep)"]["count"] += eval_count
    metrics["score_circle_support (radius sweep)"]["total_ms"] += (
        perf_counter() - start
    ) * 1000.0

    if best is None:
        fallback_circle = radsym.Circle(center, radius_hint)
        return {
            "radius": radius_hint,
            "circle": fallback_circle,
            "score": timed_call(
                metrics,
                "score_circle_support (fallback)",
                radsym.score_circle_support,
                gradient,
                fallback_circle,
            ),
        }

    return best


def sweep_ellipse_shape_at_center(
    gradient: radsym.GradientField,
    center: tuple[float, float],
    base_radius: float,
    metrics: dict,
) -> dict:
    """Estimate ellipse axes and orientation at a fixed center."""
    scoring_config = radsym.ScoringConfig(
        annulus_margin=0.10,
        min_samples=24,
        weight_ringness=0.75,
        weight_coverage=0.25,
    )
    axis_ratios = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    angle_degrees = list(range(0, 180, 15))

    best = None
    start = perf_counter()
    eval_count = 0
    for axis_ratio in axis_ratios:
        angle_candidates = [0] if axis_ratio <= 1.0 else angle_degrees
        ratio_scale = math.sqrt(axis_ratio)
        semi_major = base_radius * ratio_scale
        semi_minor = base_radius / ratio_scale

        for angle_deg in angle_candidates:
            ellipse = radsym.Ellipse(center, semi_major, semi_minor, math.radians(angle_deg))
            score = radsym.score_ellipse_support(gradient, ellipse, scoring_config)
            eval_count += 1
            if score.is_degenerate:
                continue

            candidate = {
                "ellipse": ellipse,
                "axis_ratio": axis_ratio,
                "angle_deg": angle_deg,
                "score": score,
            }
            if best is None:
                best = candidate
                continue

            score_delta = score.total - best["score"].total
            if score_delta > 1e-4 or (
                abs(score_delta) <= 1e-4 and axis_ratio < best["axis_ratio"]
            ):
                best = candidate

    metrics["score_ellipse_support (shape sweep)"]["count"] += eval_count
    metrics["score_ellipse_support (shape sweep)"]["total_ms"] += (
        perf_counter() - start
    ) * 1000.0

    if best is None:
        fallback = radsym.Ellipse(center, base_radius, base_radius, 0.0)
        return {
            "ellipse": fallback,
            "axis_ratio": 1.0,
            "angle_deg": 0.0,
            "score": timed_call(
                metrics,
                "score_ellipse_support (fallback)",
                radsym.score_ellipse_support,
                gradient,
                fallback,
            ),
        }

    return best


def collect_ellipse_fit_evidence(
    gradient: radsym.GradientField,
    ellipse: radsym.Ellipse,
    annulus_margin: float,
    min_alignment: float,
    num_angular_samples: int = 64,
    num_radial_samples: int = 9,
) -> dict:
    """Reconstruct the annulus samples used by ellipse fitting."""
    gx = gradient.gx_numpy()
    gy = gradient.gy_numpy()
    height, width = gx.shape
    cos_a = math.cos(ellipse.angle)
    sin_a = math.sin(ellipse.angle)
    inner_scale = max(0.1, 1.0 - annulus_margin)
    outer_scale = 1.0 + annulus_margin

    all_positions = []
    fit_positions = []
    fit_alignments = []

    for ai in range(num_angular_samples):
        theta = 2.0 * math.pi * ai / num_angular_samples
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        ex = ellipse.semi_major * cos_t
        ey = ellipse.semi_minor * sin_t

        for ri in range(num_radial_samples):
            if num_radial_samples <= 1:
                t = 0.5
            else:
                t = ri / (num_radial_samples - 1)
            scale = inner_scale + t * (outer_scale - inner_scale)
            lx = scale * ex
            ly = scale * ey
            sx = ellipse.center[0] + lx * cos_a - ly * sin_a
            sy = ellipse.center[1] + lx * sin_a + ly * cos_a

            ix = int(round(sx))
            iy = int(round(sy))
            if ix < 0 or ix >= width or iy < 0 or iy >= height:
                continue

            gx_value = float(gx[iy, ix])
            gy_value = float(gy[iy, ix])
            magnitude = math.hypot(gx_value, gy_value)
            if magnitude < 1e-8:
                continue

            dx = sx - ellipse.center[0]
            dy = sy - ellipse.center[1]
            distance = math.hypot(dx, dy)
            if distance < 1e-8:
                continue

            rx = dx / distance
            ry = dy / distance
            alignment = abs((gx_value * rx + gy_value * ry) / magnitude)
            all_positions.append((sx, sy))

            if alignment >= min_alignment:
                fit_positions.append((sx, sy))
                fit_alignments.append(alignment)

    if fit_positions:
        fit_xy = np.asarray(fit_positions, dtype=np.float32)
        fit_alignment_array = np.asarray(fit_alignments, dtype=np.float32)
    else:
        fit_xy = np.empty((0, 2), dtype=np.float32)
        fit_alignment_array = np.empty((0,), dtype=np.float32)

    if all_positions:
        all_xy = np.asarray(all_positions, dtype=np.float32)
    else:
        all_xy = np.empty((0, 2), dtype=np.float32)

    return {
        "all_xy": all_xy,
        "fit_xy": fit_xy,
        "fit_alignment": fit_alignment_array,
    }


def render_overlay(
    image,
    ranked: list[dict],
    downscale: int,
    best: dict,
):
    full_resolution_ellipse = best["full_resolution_ellipse"]
    fit_support = best["fit_support"]
    pre_ellipse_center = upscale_point(best["center_result"].hypothesis, downscale)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    proposals_x = [
        upscale_point(candidate["proposal"].position, downscale)[0] for candidate in ranked[:5]
    ]
    proposals_y = [
        upscale_point(candidate["proposal"].position, downscale)[1] for candidate in ranked[:5]
    ]
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

    if fit_support["all_xy"].size:
        all_xy = fit_support["all_xy"] * downscale + 0.5 * (downscale - 1)
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=9,
            c="white",
            alpha=0.12,
            linewidths=0,
            label="Annulus samples",
        )

    if fit_support["fit_xy"].size:
        fit_xy = fit_support["fit_xy"] * downscale + 0.5 * (downscale - 1)
        ax.scatter(
            fit_xy[:, 0],
            fit_xy[:, 1],
            s=15,
            c=fit_support["fit_alignment"],
            cmap="viridis",
            vmin=0.3,
            vmax=1.0,
            linewidths=0,
            alpha=0.9,
            label="Ellipse-fit samples",
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
        [pre_ellipse_center[0]],
        [pre_ellipse_center[1]],
        s=65,
        c="#ff66c4",
        marker="x",
        linewidths=1.8,
        label="Pre-ellipse center",
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


def render_heatmap(
    response,
    ranked: list[dict],
    downscale: int,
    best: dict,
):
    ellipse = best["ellipse"]
    fit_support = best["fit_support"]

    heatmap = response.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image_artist = ax.imshow(heatmap, cmap="magma")
    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04, label="FRST response")

    proposals_x = [candidate["proposal"].position[0] for candidate in ranked[:5]]
    proposals_y = [candidate["proposal"].position[1] for candidate in ranked[:5]]
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

    if fit_support["all_xy"].size:
        ax.scatter(
            fit_support["all_xy"][:, 0],
            fit_support["all_xy"][:, 1],
            s=9,
            c="white",
            alpha=0.14,
            linewidths=0,
            label="Annulus samples",
        )

    if fit_support["fit_xy"].size:
        ax.scatter(
            fit_support["fit_xy"][:, 0],
            fit_support["fit_xy"][:, 1],
            s=15,
            c=fit_support["fit_alignment"],
            cmap="viridis",
            vmin=0.3,
            vmax=1.0,
            linewidths=0,
            alpha=0.95,
            label="Ellipse-fit samples",
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
        [best["center_result"].hypothesis[0]],
        [best["center_result"].hypothesis[1]],
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


def choose_best_detection(
    image,
    gradient: radsym.GradientField,
    proposals: list[radsym.Proposal],
    radius_hint: float,
    metrics: dict,
) -> tuple[dict, list[dict]]:
    height, width = image.shape
    center_x = width * 0.5
    center_y = height * 0.5
    max_center_distance = math.hypot(center_x, center_y)

    scoring_config = radsym.ScoringConfig(annulus_margin=0.12, min_samples=32)
    refinement_annulus_margin = 0.12
    refinement_config = radsym.EllipseRefineConfig(
        max_iterations=16,
        convergence_tol=0.05,
        annulus_margin=refinement_annulus_margin,
    )
    center_config = radsym.RadialCenterConfig(
        patch_radius=max(10, int(round(radius_hint * 1.4))),
        gradient_threshold=1.0,
    )

    ranked: list[dict] = []
    for proposal in proposals[: min(len(proposals), 12)]:
        center_result = timed_call(
            metrics,
            "radial_center_refine",
            radsym.radial_center_refine,
            gradient,
            proposal.position,
            center_config,
        )
        if center_result.converged:
            center = center_result.hypothesis
        else:
            center = proposal.position

        radius_sweep = sweep_radius_at_center(gradient, center, radius_hint, metrics)
        ellipse_seed = sweep_ellipse_shape_at_center(
            gradient,
            center,
            radius_sweep["radius"],
            metrics,
        )
        refined = timed_call(
            metrics,
            "refine_ellipse",
            radsym.refine_ellipse,
            gradient,
            ellipse_seed["ellipse"],
            refinement_config,
        )
        ellipse = refined.hypothesis
        support = timed_call(
            metrics,
            "score_ellipse_support (final)",
            radsym.score_ellipse_support,
            gradient,
            ellipse,
            scoring_config,
        )

        dx = ellipse.center[0] - center_x
        dy = ellipse.center[1] - center_y
        center_distance = math.hypot(dx, dy)
        center_bonus = max(0.0, 1.0 - center_distance / max_center_distance)
        eccentricity_ratio = max(ellipse.semi_major, ellipse.semi_minor) / max(
            1e-3, min(ellipse.semi_major, ellipse.semi_minor)
        )
        shape_penalty = max(0.0, (eccentricity_ratio - 1.8) * 0.15)
        combined = 0.7 * support.total + 0.3 * center_bonus - shape_penalty

        ranked.append(
            {
                "proposal": proposal,
                "refined": refined,
                "ellipse": ellipse,
                "support": support,
                "center_distance": center_distance,
                "combined": combined,
                "radius_sweep": radius_sweep,
                "ellipse_seed": ellipse_seed,
                "center_result": center_result,
                "fit_support": collect_ellipse_fit_evidence(
                    gradient,
                    ellipse,
                    refinement_annulus_margin,
                    ELLIPSE_FIT_MIN_ALIGNMENT,
                    num_angular_samples=ELLIPSE_FIT_ANGULAR_SAMPLES,
                    num_radial_samples=ELLIPSE_FIT_RADIAL_SAMPLES,
                ),
            }
        )

    if not ranked:
        raise RuntimeError("no valid candidates remained after refinement")

    ranked.sort(key=lambda item: item["combined"], reverse=True)
    return ranked[0], ranked


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
    return parser.parse_args()


def render_summary(
    best: dict,
    width: int,
    height: int,
    working_width: int,
    working_height: int,
    downscale: int,
) -> None:
    ellipse = best["full_resolution_ellipse"]
    support = best["support"]
    ellipse_seed = best["ellipse_seed"]

    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Image size", f"{width}x{height}")
    summary.add_row("Working size", f"{working_width}x{working_height}")
    summary.add_row("Downscale", f"{downscale}x")
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
        f"{best['radius_sweep']['radius'] * downscale:.2f} px",
    )
    summary.add_row(
        "Seed ellipse ratio",
        f"{ellipse_seed['axis_ratio']:.3f} @ {ellipse_seed['angle_deg']:.0f} deg",
    )
    summary.add_row("Fit samples", str(len(best["fit_support"]["fit_xy"])))
    summary.add_row("Support", f"{support.total:.3f}")
    summary.add_row("Coverage", f"{support.angular_coverage:.3f}")
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
    if args.downscale < 1:
        raise ValueError("--downscale must be >= 1")
    overlay_path = args.output.expanduser().resolve() if args.output is not None else None
    heatmap_path = (
        args.heatmap_output.expanduser().resolve()
        if args.heatmap_output is not None
        else None
    )
    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})

    image = timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(image_path))
    working_image = downscale_image(image, args.downscale)
    height, width = image.shape
    working_height, working_width = working_image.shape
    min_dim = min(working_height, working_width)
    radius_hint = max(14.0, min_dim * 0.16)
    radii = build_radius_band(radius_hint)

    gradient = timed_call(metrics, "sobel_gradient", radsym.sobel_gradient, working_image)
    frst_config = radsym.FrstConfig(
        radii=radii,
        alpha=2.0,
        gradient_threshold=1.5,
        polarity=args.polarity,
        smoothing_factor=0.5,
    )
    response = timed_call(metrics, "frst_response", radsym.frst_response, gradient, frst_config)
    proposals = timed_call(
        metrics,
        "extract_proposals",
        radsym.extract_proposals,
        response,
        radsym.NmsConfig(
            radius=max(10, int(round(radius_hint * 0.8))),
            threshold=0.01,
            max_detections=12,
        ),
        polarity=args.polarity,
    )
    if not proposals:
        raise RuntimeError("no proposals found")

    best, ranked = choose_best_detection(
        working_image,
        gradient,
        proposals,
        radius_hint,
        metrics,
    )
    ellipse = best["ellipse"]
    full_resolution_ellipse = upscale_ellipse(ellipse, args.downscale)
    best["full_resolution_ellipse"] = full_resolution_ellipse

    overlay_figure = render_overlay(image, ranked, args.downscale, best)
    heatmap_figure = render_heatmap(response, ranked, args.downscale, best)

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

    console.print(f"[bold]image:[/bold] {image_path}")
    console.print(f"[bold]radii:[/bold] {radii}")
    console.print(f"[bold]proposals:[/bold] {len(proposals)}")
    render_summary(best, width, height, working_width, working_height, args.downscale)
    render_frst_cost(frst_config, working_width, working_height)
    render_performance(metrics)
    if overlay_path is not None:
        console.print(f"[bold]overlay:[/bold] {overlay_path}")
    if heatmap_path is not None:
        console.print(f"[bold]heatmap:[/bold] {heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
