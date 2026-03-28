from __future__ import annotations

import math
from time import perf_counter
from types import SimpleNamespace

import numpy as np

import radsym


def build_radius_band(base_radius: float, steps: int = 5) -> list[int]:
    start = max(6, int(round(base_radius * 0.65)))
    stop = max(start + 1, int(round(base_radius * 1.35)))
    if steps <= 1 or start == stop:
        return [start]

    radii = {
        int(round(start + (stop - start) * index / (steps - 1)))
        for index in range(steps)
    }
    return sorted(radius for radius in radii if radius > 0)


def downscale_image(image: np.ndarray, factor: int) -> np.ndarray:
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


def choose_best_detection(
    image: np.ndarray,
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
    refinement_config = radsym.EllipseRefineConfig(
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

    ranked: list[dict] = []
    for proposal in proposals[: min(len(proposals), 12)]:
        center_result = SimpleNamespace(
            hypothesis=proposal.position,
            residual=0.0,
            iterations=0,
            status="seed",
            converged=False,
        )
        center = proposal.position

        radius_sweep = sweep_radius_at_center(gradient, center, radius_hint, metrics)
        seed_radius = max(radius_sweep["radius"], 0.8 * radius_hint)
        ellipse_seed = {
            "ellipse": radsym.Ellipse(center, seed_radius, seed_radius, 0.0),
            "axis_ratio": 1.0,
            "angle_deg": 0.0,
            "score": radius_sweep["score"],
        }
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
        mean_radius = 0.5 * (ellipse.semi_major + ellipse.semi_minor)
        radius_error = abs(mean_radius - radius_sweep["radius"])
        size_sigma = max(1.0, 0.22 * radius_sweep["radius"])
        size_consistency = math.exp(-0.5 * (radius_error / size_sigma) ** 2)
        fit_quality = 1.0 / (1.0 + 12.0 * max(0.0, refined.residual))
        combined = (
            fit_quality
            * support.angular_coverage
            * (0.6 + 0.4 * support.ringness)
            * size_consistency
            * (0.85 + 0.15 * center_bonus)
        )

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
            }
        )

    if not ranked:
        raise RuntimeError("no valid candidates remained after refinement")

    ranked.sort(key=lambda item: item["combined"], reverse=True)
    return ranked[0], ranked
