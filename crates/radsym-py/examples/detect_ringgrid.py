"""Detect many ring-like structures in an input image.

Usage:
    python detect_ringgrid.py path/to/ringgrid.png [--output result.png]

The script uses the `radsym` Python bindings to:
1. load a grayscale image,
2. compute an FRST response map,
3. extract proposals,
4. refine and score circle candidates,
5. render matplotlib overlays on the source image and FRST heatmap.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.patches import Circle as MplCircle

import radsym


def build_radius_band(base_radius: float, steps: int = 5) -> list[int]:
    start = max(4, int(round(base_radius * 0.7)))
    stop = max(start + 1, int(round(base_radius * 1.35)))
    if steps <= 1 or start == stop:
        return [start]

    radii = {
        int(round(start + (stop - start) * index / (steps - 1)))
        for index in range(steps)
    }
    return sorted(r for r in radii if r > 0)


def render_overlay(image, detections: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)

    for detection in detections:
        circle = detection["circle"]
        ax.add_patch(
            MplCircle(
                xy=circle.center,
                radius=circle.radius,
                edgecolor="#ff8c1a",
                facecolor="none",
                linewidth=1.8,
            )
        )

    centers_x = [detection["circle"].center[0] for detection in detections]
    centers_y = [detection["circle"].center[1] for detection in detections]
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

    ax.set_title("Ringgrid detections")
    ax.set_axis_off()
    if centers_x:
        ax.legend(loc="lower right", frameon=True)
    return fig


def render_heatmap(response, detections: list[dict]):
    heatmap = response.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image_artist = ax.imshow(heatmap, cmap="hot")
    fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04, label="FRST response")

    for detection in detections:
        circle = detection["circle"]
        ax.add_patch(
            MplCircle(
                xy=circle.center,
                radius=circle.radius,
                edgecolor="white",
                facecolor="none",
                linewidth=1.6,
            )
        )

    centers_x = [detection["circle"].center[0] for detection in detections]
    centers_y = [detection["circle"].center[1] for detection in detections]
    if centers_x:
        ax.scatter(
            centers_x,
            centers_y,
            s=26,
            c="#6aff6a",
            edgecolors="black",
            linewidths=0.3,
            label="Detected centers",
        )

    ax.set_title("Ringgrid FRST heatmap")
    ax.set_axis_off()
    if centers_x:
        ax.legend(loc="lower right", frameon=True)
    return fig


def suppress_near_duplicates(candidates: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for candidate in candidates:
        circle = candidate["circle"]
        center_x, center_y = circle.center
        duplicate = False
        for other in kept:
            other_circle = other["circle"]
            dx = center_x - other_circle.center[0]
            dy = center_y - other_circle.center[1]
            distance = math.hypot(dx, dy)
            min_separation = 0.5 * min(circle.radius, other_circle.radius)
            if distance < max(4.0, min_separation):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect multiple ring-like structures in a grid image."
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
        default="dark",
        choices=("bright", "dark", "both"),
        help="FRST polarity mode. Default is dark for the provided ringgrid image.",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.28,
        help="Minimum support score required after refinement.",
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

    image = radsym.load_grayscale(str(image_path))
    height, width = image.shape
    min_dim = min(height, width)
    radius_hint = max(6.0, min_dim * 0.022)
    radii = build_radius_band(radius_hint)

    gradient = radsym.sobel_gradient(image)
    frst_config = radsym.FrstConfig(
        radii=radii,
        alpha=2.0,
        gradient_threshold=2.0,
        polarity=args.polarity,
        smoothing_factor=0.5,
    )
    response = radsym.frst_response(gradient, frst_config)
    proposals = radsym.extract_proposals(
        response,
        radsym.NmsConfig(
            radius=max(6, int(round(radius_hint))),
            threshold=0.01,
            max_detections=256,
        ),
        polarity=args.polarity,
    )
    if not proposals:
        raise RuntimeError("no proposals found")

    scoring_config = radsym.ScoringConfig(annulus_margin=0.25, min_samples=16)
    refinement_config = radsym.CircleRefineConfig(
        max_iterations=14,
        convergence_tol=0.05,
        annulus_margin=0.25,
    )

    refined_candidates: list[dict] = []
    for proposal in proposals:
        proposal_radius = proposal.scale_hint or radius_hint
        initial = radsym.Circle(proposal.position, proposal_radius)
        refined = radsym.refine_circle(gradient, initial, refinement_config)
        circle = refined.hypothesis
        support = radsym.score_circle_support(gradient, circle, scoring_config)

        if (
            support.total < args.min_support
            or circle.center[0] - circle.radius < 0.0
            or circle.center[1] - circle.radius < 0.0
            or circle.center[0] + circle.radius >= width
            or circle.center[1] + circle.radius >= height
        ):
            continue

        refined_candidates.append(
            {
                "proposal": proposal,
                "refined": refined,
                "circle": circle,
                "support": support,
                "rank_score": support.total,
            }
        )

    if not refined_candidates:
        raise RuntimeError("no refined detections passed the support threshold")

    refined_candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    detections = suppress_near_duplicates(refined_candidates)

    overlay_figure = render_overlay(image, detections)
    heatmap_figure = render_heatmap(response, detections)

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

    print(f"image: {image_path}")
    print(f"shape: {width}x{height}")
    print(f"radii: {radii}")
    print(f"raw proposals: {len(proposals)}")
    print(f"final detections: {len(detections)}")
    for index, detection in enumerate(detections[:10], start=1):
        circle = detection["circle"]
        support = detection["support"]
        print(
            f"{index:>2}: center=({circle.center[0]:.2f}, {circle.center[1]:.2f}) "
            f"radius={circle.radius:.2f} support={support.total:.3f}"
        )
    if len(detections) > 10:
        print(f"... {len(detections) - 10} more detections omitted")
    if overlay_path is not None:
        print(f"overlay: {overlay_path}")
    if heatmap_path is not None:
        print(f"heatmap: {heatmap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
