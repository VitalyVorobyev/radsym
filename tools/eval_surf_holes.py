#!/usr/bin/env python3
"""Evaluate surf-hole detection against labeled SVG ellipses.

Run from the repo root with the Python bindings installed into `.venv`, e.g.:

    ./.venv/bin/python tools/eval_surf_holes.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import radsym

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "crates" / "radsym-py" / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import _surf_hole_common as surf_hole  # noqa: E402

SVG_NS = {
    "svg": "http://www.w3.org/2000/svg",
    "s": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
}


@dataclass(frozen=True)
class EllipseLabel:
    center: tuple[float, float]
    semi_major: float
    semi_minor: float
    angle: float


@dataclass(frozen=True)
class EvalMetrics:
    center_error: float
    major_rel_error: float
    minor_rel_error: float
    angle_error_deg: float
    iou: float
    catastrophic: bool


def wrap_half_turn(angle: float) -> float:
    while angle <= -math.pi / 2.0:
        angle += math.pi
    while angle > math.pi / 2.0:
        angle -= math.pi
    return angle


def normalize_ellipse(
    center: tuple[float, float],
    semi_major: float,
    semi_minor: float,
    angle: float,
) -> EllipseLabel:
    if semi_minor > semi_major:
        semi_major, semi_minor = semi_minor, semi_major
        angle += math.pi / 2.0
    return EllipseLabel(center, semi_major, semi_minor, wrap_half_turn(angle))


def parse_label(svg_path: Path, image_shape: tuple[int, int]) -> EllipseLabel:
    root = ET.parse(svg_path).getroot()
    path = root.find(".//svg:path", SVG_NS)
    if path is None:
        raise RuntimeError(f"no ellipse path found in {svg_path}")

    view_box = root.attrib.get("viewBox")
    if view_box:
        _, _, view_w, view_h = map(float, view_box.split())
    else:
        view_w = float(root.attrib["width"])
        view_h = float(root.attrib["height"])

    image_h, image_w = image_shape
    scale_x = image_w / view_w
    scale_y = image_h / view_h
    if abs(scale_x - scale_y) > 1e-4:
        raise RuntimeError(
            f"non-uniform SVG scaling is not supported for {svg_path}: {scale_x} vs {scale_y}"
        )
    scale = 0.5 * (scale_x + scale_y)

    angle_deg = 0.0
    transform = path.attrib.get("transform")
    if transform:
        if not transform.startswith("rotate(") or not transform.endswith(")"):
            raise RuntimeError(f"unsupported transform in {svg_path}: {transform}")
        angle_deg = float(transform[len("rotate(") : -1])

    angle = math.radians(angle_deg)
    cx = float(path.attrib["{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}cx"])
    cy = float(path.attrib["{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}cy"])
    rx = float(path.attrib["{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}rx"])
    ry = float(path.attrib["{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}ry"])

    rotated_x = cx * math.cos(angle) - cy * math.sin(angle)
    rotated_y = cx * math.sin(angle) + cy * math.cos(angle)
    return normalize_ellipse(
        (rotated_x * scale, rotated_y * scale),
        rx * scale,
        ry * scale,
        angle,
    )


def to_label(ellipse: radsym.Ellipse) -> EllipseLabel:
    return normalize_ellipse(ellipse.center, ellipse.semi_major, ellipse.semi_minor, ellipse.angle)


def angle_error_deg(pred: float, target: float) -> float:
    delta = math.degrees(pred - target)
    delta = (delta + 90.0) % 180.0 - 90.0
    return abs(delta)


def ellipse_mask(ellipse: EllipseLabel, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    dx = xx - ellipse.center[0]
    dy = yy - ellipse.center[1]
    cos_a = math.cos(ellipse.angle)
    sin_a = math.sin(ellipse.angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    return (lx / ellipse.semi_major) ** 2 + (ly / ellipse.semi_minor) ** 2 <= 1.0


def ellipse_iou(pred: EllipseLabel, target: EllipseLabel) -> float:
    min_x = math.floor(
        min(
            pred.center[0] - pred.semi_major,
            target.center[0] - target.semi_major,
        )
        - 4.0
    )
    max_x = math.ceil(
        max(
            pred.center[0] + pred.semi_major,
            target.center[0] + target.semi_major,
        )
        + 4.0
    )
    min_y = math.floor(
        min(
            pred.center[1] - pred.semi_major,
            target.center[1] - target.semi_major,
        )
        - 4.0
    )
    max_y = math.ceil(
        max(
            pred.center[1] + pred.semi_major,
            target.center[1] + target.semi_major,
        )
        + 4.0
    )

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    pred_mask = ellipse_mask(pred, xx, yy)
    target_mask = ellipse_mask(target, xx, yy)
    intersection = np.count_nonzero(pred_mask & target_mask)
    union = np.count_nonzero(pred_mask | target_mask)
    return float(intersection / max(1, union))


def bbox_contains_with_margin(pred: EllipseLabel, target: EllipseLabel, margin: float) -> bool:
    return (
        target.center[0] - target.semi_major - margin
        <= pred.center[0]
        <= target.center[0] + target.semi_major + margin
        and target.center[1] - target.semi_major - margin
        <= pred.center[1]
        <= target.center[1] + target.semi_major + margin
    )


def evaluate_prediction(pred: EllipseLabel, target: EllipseLabel) -> EvalMetrics:
    center_error = math.hypot(pred.center[0] - target.center[0], pred.center[1] - target.center[1])
    major_rel_error = abs(pred.semi_major - target.semi_major) / max(1e-6, target.semi_major)
    minor_rel_error = abs(pred.semi_minor - target.semi_minor) / max(1e-6, target.semi_minor)
    angle_err = angle_error_deg(pred.angle, target.angle)
    iou = ellipse_iou(pred, target)
    catastrophic = (
        center_error > 80.0
        or pred.semi_minor < 0.5 * target.semi_minor
        or not bbox_contains_with_margin(pred, target, 32.0)
    )
    return EvalMetrics(center_error, major_rel_error, minor_rel_error, angle_err, iou, catastrophic)


def write_debug_overlay(image: np.ndarray, pred: EllipseLabel, target: EllipseLabel, output_path: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse as MplEllipse

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.add_patch(
        MplEllipse(
            pred.center,
            2.0 * pred.semi_major,
            2.0 * pred.semi_minor,
            angle=math.degrees(pred.angle),
            edgecolor="#ff7f0e",
            facecolor="none",
            linewidth=2.2,
            label="prediction",
        )
    )
    ax.add_patch(
        MplEllipse(
            target.center,
            2.0 * target.semi_major,
            2.0 * target.semi_minor,
            angle=math.degrees(target.angle),
            edgecolor="#00e5ff",
            facecolor="none",
            linewidth=2.2,
            linestyle="--",
            label="label",
        )
    )
    ax.set_axis_off()
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_homography_sidecar(path: Path) -> tuple[radsym.Homography, radsym.RectifiedGrid]:
    payload = json.loads(path.read_text())
    matrix = payload.get("image_to_rectified", payload)
    grid_payload = payload.get("grid")
    homography = radsym.Homography(matrix)
    if grid_payload is None:
        grid = radsym.RectifiedGrid(1024, 1024)
    elif isinstance(grid_payload, dict):
        grid = radsym.RectifiedGrid(int(grid_payload["width"]), int(grid_payload["height"]))
    else:
        grid = radsym.RectifiedGrid(int(grid_payload[0]), int(grid_payload[1]))
    return homography, grid


def run_detector(
    image_path: Path,
    downscale: int,
    polarity: str,
) -> tuple[radsym.Ellipse, dict, np.ndarray]:
    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    image = surf_hole.timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(image_path))
    working_image = surf_hole.downscale_image(image, downscale)
    working_h, working_w = working_image.shape
    radius_hint = max(14.0, min(working_h, working_w) * 0.16)
    radii = surf_hole.build_radius_band(radius_hint)

    gradient = surf_hole.timed_call(metrics, "sobel_gradient", radsym.sobel_gradient, working_image)
    response = surf_hole.timed_call(
        metrics,
        "frst_response",
        radsym.frst_response,
        gradient,
        radsym.FrstConfig(
            radii=radii,
            alpha=2.0,
            gradient_threshold=1.5,
            polarity=polarity,
            smoothing_factor=0.5,
        ),
    )
    proposals = surf_hole.timed_call(
        metrics,
        "extract_proposals",
        radsym.extract_proposals,
        response,
        radsym.NmsConfig(
            radius=max(10, int(round(radius_hint * 0.8))),
            threshold=0.01,
            max_detections=12,
        ),
        polarity=polarity,
    )
    best, _ = surf_hole.choose_best_detection(working_image, gradient, proposals, radius_hint, metrics)
    return surf_hole.upscale_ellipse(best["ellipse"], downscale), best, image


def run_detector_homography(
    image_path: Path,
    homography_path: Path,
    mode: str,
    polarity: str,
) -> tuple[radsym.Ellipse, dict, np.ndarray]:
    metrics = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    image = surf_hole.timed_call(metrics, "load_grayscale", radsym.load_grayscale, str(image_path))
    homography, grid = load_homography_sidecar(homography_path)
    gradient = surf_hole.timed_call(metrics, "sobel_gradient", radsym.sobel_gradient, image)

    if mode == "rerank":
        radius_hint = max(18.0, min(image.shape) * 0.16)
        response = surf_hole.timed_call(
            metrics,
            "frst_response",
            radsym.frst_response,
            gradient,
            radsym.FrstConfig(
                radii=surf_hole.build_radius_band(radius_hint),
                alpha=2.0,
                gradient_threshold=1.5,
                polarity=polarity,
                smoothing_factor=0.5,
            ),
        )
        proposals = surf_hole.timed_call(
            metrics,
            "extract_proposals",
            radsym.extract_proposals,
            response,
            radsym.NmsConfig(radius=max(10, int(round(radius_hint * 0.8))), threshold=0.01, max_detections=12),
            polarity=polarity,
        )
        reranked = surf_hole.timed_call(
            metrics,
            "rerank_proposals_homography",
            radsym.rerank_proposals_homography,
            gradient,
            proposals,
            homography,
        )
        best = reranked[0]
        if best.image_ellipse_hint is not None:
            initial = best.image_ellipse_hint
        elif best.rectified_circle_hint is not None:
            initial = radsym.rectified_circle_to_image_ellipse(homography, best.rectified_circle_hint)
        else:
            initial = radsym.Ellipse(best.proposal.position, radius_hint, radius_hint, 0.0)
    else:
        response = surf_hole.timed_call(
            metrics,
            "frst_response_homography",
            radsym.frst_response_homography,
            gradient,
            homography,
            grid,
            radsym.FrstConfig(
                radii=[12, 16, 20, 24, 28, 32],
                alpha=2.0,
                gradient_threshold=1.5,
                polarity=polarity,
                smoothing_factor=0.5,
            ),
        )
        proposals = surf_hole.timed_call(
            metrics,
            "extract_rectified_proposals",
            radsym.extract_rectified_proposals,
            response,
            homography,
            radsym.NmsConfig(radius=8, threshold=0.01, max_detections=12),
            polarity=polarity,
        )
        best = proposals[0]
        if best.image_ellipse_hint is not None:
            initial = best.image_ellipse_hint
        elif best.rectified_circle_hint is not None:
            initial = radsym.rectified_circle_to_image_ellipse(homography, best.rectified_circle_hint)
        else:
            raise RuntimeError(f"no ellipse hint available from homography proposals for {image_path}")

    result = surf_hole.timed_call(
        metrics,
        "refine_ellipse_homography",
        radsym.refine_ellipse_homography,
        gradient,
        initial,
        homography,
        radsym.HomographyEllipseRefineConfig(),
    )
    return result.image_ellipse, {"homography_mode": mode, "result": result}, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate surf-hole detection against labeled ellipses.")
    parser.add_argument("--downscale", type=int, default=8, help="Working-image downscale factor.")
    parser.add_argument(
        "--mode",
        default="image",
        choices=("image", "rerank", "homography_frst"),
        help="Detection mode. Homography modes require sidecars.",
    )
    parser.add_argument(
        "--polarity",
        default="bright",
        choices=("bright", "dark", "both"),
        help="FRST polarity to use for all surf images.",
    )
    parser.add_argument(
        "--homography-dir",
        type=Path,
        help="Directory containing surfN.homography.json sidecars for homography-aware modes.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Optional directory for predicted-vs-labeled overlays of failing cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[tuple[str, EvalMetrics]] = []

    for index in range(1, 9):
        image_path = ROOT / "testdata" / f"surf{index}.png"
        svg_path = ROOT / "testdata" / f"surf{index}.svg"
        if args.mode == "image":
            predicted, best, image = run_detector(image_path, args.downscale, args.polarity)
        else:
            if args.homography_dir is None:
                raise RuntimeError("--homography-dir is required for homography-aware modes")
            homography_path = args.homography_dir / f"surf{index}.homography.json"
            predicted, best, image = run_detector_homography(
                image_path,
                homography_path,
                args.mode,
                args.polarity,
            )
        pred = to_label(predicted)
        label = parse_label(svg_path, image.shape)
        metrics = evaluate_prediction(pred, label)
        results.append((image_path.name, metrics))

        status = "CATASTROPHIC" if metrics.catastrophic else "ok"
        print(
            f"{image_path.name}: center={metrics.center_error:6.1f}px "
            f"major={metrics.major_rel_error:5.2%} minor={metrics.minor_rel_error:5.2%} "
            f"angle={metrics.angle_error_deg:5.1f}deg IoU={metrics.iou:0.3f} {status}"
        )

        if args.debug_dir is not None and (metrics.catastrophic or metrics.iou < 0.65):
            write_debug_overlay(image, pred, label, args.debug_dir / f"{image_path.stem}_overlay.png")

    center_errors = [metrics.center_error for _, metrics in results]
    ious = [metrics.iou for _, metrics in results]
    catastrophic_count = sum(1 for _, metrics in results if metrics.catastrophic)
    median_center = statistics.median(center_errors)
    worst_center = max(center_errors)
    median_iou = statistics.median(ious)

    print(
        "\nSummary: "
        f"median_center={median_center:.1f}px "
        f"worst_center={worst_center:.1f}px "
        f"median_iou={median_iou:.3f} "
        f"catastrophic={catastrophic_count}"
    )

    passed = catastrophic_count == 0 and median_center <= 20.0 and worst_center <= 50.0 and median_iou >= 0.65
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
