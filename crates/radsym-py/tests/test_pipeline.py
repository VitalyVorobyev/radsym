"""End-to-end pipeline test: gradient -> FRST -> NMS -> score -> refine."""

import sys
from pathlib import Path

import numpy as np
import radsym

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = ROOT / "crates" / "radsym-py" / "examples"
TOOLS_DIR = ROOT / "tools"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import detect_surf_hole  # noqa: E402
import eval_surf_holes  # noqa: E402


def make_bright_disk(size: int = 64, cx: float = 32.0, cy: float = 32.0, r: float = 10.0) -> np.ndarray:
    """Create a synthetic bright disk image."""
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= r:
                image[y, x] = 255
    return image


def make_ellipse_fill(size: int, ellipse: radsym.Ellipse) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    cos_a = np.cos(ellipse.angle)
    sin_a = np.sin(ellipse.angle)
    for y in range(size):
        for x in range(size):
            dx = x - ellipse.center[0]
            dy = y - ellipse.center[1]
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            if (lx / ellipse.semi_major) ** 2 + (ly / ellipse.semi_minor) ** 2 <= 1.0:
                image[y, x] = 255
    return image


def test_full_pipeline():
    """Test the complete detection pipeline on a synthetic bright disk."""
    image = make_bright_disk()

    # Step 1: Gradient
    gradient = radsym.sobel_gradient(image)
    assert isinstance(gradient, radsym.GradientField)
    assert gradient.width == 64
    assert gradient.height == 64

    # Step 2: FRST response
    config = radsym.FrstConfig(radii=[9, 10, 11])
    response = radsym.frst_response(gradient, config)
    assert isinstance(response, radsym.ResponseMap)
    assert response.width == 64
    assert response.height == 64

    # Step 3: Extract proposals
    nms = radsym.NmsConfig(radius=5, threshold=0.0, max_detections=5)
    proposals = radsym.extract_proposals(response, nms)
    assert len(proposals) > 0
    assert isinstance(proposals[0], radsym.Proposal)

    # The best proposal should be near the disk center (32, 32)
    best = proposals[0]
    px, py = best.position
    assert abs(px - 32.0) < 8.0, f"x={px} too far from center"
    assert abs(py - 32.0) < 8.0, f"y={py} too far from center"

    # Step 4: Score
    circle = radsym.Circle(best.position, 10.0)
    score = radsym.score_circle_support(gradient, circle)
    assert isinstance(score, radsym.SupportScore)
    assert score.total > 0.0

    # Step 5: Refine circle
    result = radsym.refine_circle(gradient, circle)
    assert isinstance(result, radsym.CircleRefinementResult)
    refined = result.hypothesis
    assert isinstance(refined, radsym.Circle)
    assert abs(refined.center[0] - 32.0) < 5.0


def test_rsd_pipeline():
    """Test the RSD variant of the pipeline."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)

    config = radsym.RsdConfig(radii=[9, 10, 11])
    response = radsym.rsd_response(gradient, config)
    assert isinstance(response, radsym.ResponseMap)

    proposals = radsym.extract_proposals(response)
    assert len(proposals) > 0


def test_default_configs():
    """Test that all functions work with default (None) configs."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)

    response = radsym.frst_response(gradient)
    proposals = radsym.extract_proposals(response)
    assert len(proposals) > 0

    circle = radsym.Circle(proposals[0].position, 10.0)
    score = radsym.score_circle_support(gradient, circle)
    assert score.total >= 0.0

    result = radsym.refine_circle(gradient, circle)
    assert result.status in ("converged", "max_iterations", "degenerate", "out_of_bounds")


def test_radial_center_refine():
    """Test subpixel center refinement."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)

    result = radsym.radial_center_refine(gradient, (33.0, 31.0))
    assert isinstance(result, radsym.PointRefinementResult)

    rx, ry = result.hypothesis
    # Should refine toward (32, 32)
    assert abs(rx - 32.0) < 3.0
    assert abs(ry - 32.0) < 3.0


def test_ellipse_refine():
    """Test ellipse refinement starting from a circular guess."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)

    ellipse = radsym.Ellipse((33.0, 31.0), 10.0, 10.0, 0.0)
    result = radsym.refine_ellipse(gradient, ellipse, radsym.EllipseRefineConfig())
    assert isinstance(result, radsym.EllipseRefinementResult)
    assert isinstance(result.hypothesis, radsym.Ellipse)


def test_polarity_dark():
    """Test dark polarity detection."""
    # Dark disk on bright background
    image = np.full((64, 64), 255, dtype=np.uint8)
    for y in range(64):
        for x in range(64):
            if ((x - 32) ** 2 + (y - 32) ** 2) ** 0.5 <= 10:
                image[y, x] = 0

    gradient = radsym.sobel_gradient(image)
    config = radsym.FrstConfig(radii=[9, 10, 11], polarity="dark")
    response = radsym.frst_response(gradient, config)
    proposals = radsym.extract_proposals(response, polarity="dark")
    assert len(proposals) > 0

    best = proposals[0]
    px, py = best.position
    assert abs(px - 32.0) < 8.0
    assert abs(py - 32.0) < 8.0


def test_homography_pipeline():
    homography = radsym.Homography(
        [
            [1.10, 0.06, 16.0],
            [0.03, 0.98, 12.0],
            [0.0011, -0.0007, 1.0],
        ]
    )
    rectified_circle = radsym.Circle((64.0, 60.0), 16.0)
    ellipse = radsym.rectified_circle_to_image_ellipse(homography, rectified_circle)
    image = make_ellipse_fill(128, ellipse)
    gradient = radsym.sobel_gradient(image)
    grid = radsym.RectifiedGrid(128, 128)

    response = radsym.frst_response_homography(
        gradient,
        homography,
        grid,
        radsym.FrstConfig(radii=[15, 16, 17], polarity="bright"),
    )
    assert isinstance(response, radsym.RectifiedResponseMap)
    proposals = radsym.extract_rectified_proposals(response, homography)
    assert len(proposals) > 0
    best = proposals[0]
    assert best.rectified_circle_hint is not None
    score = radsym.score_rectified_circle_support(
        gradient,
        best.rectified_circle_hint,
        homography,
    )
    assert score.total > 0.0

    image_response = radsym.frst_response(gradient, radsym.FrstConfig(radii=[15, 16, 17], polarity="bright"))
    image_proposals = radsym.extract_proposals(image_response, radsym.NmsConfig(radius=5))
    reranked = radsym.rerank_proposals_homography(gradient, image_proposals, homography)
    assert len(reranked) > 0
    assert reranked[0].total_score >= reranked[-1].total_score

    refine_result = radsym.refine_ellipse_homography(
        gradient,
        ellipse,
        homography,
        radsym.HomographyEllipseRefineConfig(),
    )
    assert isinstance(refine_result, radsym.HomographyRefinementResult)
    assert refine_result.status in ("converged", "max_iterations", "degenerate", "out_of_bounds")
    assert refine_result.rectified_circle.radius > 0.0


def test_detect_surf_hole_cli_rejects_removed_edge_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["detect_surf_hole.py", "testdata/surf1.png", "--edge-mode", "outer"],
    )
    try:
        detect_surf_hole.parse_args()
        assert False, "should have exited on removed --edge-mode"
    except SystemExit:
        pass


def test_eval_surf_holes_cli_rejects_removed_edge_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_surf_holes.py", "--edge-mode", "best"],
    )
    try:
        eval_surf_holes.parse_args()
        assert False, "should have exited on removed --edge-mode"
    except SystemExit:
        pass
