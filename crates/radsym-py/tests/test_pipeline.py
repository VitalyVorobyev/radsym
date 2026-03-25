"""End-to-end pipeline test: gradient -> FRST -> NMS -> score -> refine."""

import numpy as np
import radsym


def make_bright_disk(size: int = 64, cx: float = 32.0, cy: float = 32.0, r: float = 10.0) -> np.ndarray:
    """Create a synthetic bright disk image."""
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 <= r:
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
    result = radsym.refine_ellipse(gradient, ellipse)
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
