"""Test diagnostic visualization: heatmaps, overlays, numpy conversion."""

import numpy as np
import radsym


def make_bright_disk(size: int = 64) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x - 32) ** 2 + (y - 32) ** 2) ** 0.5 <= 10:
                image[y, x] = 255
    return image


def test_response_heatmap():
    """Test heatmap generation from a response map."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)

    heatmap = radsym.response_heatmap(response, colormap="hot")
    assert isinstance(heatmap, radsym.DiagnosticImage)
    assert heatmap.width == 64
    assert heatmap.height == 64


def test_heatmap_colormaps():
    """Test all available colormaps."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)

    for cmap in ("jet", "hot", "magma"):
        hm = radsym.response_heatmap(response, colormap=cmap)
        assert hm.width == 64


def test_heatmap_invalid_colormap():
    """Test that invalid colormap raises ValueError."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)

    try:
        radsym.response_heatmap(response, colormap="invalid")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_diagnostic_to_numpy():
    """Test converting a DiagnosticImage to a numpy array."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)
    heatmap = radsym.response_heatmap(response)

    arr = heatmap.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (64, 64, 4)  # H x W x RGBA


def test_overlay_circle():
    """Test drawing a circle on a diagnostic image."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)
    heatmap = radsym.response_heatmap(response)

    circle = radsym.Circle((32.0, 32.0), 10.0)
    radsym.overlay_circle(heatmap, circle, color=(255, 0, 0, 255))

    arr = heatmap.to_numpy()
    # Check that some red pixels exist near the circle edge
    right_edge = arr[32, 42]  # (x=42, y=32) is on the right of the circle
    assert right_edge[0] == 255  # red channel


def test_overlay_ellipse():
    """Test drawing an ellipse on a diagnostic image."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)
    heatmap = radsym.response_heatmap(response)

    ellipse = radsym.Ellipse((32.0, 32.0), 20.0, 10.0)
    radsym.overlay_ellipse(heatmap, ellipse, color=(0, 255, 0, 255))
    assert "DiagnosticImage" in repr(heatmap)


def test_response_to_numpy():
    """Test converting a ResponseMap to numpy."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)
    response = radsym.frst_response(gradient)

    arr = response.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (64, 64)


def test_gradient_to_numpy():
    """Test extracting gradient components as numpy arrays."""
    image = make_bright_disk()
    gradient = radsym.sobel_gradient(image)

    gx = gradient.gx_numpy()
    gy = gradient.gy_numpy()
    assert gx.shape == (64, 64)
    assert gy.shape == (64, 64)
    assert gx.dtype == np.float32
    assert gy.dtype == np.float32
