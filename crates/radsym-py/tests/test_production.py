"""Tests for the production-hardening binding features: ergonomic detect_circles,
multi-dtype input (uint8/uint16/float32), ROI, and clear error reporting."""

import numpy as np
import pytest
import radsym


def disk(size=96, cx=48, cy=48, r=12, dtype=np.uint8, hi=255):
    img = np.zeros((size, size), dtype=dtype)
    yy, xx = np.ogrid[:size, :size]
    img[((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r] = hi
    return img


def test_detect_circles_ergonomic_radii():
    """detect_circles is drivable with a top-level radii= kwarg (no FrstConfig)."""
    cfg = radsym.DetectCirclesConfig(radii=[10, 12, 14], polarity="bright", radius_hint=12.0)
    assert cfg.radii == [10, 12, 14]
    dets = radsym.detect_circles(disk(), cfg)
    assert len(dets) >= 1
    cx, cy = dets[0].hypothesis.center
    assert abs(cx - 48) < 3 and abs(cy - 48) < 3


def test_detect_circles_default_config():
    """detect_circles works with no config and finds a default-radii disk."""
    img = disk(r=10)
    dets = radsym.detect_circles(img, radsym.DetectCirclesConfig(polarity="bright"))
    assert isinstance(dets, list)


def test_detect_and_gradient_accept_uint16_and_float32():
    cfg = radsym.DetectCirclesConfig(radii=[10, 12, 14], polarity="bright", radius_hint=12.0)
    base = disk()  # uint8
    variants = {
        "uint8": base,
        "uint16": base.astype(np.uint16) * 257,  # full 16-bit range
        "float32": base.astype(np.float32),
    }
    for name, arr in variants.items():
        g = radsym.sobel_gradient(arr)
        assert g.width == 96, f"gradient failed for {name}"
        dets = radsym.detect_circles(arr, cfg)
        assert len(dets) >= 1, f"{name} should detect the disk"
        cx, cy = dets[0].hypothesis.center
        assert abs(cx - 48) < 3 and abs(cy - 48) < 3, f"{name} center off"


def test_roi_restricts_and_restores_full_frame_coords():
    size = 128
    img = np.zeros((size, size), np.uint8)
    yy, xx = np.ogrid[:size, :size]
    img[((xx - 30) ** 2 + (yy - 30) ** 2) <= 100] = 255  # disk A
    img[((xx - 96) ** 2 + (yy - 96) ** 2) <= 100] = 255  # disk B

    cfg = radsym.DetectCirclesConfig(
        radii=[9, 10, 11], polarity="bright", radius_hint=10.0, roi=(70, 70, 50, 50)
    )
    assert cfg.roi == (70, 70, 50, 50)
    dets = radsym.detect_circles(img, cfg)
    assert len(dets) >= 1
    cx, cy = dets[0].hypothesis.center
    # Full-frame coordinates near disk B, not disk A.
    assert abs(cx - 96) < 5 and abs(cy - 96) < 5


def test_unsupported_dtype_raises_clear_error():
    cfg = radsym.DetectCirclesConfig(radii=[9, 10, 11])
    with pytest.raises(ValueError, match="uint8"):
        radsym.detect_circles(np.zeros((16, 16), np.int32), cfg)
    with pytest.raises(ValueError, match="uint8"):
        radsym.sobel_gradient(np.zeros((16, 16), np.int64))


def test_invalid_config_raises_actionable_error():
    img = disk()
    cfg = radsym.DetectCirclesConfig(
        radii=[10, 12], nms=radsym.NmsConfig(radius=5, max_detections=0)
    )
    with pytest.raises(ValueError):
        radsym.detect_circles(img, cfg)
