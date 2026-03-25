"""Test Python wrapper type construction, attributes, and repr."""

import radsym


def test_circle_construction():
    c = radsym.Circle((10.0, 20.0), 5.0)
    assert c.center == (10.0, 20.0)
    assert c.radius == 5.0
    assert "Circle" in repr(c)


def test_circle_equality():
    c1 = radsym.Circle((10.0, 20.0), 5.0)
    c2 = radsym.Circle((10.0, 20.0), 5.0)
    c3 = radsym.Circle((10.0, 20.0), 6.0)
    assert c1 == c2
    assert c1 != c3


def test_ellipse_construction():
    e = radsym.Ellipse((100.0, 200.0), 20.0, 10.0, angle=0.5)
    assert e.center == (100.0, 200.0)
    assert e.semi_major == 20.0
    assert e.semi_minor == 10.0
    assert abs(e.angle - 0.5) < 1e-6
    assert "Ellipse" in repr(e)


def test_ellipse_default_angle():
    e = radsym.Ellipse((0.0, 0.0), 10.0, 5.0)
    assert e.angle == 0.0


def test_frst_config_defaults():
    cfg = radsym.FrstConfig()
    assert cfg.radii == [3, 5, 7, 9, 11]
    assert cfg.alpha == 2.0
    assert cfg.gradient_threshold == 0.0
    assert cfg.polarity == "both"
    assert cfg.smoothing_factor == 0.5
    assert "FrstConfig" in repr(cfg)


def test_frst_config_custom():
    cfg = radsym.FrstConfig(radii=[5, 10], alpha=3.0, polarity="dark")
    assert cfg.radii == [5, 10]
    assert cfg.alpha == 3.0
    assert cfg.polarity == "dark"


def test_frst_config_invalid_polarity():
    try:
        radsym.FrstConfig(polarity="invalid")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_nms_config():
    nms = radsym.NmsConfig(radius=10, threshold=0.5, max_detections=50)
    assert nms.radius == 10
    assert nms.threshold == 0.5
    assert nms.max_detections == 50
    assert "NmsConfig" in repr(nms)


def test_nms_config_defaults():
    nms = radsym.NmsConfig()
    assert nms.radius == 5
    assert nms.threshold == 0.0
    assert nms.max_detections == 1000


def test_scoring_config():
    cfg = radsym.ScoringConfig(annulus_margin=0.5)
    assert "ScoringConfig" in repr(cfg)


def test_circle_refine_config():
    cfg = radsym.CircleRefineConfig(max_iterations=20, convergence_tol=0.05)
    assert "CircleRefineConfig" in repr(cfg)


def test_ellipse_refine_config():
    cfg = radsym.EllipseRefineConfig()
    assert "EllipseRefineConfig" in repr(cfg)


def test_radial_center_config():
    cfg = radsym.RadialCenterConfig(patch_radius=15)
    assert "RadialCenterConfig" in repr(cfg)
