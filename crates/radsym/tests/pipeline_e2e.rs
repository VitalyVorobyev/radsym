//! End-to-end pipeline integration tests.

use radsym::{
    CircleRefineAdvanced, CircleRefineConfig, DetectCirclesAdvanced, DetectCirclesConfig,
    FrstConfig, ImageView, Polarity, RadialCenterConfig, detect_circles,
};

/// Create a bright disk (white circle on black background).
fn make_bright_disk(size: usize, cx: f32, cy: f32, radius: f32) -> Vec<u8> {
    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= radius {
                data[y * size + x] = 255;
            }
        }
    }
    data
}

/// Create a dark disk (black circle on white background).
fn make_dark_disk(size: usize, cx: f32, cy: f32, radius: f32) -> Vec<u8> {
    let mut data = vec![255u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= radius {
                data[y * size + x] = 0;
            }
        }
    }
    data
}

#[test]
fn detect_bright_circles() {
    let size = 128;
    let cx = 64.0f32;
    let cy = 64.0f32;
    let radius = 16.0f32;
    let data = make_bright_disk(size, cx, cy, radius);
    let image = ImageView::from_slice(&data, size, size).unwrap();

    let mut radial_center = RadialCenterConfig::default();
    radial_center.patch_radius = 20;
    let mut refine_advanced = CircleRefineAdvanced::default();
    refine_advanced.radial_center = radial_center;
    let mut refinement = CircleRefineConfig::default();
    refinement.advanced = refine_advanced;
    let mut frst = FrstConfig::default();
    frst.gradient_threshold = 1.0;
    let mut advanced = DetectCirclesAdvanced::default();
    advanced.frst = frst;
    advanced.refinement = refinement;
    let mut config = DetectCirclesConfig::default();
    config.radii = vec![14, 15, 16, 17, 18];
    config.polarity = Polarity::Bright;
    config.radius_hint = radius;
    config.advanced = advanced;

    let detections = detect_circles(&image, &config).unwrap();
    assert!(
        !detections.is_empty(),
        "should detect at least one bright circle"
    );

    let mut best_err = f32::MAX;
    for d in &detections {
        let dx = d.hypothesis.center.x - cx;
        let dy = d.hypothesis.center.y - cy;
        let err = (dx * dx + dy * dy).sqrt();
        if err < best_err {
            best_err = err;
        }
    }
    assert!(
        best_err < 3.0,
        "closest detection should be within 3px of true center ({cx}, {cy}), got {best_err}"
    );
}

#[test]
fn detect_dark_circles() {
    let size = 128;
    let cx = 64.0f32;
    let cy = 64.0f32;
    let radius = 16.0f32;
    let data = make_dark_disk(size, cx, cy, radius);
    let image = ImageView::from_slice(&data, size, size).unwrap();

    let mut radial_center = RadialCenterConfig::default();
    radial_center.patch_radius = 20;
    let mut refine_advanced = CircleRefineAdvanced::default();
    refine_advanced.radial_center = radial_center;
    let mut refinement = CircleRefineConfig::default();
    refinement.advanced = refine_advanced;
    let mut frst = FrstConfig::default();
    frst.gradient_threshold = 1.0;
    let mut advanced = DetectCirclesAdvanced::default();
    advanced.frst = frst;
    advanced.refinement = refinement;
    let mut config = DetectCirclesConfig::default();
    config.radii = vec![14, 15, 16, 17, 18];
    config.polarity = Polarity::Dark;
    config.radius_hint = radius;
    config.advanced = advanced;

    let detections = detect_circles(&image, &config).unwrap();
    assert!(
        !detections.is_empty(),
        "should detect at least one dark circle"
    );

    let mut best_err = f32::MAX;
    for d in &detections {
        let dx = d.hypothesis.center.x - cx;
        let dy = d.hypothesis.center.y - cy;
        let err = (dx * dx + dy * dy).sqrt();
        if err < best_err {
            best_err = err;
        }
    }
    assert!(
        best_err < 3.0,
        "closest detection should be within 3px of true center ({cx}, {cy}), got {best_err}"
    );
}
