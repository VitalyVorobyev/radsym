//! End-to-end pipeline integration tests.

use radsym::{
    detect_circles, CircleRefineConfig, DetectCirclesConfig, FrstConfig, ImageView, Polarity,
    RadialCenterConfig,
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

    let config = DetectCirclesConfig {
        frst: FrstConfig {
            radii: vec![14, 15, 16, 17, 18],
            gradient_threshold: 1.0,
            ..FrstConfig::default()
        },
        polarity: Polarity::Bright,
        radius_hint: radius,
        refinement: CircleRefineConfig {
            radial_center: RadialCenterConfig {
                patch_radius: 20,
                ..RadialCenterConfig::default()
            },
            ..CircleRefineConfig::default()
        },
        ..DetectCirclesConfig::default()
    };

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

    let config = DetectCirclesConfig {
        frst: FrstConfig {
            radii: vec![14, 15, 16, 17, 18],
            gradient_threshold: 1.0,
            ..FrstConfig::default()
        },
        polarity: Polarity::Dark,
        radius_hint: radius,
        refinement: CircleRefineConfig {
            radial_center: RadialCenterConfig {
                patch_radius: 20,
                ..RadialCenterConfig::default()
            },
            ..CircleRefineConfig::default()
        },
        ..DetectCirclesConfig::default()
    };

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
