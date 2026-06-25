//! Integration tests for multi-bit-depth ingestion (`SourcePixel`) and the
//! up-front config validation added in the production-hardening pass.

use radsym::{
    Circle, DetectCirclesConfig, ImageView, Polarity, RadSymError, Rect, detect_circles,
    score_circle_support, sobel_gradient,
};

/// Paint a bright disk into an existing buffer.
fn paint_disk(data: &mut [u8], size: usize, cx: f32, cy: f32, r: f32) {
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= r {
                data[y * size + x] = 255;
            }
        }
    }
}

/// Bright disk as an 8-bit buffer (0 / 255).
fn disk_u8(size: usize, cx: f32, cy: f32, r: f32) -> Vec<u8> {
    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= r {
                data[y * size + x] = 255;
            }
        }
    }
    data
}

fn detect_config() -> DetectCirclesConfig {
    let mut cfg = DetectCirclesConfig::for_radii([10, 12, 14])
        .polarity(Polarity::Bright)
        .radius_hint(12.0);
    // A low absolute threshold passes for every bit depth (u8 edge ~127,
    // u16 edge ~32767), isolating the ingestion path from threshold tuning.
    cfg.advanced.frst.gradient_threshold = 1.0;
    cfg
}

#[test]
fn detect_circles_agrees_across_u8_u16_f32() {
    let size = 96;
    let (cx, cy, r) = (48.0f32, 48.0f32, 12.0f32);
    let u8data = disk_u8(size, cx, cy, r);
    // 255 * 257 = 65535 — exercise the full 16-bit range.
    let u16data: Vec<u16> = u8data.iter().map(|&v| v as u16 * 257).collect();
    let f32data: Vec<f32> = u8data.iter().map(|&v| v as f32).collect();

    let cfg = detect_config();
    let u8v = ImageView::from_slice(&u8data, size, size).unwrap();
    let u16v = ImageView::from_slice(&u16data, size, size).unwrap();
    let f32v = ImageView::from_slice(&f32data, size, size).unwrap();

    let d8 = detect_circles(&u8v, &cfg).unwrap();
    let d16 = detect_circles(&u16v, &cfg).unwrap();
    let df = detect_circles(&f32v, &cfg).unwrap();

    for (label, det) in [("u8", &d8), ("u16", &d16), ("f32", &df)] {
        assert!(!det.is_empty(), "{label}: expected a detection");
        let c = det[0].hypothesis.center;
        assert!(
            (c.x - cx).abs() < 3.0 && (c.y - cy).abs() < 3.0,
            "{label}: center ({}, {}) not near ({cx}, {cy})",
            c.x,
            c.y
        );
    }

    // u8 and f32 carry identical intensities, so the centers must match closely.
    assert!((d8[0].hypothesis.center.x - df[0].hypothesis.center.x).abs() < 1e-3);
    assert!((d8[0].hypothesis.center.y - df[0].hypothesis.center.y).abs() < 1e-3);
}

#[test]
fn u16_gradient_scales_with_bit_depth() {
    // Sanity check that u16 ingestion actually reads the high bits: a 16-bit
    // edge produces a ~257x larger gradient than the same 8-bit edge.
    let size = 16;
    let u8data = disk_u8(size, 8.0, 8.0, 4.0);
    let u16data: Vec<u16> = u8data.iter().map(|&v| v as u16 * 257).collect();
    let g8 = sobel_gradient(&ImageView::from_slice(&u8data, size, size).unwrap()).unwrap();
    let g16 = sobel_gradient(&ImageView::from_slice(&u16data, size, size).unwrap()).unwrap();
    let ratio = g16.max_magnitude() / g8.max_magnitude();
    assert!(
        (ratio - 257.0).abs() < 1.0,
        "u16/u8 gradient ratio {ratio} should be ~257"
    );
}

#[test]
fn invalid_nms_config_errors_instead_of_silent_empty() {
    let u8data = disk_u8(64, 32.0, 32.0, 10.0);
    let view = ImageView::from_slice(&u8data, 64, 64).unwrap();

    let mut cfg = DetectCirclesConfig::for_radii([9, 10, 11]).polarity(Polarity::Bright);
    cfg.advanced.nms.max_detections = 0; // previously returned [] silently

    let result = detect_circles(&view, &cfg);
    assert!(
        matches!(result, Err(RadSymError::InvalidConfig { .. })),
        "max_detections = 0 should be a hard config error, got {result:?}"
    );
}

#[test]
fn invalid_scoring_config_errors() {
    let u8data = disk_u8(64, 32.0, 32.0, 10.0);
    let view = ImageView::from_slice(&u8data, 64, 64).unwrap();

    let mut cfg = DetectCirclesConfig::for_radii([9, 10, 11]).polarity(Polarity::Bright);
    cfg.advanced.scoring.annulus_margin = 0.0; // degenerate ring

    let result = detect_circles(&view, &cfg);
    assert!(
        matches!(result, Err(RadSymError::InvalidConfig { .. })),
        "annulus_margin = 0 should be a hard config error, got {result:?}"
    );
}

#[test]
fn f32_scoring_path_still_available() {
    // The composable f32 path remains usable for pre-normalized imagery.
    let size = 64;
    let f32data: Vec<f32> = disk_u8(size, 32.0, 32.0, 10.0)
        .iter()
        .map(|&v| v as f32)
        .collect();
    let grad = sobel_gradient(&ImageView::from_slice(&f32data, size, size).unwrap()).unwrap();
    let circle = Circle::new(radsym::PixelCoord::new(32.0, 32.0), 10.0);
    let score = score_circle_support(&grad, &circle, &Default::default());
    assert!(score.total > 0.0);
}

#[test]
fn roi_restricts_search_and_restores_full_frame_coordinates() {
    // Two disks: A near the top-left, B near the bottom-right. An ROI around B
    // must yield exactly one detection at B's *full-frame* coordinates.
    let size = 128;
    let mut data = vec![0u8; size * size];
    paint_disk(&mut data, size, 30.0, 30.0, 10.0); // disk A
    paint_disk(&mut data, size, 96.0, 96.0, 10.0); // disk B
    let view = ImageView::from_slice(&data, size, size).unwrap();

    let cfg = DetectCirclesConfig::for_radii([9, 10, 11])
        .polarity(Polarity::Bright)
        .radius_hint(10.0)
        .roi(Rect::new(70, 70, 50, 50)); // window around disk B only

    let dets = detect_circles(&view, &cfg).unwrap();
    assert!(!dets.is_empty(), "ROI should still detect disk B");
    let c = dets[0].hypothesis.center;
    // Center is translated back to full-frame coordinates near disk B (96, 96)…
    assert!(
        (c.x - 96.0).abs() < 4.0 && (c.y - 96.0).abs() < 4.0,
        "ROI center ({}, {}) should be near disk B (96, 96) in full-frame coords",
        c.x,
        c.y
    );
    // …and disk A (outside the ROI) must not appear.
    for d in &dets {
        assert!(
            (d.hypothesis.center.x - 30.0).abs() > 15.0
                || (d.hypothesis.center.y - 30.0).abs() > 15.0,
            "disk A should be outside the ROI"
        );
    }
}

#[test]
fn mixed_size_circles_detected_via_scale_hint() {
    // Two disks of very different radii (8 and 24) in one frame. The pipeline
    // now scores/refines each proposal at its own winning radius, so a single
    // detect_circles call recovers both — and each reported radius matches its
    // own disk, not the global radius_hint.
    let size = 176;
    let mut data = vec![0u8; size * size];
    paint_disk(&mut data, size, 44.0, 88.0, 8.0); // small disk
    paint_disk(&mut data, size, 120.0, 88.0, 24.0); // large disk
    let view = ImageView::from_slice(&data, size, size).unwrap();

    let mut cfg = DetectCirclesConfig::for_radii([6, 8, 10, 22, 24, 26])
        .polarity(Polarity::Bright)
        .radius_hint(8.0) // deliberately the *small* radius
        .min_score(0.15);
    cfg.advanced.frst.gradient_threshold = 1.0;

    let dets = detect_circles(&view, &cfg).unwrap();

    let near = |d: &radsym::CircleDetection, cx: f32, cy: f32| {
        (d.hypothesis.center.x - cx).abs() < 5.0 && (d.hypothesis.center.y - cy).abs() < 5.0
    };
    let small = dets.iter().find(|d| near(d, 44.0, 88.0));
    let large = dets.iter().find(|d| near(d, 120.0, 88.0));

    let summary: Vec<_> = dets
        .iter()
        .map(|d| {
            (
                d.hypothesis.center.x,
                d.hypothesis.center.y,
                d.hypothesis.radius,
            )
        })
        .collect();
    assert!(
        small.is_some(),
        "small disk missing; detections: {summary:?}"
    );
    assert!(
        large.is_some(),
        "large disk missing; detections: {summary:?}"
    );

    // The reported radii reflect each proposal's own scale, not radius_hint = 8.
    assert!(
        large.unwrap().hypothesis.radius > 18.0,
        "large disk radius {} should be ~24",
        large.unwrap().hypothesis.radius
    );
    assert!(
        small.unwrap().hypothesis.radius < 14.0,
        "small disk radius {} should be ~8",
        small.unwrap().hypothesis.radius
    );
}

#[test]
fn out_of_bounds_roi_errors() {
    let data = vec![0u8; 64 * 64];
    let view = ImageView::from_slice(&data, 64, 64).unwrap();
    let cfg = DetectCirclesConfig::for_radii([9, 10, 11])
        .polarity(Polarity::Bright)
        .roi(Rect::new(40, 40, 40, 40)); // extends past the 64×64 image

    let result = detect_circles(&view, &cfg);
    assert!(
        matches!(result, Err(RadSymError::InvalidDimensions { .. })),
        "out-of-bounds ROI should error, got {result:?}"
    );
}
