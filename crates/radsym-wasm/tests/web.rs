use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use radsym_wasm::RadSymProcessor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic grayscale image with a grid of dark rings on a bright
/// background (similar to the real ringgrid.png calibration target).
/// Returns (width, height, grayscale_pixels).
fn synthetic_ringgrid() -> (usize, usize, Vec<u8>) {
    let w: usize = 256;
    let h: usize = 256;
    let mut data = vec![220u8; w * h]; // bright background

    // Place dark rings in a 4x4 grid
    let outer_r: f32 = 18.0;
    let inner_r: f32 = 10.0;
    let spacing: f32 = 56.0;
    let x0: f32 = 36.0;
    let y0: f32 = 36.0;

    for row in 0..4 {
        for col in 0..4 {
            let cx = x0 + col as f32 * spacing;
            let cy = y0 + row as f32 * spacing;
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= outer_r && dist >= inner_r {
                        data[y * w + x] = 30; // dark ring
                    }
                }
            }
        }
    }
    (w, h, data)
}

/// Convert grayscale pixels to RGBA (R=G=B=gray, A=255).
fn gray_to_rgba(gray: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(gray.len() * 4);
    for &g in gray {
        rgba.push(g);
        rgba.push(g);
        rgba.push(g);
        rgba.push(255);
    }
    rgba
}

/// Generate a synthetic RGBA image with a bright disk.
fn synthetic_disk_rgba(size: usize, cx: f32, cy: f32, radius: f32) -> Vec<u8> {
    let mut pixels = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let v = if (dx * dx + dy * dy).sqrt() <= radius {
                255u8
            } else {
                0u8
            };
            let i = (y * size + x) * 4;
            pixels[i] = v;
            pixels[i + 1] = v;
            pixels[i + 2] = v;
            pixels[i + 3] = 255;
        }
    }
    pixels
}

#[wasm_bindgen_test]
fn test_new() {
    let _processor = RadSymProcessor::new();
}

#[wasm_bindgen_test]
fn test_detect_circles_synthetic() {
    let size = 128;
    let pixels = synthetic_disk_rgba(size, 64.0, 64.0, 18.0);

    let mut processor = RadSymProcessor::new();
    processor.set_radii(&[17, 18, 19]);
    processor.set_polarity("bright").unwrap();
    processor.set_radius_hint(18.0);

    let result = processor
        .detect_circles(&pixels, size, size)
        .expect("detect_circles should succeed");
    assert!(result.length() > 0, "should detect at least one circle");
    assert_eq!(
        result.length() % 4,
        0,
        "result length must be multiple of 4"
    );
}

#[wasm_bindgen_test]
fn test_frst_response_dimensions() {
    let size = 64;
    let pixels = synthetic_disk_rgba(size, 32.0, 32.0, 10.0);

    let mut processor = RadSymProcessor::new();
    let response = processor
        .frst_response(&pixels, size, size)
        .expect("frst_response should succeed");
    assert_eq!(
        response.length() as usize,
        size * size,
        "response map must have width*height elements"
    );
}

#[wasm_bindgen_test]
fn test_gradient_field_dimensions() {
    let size = 64;
    let pixels = synthetic_disk_rgba(size, 32.0, 32.0, 10.0);

    let mut processor = RadSymProcessor::new();
    let field = processor
        .gradient_field(&pixels, size, size)
        .expect("gradient_field should succeed");
    assert_eq!(
        field.length() as usize,
        size * size * 2,
        "gradient field must have width*height*2 elements"
    );
}

#[wasm_bindgen_test]
fn test_response_heatmap_dimensions() {
    let size = 64;
    let pixels = synthetic_disk_rgba(size, 32.0, 32.0, 10.0);

    let mut processor = RadSymProcessor::new();
    let heatmap = processor
        .response_heatmap(&pixels, size, size, "hot")
        .expect("response_heatmap should succeed");
    assert_eq!(
        heatmap.len(),
        size * size * 4,
        "heatmap must have width*height*4 bytes (RGBA)"
    );
}

#[wasm_bindgen_test]
fn test_invalid_zero_width() {
    let mut processor = RadSymProcessor::new();
    let result = processor.detect_circles(&[], 0, 10);
    assert!(result.is_err(), "zero width should error");
}

#[wasm_bindgen_test]
fn test_invalid_buffer_length() {
    let mut processor = RadSymProcessor::new();
    let result = processor.detect_circles(&[0u8; 10], 4, 4);
    assert!(result.is_err(), "wrong buffer length should error");
}

#[wasm_bindgen_test]
fn test_invalid_polarity() {
    let mut processor = RadSymProcessor::new();
    assert!(processor.set_polarity("invalid").is_err());
}

#[wasm_bindgen_test]
fn test_invalid_colormap() {
    let size = 16;
    let pixels = vec![128u8; size * size * 4];
    let mut processor = RadSymProcessor::new();
    assert!(
        processor
            .response_heatmap(&pixels, size, size, "invalid")
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Ringgrid WASM-vs-native comparison tests
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn test_gradient_field_matches_native_ringgrid() {
    let (w, h, gray) = synthetic_ringgrid();
    let rgba = gray_to_rgba(&gray);

    // WASM path
    let mut processor = RadSymProcessor::new();
    let wasm_field = processor.gradient_field(&rgba, w, h).unwrap();
    let wasm_data: Vec<f32> = wasm_field.to_vec();

    // Native path
    let view = radsym::ImageView::from_slice(&gray, w, h).unwrap();
    let gradient = radsym::compute_gradient(&view, radsym::GradientOperator::Sobel).unwrap();
    let gx = gradient.gx().as_slice();
    let gy = gradient.gy().as_slice();
    let mut native_data = Vec::with_capacity(w * h * 2);
    for (&gx_val, &gy_val) in gx.iter().zip(gy.iter()) {
        native_data.push(gx_val);
        native_data.push(gy_val);
    }

    assert_eq!(wasm_data.len(), native_data.len());
    assert_eq!(
        wasm_data, native_data,
        "gradient field must be bitwise identical"
    );
}

#[wasm_bindgen_test]
fn test_frst_response_matches_native_ringgrid() {
    let (w, h, gray) = synthetic_ringgrid();
    let rgba = gray_to_rgba(&gray);

    // WASM path (uses default config: radii [3,5,7,9,11], polarity Both)
    let mut processor = RadSymProcessor::new();
    let wasm_response = processor.frst_response(&rgba, w, h).unwrap();
    let wasm_data: Vec<f32> = wasm_response.to_vec();

    // Native path — replicate what RadSymProcessor::frst_response does internally
    let view = radsym::ImageView::from_slice(&gray, w, h).unwrap();
    let gradient = radsym::compute_gradient(&view, radsym::GradientOperator::Sobel).unwrap();
    let frst_cfg = radsym::FrstConfig {
        polarity: radsym::Polarity::Both,
        ..radsym::FrstConfig::default()
    };
    let response = radsym::frst_response(&gradient, &frst_cfg).unwrap();
    let native_data: &[f32] = response.response().data();

    assert_eq!(wasm_data.len(), native_data.len());
    assert_eq!(
        wasm_data, native_data,
        "FRST response must be bitwise identical"
    );
}

#[wasm_bindgen_test]
fn test_response_heatmap_matches_native_ringgrid() {
    let (w, h, gray) = synthetic_ringgrid();
    let rgba = gray_to_rgba(&gray);

    // WASM path
    let mut processor = RadSymProcessor::new();
    let wasm_heatmap = processor.response_heatmap(&rgba, w, h, "hot").unwrap();

    // Native path
    let view = radsym::ImageView::from_slice(&gray, w, h).unwrap();
    let gradient = radsym::compute_gradient(&view, radsym::GradientOperator::Sobel).unwrap();
    let frst_cfg = radsym::FrstConfig {
        polarity: radsym::Polarity::Both,
        ..radsym::FrstConfig::default()
    };
    let response = radsym::frst_response(&gradient, &frst_cfg).unwrap();
    let native_heatmap = radsym::response_heatmap(response.response(), radsym::Colormap::Hot);
    let native_data = native_heatmap.into_data();

    assert_eq!(wasm_heatmap.len(), native_data.len());
    assert_eq!(wasm_heatmap, native_data, "heatmap must be byte-identical");
}

#[wasm_bindgen_test]
fn test_detect_circles_matches_native_ringgrid() {
    let (w, h, gray) = synthetic_ringgrid();
    let rgba = gray_to_rgba(&gray);

    let radii = vec![9u32, 11, 13, 15, 17];

    // WASM path
    let mut processor = RadSymProcessor::new();
    processor.set_radii(&radii);
    processor.set_polarity("dark").unwrap();
    processor.set_radius_hint(13.0);
    let wasm_result = processor.detect_circles(&rgba, w, h).unwrap();
    let wasm_data: Vec<f32> = wasm_result.to_vec();

    // Native path
    let view = radsym::ImageView::from_slice(&gray, w, h).unwrap();
    let config = radsym::DetectCirclesConfig {
        frst: radsym::FrstConfig {
            radii: radii.clone(),
            ..radsym::FrstConfig::default()
        },
        polarity: radsym::Polarity::Dark,
        radius_hint: 13.0,
        ..radsym::DetectCirclesConfig::default()
    };
    let detections = radsym::detect_circles(&view, &config).unwrap();

    assert_eq!(
        wasm_data.len(),
        detections.len() * 4,
        "detection count mismatch: WASM={} native={}",
        wasm_data.len() / 4,
        detections.len()
    );
    for (i, det) in detections.iter().enumerate() {
        let base = i * 4;
        assert_eq!(
            wasm_data[base], det.hypothesis.center.x,
            "detection {i} x mismatch"
        );
        assert_eq!(
            wasm_data[base + 1],
            det.hypothesis.center.y,
            "detection {i} y mismatch"
        );
        assert_eq!(
            wasm_data[base + 2],
            det.hypothesis.radius,
            "detection {i} radius mismatch"
        );
        assert_eq!(
            wasm_data[base + 3],
            det.score.total,
            "detection {i} score mismatch"
        );
    }
}

#[wasm_bindgen_test]
fn test_polarity_config_affects_frst_output() {
    let (w, h, gray) = synthetic_ringgrid();
    let rgba = gray_to_rgba(&gray);

    let mut processor = RadSymProcessor::new();

    processor.set_polarity("bright").unwrap();
    let bright: Vec<f32> = processor.frst_response(&rgba, w, h).unwrap().to_vec();

    processor.set_polarity("dark").unwrap();
    let dark: Vec<f32> = processor.frst_response(&rgba, w, h).unwrap().to_vec();

    assert_eq!(bright.len(), dark.len());
    assert_ne!(bright, dark, "bright and dark FRST responses must differ");
}

#[wasm_bindgen_test]
fn test_gradient_operator_config_affects_output() {
    let (w, h, gray) = synthetic_ringgrid();
    let rgba = gray_to_rgba(&gray);

    let mut processor = RadSymProcessor::new();

    processor.set_gradient_operator("sobel").unwrap();
    let sobel: Vec<f32> = processor.gradient_field(&rgba, w, h).unwrap().to_vec();

    processor.set_gradient_operator("scharr").unwrap();
    let scharr: Vec<f32> = processor.gradient_field(&rgba, w, h).unwrap().to_vec();

    assert_eq!(sobel.len(), scharr.len());
    assert_ne!(
        sobel, scharr,
        "Sobel and Scharr gradient fields must differ"
    );
}
