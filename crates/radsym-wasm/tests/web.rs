use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use radsym_wasm::RadSymProcessor;

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
