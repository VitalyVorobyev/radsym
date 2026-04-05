//! WebAssembly bindings for the radsym radial symmetry detection library.
//!
//! Exposes a stateful [`RadSymProcessor`] class to JavaScript with flat
//! typed-array inputs/outputs. All methods accept RGBA pixel data from
//! `canvas.getImageData()` and return `Float32Array` or `Uint8Array`.

use wasm_bindgen::prelude::*;

use radsym::{
    Colormap, DetectCirclesConfig, GradientOperator, ImageView, Polarity, compute_gradient,
    detect_circles, frst_response, response_heatmap,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `RadSymError` into a JS-friendly error string.
fn to_js_err(e: radsym::RadSymError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Convert RGBA pixels to grayscale using BT.601 luma weights.
///
/// Reuses `buf` to avoid per-call allocation.
fn rgba_to_gray(rgba: &[u8], w: usize, h: usize, buf: &mut Vec<u8>) -> Result<(), JsValue> {
    if w == 0 || h == 0 {
        return Err(JsValue::from_str("width and height must be > 0"));
    }
    let expected = w * h * 4;
    if rgba.len() != expected {
        return Err(JsValue::from_str(&format!(
            "expected {expected} bytes for {w}x{h} RGBA, got {}",
            rgba.len()
        )));
    }
    buf.resize(w * h, 0);
    for (i, chunk) in rgba.chunks_exact(4).enumerate() {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        buf[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
    }
    Ok(())
}

/// Parse a colormap name string.
fn parse_colormap(name: &str) -> Result<Colormap, JsValue> {
    match name {
        "jet" => Ok(Colormap::Jet),
        "hot" => Ok(Colormap::Hot),
        "magma" => Ok(Colormap::Magma),
        _ => Err(JsValue::from_str(&format!(
            "unknown colormap \"{name}\": expected \"jet\", \"hot\", or \"magma\""
        ))),
    }
}

// ---------------------------------------------------------------------------
// RadSymProcessor
// ---------------------------------------------------------------------------

/// Stateful radial symmetry processor.
///
/// Holds configuration and a reusable grayscale buffer. Create one instance,
/// configure it with `set_*` methods, then call processing methods repeatedly.
///
/// ## Output formats
///
/// | Method | Type | Stride | Fields |
/// |--------|------|--------|--------|
/// | `detect_circles` | `Float32Array` | 4 | `[x, y, radius, score, ...]` |
/// | `frst_response` | `Float32Array` | 1 | row-major response values |
/// | `response_heatmap` | `Uint8Array` | 4 | RGBA pixels, row-major |
/// | `gradient_field` | `Float32Array` | 2 | `[gx, gy, ...]` per pixel |
#[wasm_bindgen]
pub struct RadSymProcessor {
    config: DetectCirclesConfig,
    gray_buf: Vec<u8>,
}

impl Default for RadSymProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl RadSymProcessor {
    // -- Constructor --------------------------------------------------------

    /// Create a new processor with default configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            config: DetectCirclesConfig::default(),
            gray_buf: Vec::new(),
        }
    }

    // -- Processing methods -------------------------------------------------

    /// Run the full detection pipeline on RGBA pixel data.
    ///
    /// Returns a `Float32Array` with stride 4: `[x, y, radius, score, ...]`.
    /// Returns an empty array if no circles are detected.
    pub fn detect_circles(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let detections = detect_circles(&view, &self.config).map_err(to_js_err)?;

        let mut flat = Vec::with_capacity(detections.len() * 4);
        for d in &detections {
            flat.push(d.hypothesis.center.x);
            flat.push(d.hypothesis.center.y);
            flat.push(d.hypothesis.radius);
            flat.push(d.score.total);
        }
        Ok(js_sys::Float32Array::from(&flat[..]))
    }

    /// Compute the FRST response map from RGBA pixel data.
    ///
    /// Returns a `Float32Array` of length `width * height` (row-major).
    pub fn frst_response(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let mut frst_cfg = self.config.frst.clone();
        frst_cfg.polarity = self.config.polarity;
        let response = frst_response(&gradient, &frst_cfg).map_err(to_js_err)?;

        let data = response.response().data();
        Ok(js_sys::Float32Array::from(data))
    }

    /// Compute a colorized FRST response heatmap from RGBA pixel data.
    ///
    /// `colormap` must be one of `"jet"`, `"hot"`, or `"magma"`.
    /// Returns a `Uint8Array` of RGBA pixels (length `width * height * 4`).
    pub fn response_heatmap(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        colormap: &str,
    ) -> Result<Vec<u8>, JsValue> {
        let cmap = parse_colormap(colormap)?;

        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let mut frst_cfg = self.config.frst.clone();
        frst_cfg.polarity = self.config.polarity;
        let response = frst_response(&gradient, &frst_cfg).map_err(to_js_err)?;

        let heatmap = response_heatmap(response.response(), cmap);
        Ok(heatmap.into_data())
    }

    /// Compute the gradient field from RGBA pixel data.
    ///
    /// Returns a `Float32Array` with stride 2: `[gx, gy, ...]` per pixel,
    /// length `width * height * 2`, row-major order.
    pub fn gradient_field(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let gx = gradient.gx();
        let gy = gradient.gy();
        let gx_data = gx.as_slice();
        let gy_data = gy.as_slice();

        let mut flat = Vec::with_capacity(width * height * 2);
        for (gx_val, gy_val) in gx_data.iter().zip(gy_data.iter()) {
            flat.push(*gx_val);
            flat.push(*gy_val);
        }
        Ok(js_sys::Float32Array::from(&flat[..]))
    }

    // -- FrstConfig setters -------------------------------------------------

    /// Set the radii to test (in pixels).
    pub fn set_radii(&mut self, radii: &[u32]) {
        self.config.frst.radii = radii.to_vec();
    }

    /// Set the radial strictness exponent (alpha). Default: 2.0.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.config.frst.alpha = alpha;
    }

    /// Set the minimum gradient magnitude for voting. Default: 0.0.
    pub fn set_gradient_threshold(&mut self, threshold: f32) {
        self.config.frst.gradient_threshold = threshold;
    }

    /// Set the Gaussian smoothing factor (kn). Default: 0.5.
    pub fn set_smoothing_factor(&mut self, factor: f32) {
        self.config.frst.smoothing_factor = factor;
    }

    // -- NmsConfig setters --------------------------------------------------

    /// Set the NMS suppression radius in pixels. Default: 5.
    pub fn set_nms_radius(&mut self, radius: usize) {
        self.config.nms.radius = radius;
    }

    /// Set the NMS minimum response threshold. Default: 0.0.
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.config.nms.threshold = threshold;
    }

    /// Set the maximum number of detections. Default: 50.
    pub fn set_max_detections(&mut self, max: usize) {
        self.config.nms.max_detections = max;
    }

    // -- ScoringConfig setters ----------------------------------------------

    /// Set the number of angular samples around the annulus.
    pub fn set_num_angular_samples(&mut self, n: usize) {
        self.config.scoring.sampling.num_angular_samples = n;
    }

    /// Set the number of radial samples across the annulus width.
    pub fn set_num_radial_samples(&mut self, n: usize) {
        self.config.scoring.sampling.num_radial_samples = n;
    }

    /// Set the annulus margin as a fraction of radius. Default: 0.3.
    pub fn set_annulus_margin(&mut self, margin: f32) {
        self.config.scoring.annulus_margin = margin;
    }

    /// Set the minimum number of gradient samples. Default: 8.
    pub fn set_min_samples(&mut self, n: usize) {
        self.config.scoring.min_samples = n;
    }

    /// Set the weight of ringness in total score. Default: 0.7.
    pub fn set_weight_ringness(&mut self, w: f32) {
        self.config.scoring.weight_ringness = w;
    }

    /// Set the weight of angular coverage in total score. Default: 0.3.
    pub fn set_weight_coverage(&mut self, w: f32) {
        self.config.scoring.weight_coverage = w;
    }

    // -- CircleRefineConfig setters -----------------------------------------

    /// Set the maximum refinement iterations. Default: 10.
    pub fn set_max_iterations(&mut self, n: usize) {
        self.config.refinement.max_iterations = n;
    }

    /// Set the convergence tolerance in pixels. Default: 0.1.
    pub fn set_convergence_tol(&mut self, tol: f32) {
        self.config.refinement.convergence_tol = tol;
    }

    // -- Top-level config setters -------------------------------------------

    /// Set polarity: `"bright"`, `"dark"`, or `"both"`. Default: `"both"`.
    pub fn set_polarity(&mut self, polarity: &str) -> Result<(), JsValue> {
        self.config.polarity = match polarity {
            "bright" => Polarity::Bright,
            "dark" => Polarity::Dark,
            "both" => Polarity::Both,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown polarity \"{polarity}\": expected \"bright\", \"dark\", or \"both\""
                )));
            }
        };
        Ok(())
    }

    /// Set the approximate expected radius. Default: 10.0.
    pub fn set_radius_hint(&mut self, radius: f32) {
        self.config.radius_hint = radius;
    }

    /// Set the minimum support score threshold. Default: 0.0.
    pub fn set_min_score(&mut self, score: f32) {
        self.config.min_score = score;
    }

    /// Set gradient operator: `"sobel"` or `"scharr"`. Default: `"sobel"`.
    pub fn set_gradient_operator(&mut self, op: &str) -> Result<(), JsValue> {
        self.config.gradient_operator = match op {
            "sobel" => GradientOperator::Sobel,
            "scharr" => GradientOperator::Scharr,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown gradient operator \"{op}\": expected \"sobel\" or \"scharr\""
                )));
            }
        };
        Ok(())
    }
}
