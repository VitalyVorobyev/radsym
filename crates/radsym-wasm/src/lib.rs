//! WebAssembly bindings for the radsym radial symmetry detection library.
//!
//! Exposes a stateful [`RadSymProcessor`] class to JavaScript with flat
//! typed-array inputs/outputs. All methods accept RGBA pixel data from
//! `canvas.getImageData()` and return `Float32Array` or `Uint8Array`.

use wasm_bindgen::prelude::*;

use radsym::diagnostics::{Colormap, response_heatmap};
use radsym::{
    Circle, DetectCirclesConfig, FrstConfig, GradientField, GradientOperator, ImageView, Polarity,
    RefinementStatus, ResponseMap, RsdConfig, compute_gradient, detect_circles,
    detect_circles_with_diagnostics, extract_proposals, frst_response, frst_response_fused,
    refine_circle, rsd_response, rsd_response_fused, score_circle_support_detailed,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `RadSymError` into a JS-friendly error string.
fn to_js_err(e: radsym::RadSymError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Build a real JS `Error` (with `message` and a stack) from a message.
///
/// Used by the JSON config API so callers can `catch (e) { e.message }`.
fn js_error(message: &str) -> JsValue {
    js_sys::Error::new(message).into()
}

/// Parse a JSON document into a [`DetectCirclesConfig`].
///
/// Missing fields fall back to their defaults (the config is deserialized with
/// `serde(default)`), so a partial document such as `{"radii": [8, 10]}` is
/// valid. Ill-typed fields are errors; unknown fields are ignored.
fn parse_config_json(json: &str) -> Result<DetectCirclesConfig, String> {
    serde_json::from_str(json).map_err(|e| format!("invalid config JSON: {e}"))
}

/// Serialize a [`DetectCirclesConfig`] to compact JSON.
fn config_to_json(config: &DetectCirclesConfig) -> Result<String, String> {
    serde_json::to_string(config).map_err(|e| format!("cannot serialize config: {e}"))
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
// JSON config
// ---------------------------------------------------------------------------

/// The default detection configuration as a JSON string.
///
/// The JSON shape is described by `schemas/detect_circles_config.json`, which
/// ships inside the npm package. Enum values keep their Rust names
/// (`"Bright"`, `"Dark"`, `"Both"`; `"Sobel"`, `"Scharr"`), unlike the
/// lowercase strings accepted by `set_polarity` / `set_gradient_operator`.
///
/// The proposal algorithm (`"frst"`, `"rsd"`, ...) is not part of this config;
/// it is an argument of `detect_circles_detailed_with`, `response_heatmap` and
/// `extract_proposals`.
#[wasm_bindgen]
pub fn default_config_json() -> Result<String, JsValue> {
    config_to_json(&DetectCirclesConfig::default()).map_err(|e| js_error(&e))
}

// ---------------------------------------------------------------------------
// RadSymProcessor
// ---------------------------------------------------------------------------

/// Stateful radial symmetry processor.
///
/// Holds configuration and a reusable grayscale buffer. Create one instance,
/// configure it with `set_*` methods, then call processing methods repeatedly.
///
/// ## Proposal algorithms
///
/// | Method | Algorithm | Description |
/// |--------|-----------|-------------|
/// | `frst_response` | FRST | Multi-radius with orientation accumulator |
/// | `frst_response_fused` | FRST (fused) | Single-pass fused FRST (~faster) |
/// | `rsd_response` | RSD | Magnitude-only voting (~2× faster than FRST) |
/// | `rsd_response_fused` | RSD (fused) | Single-pass fused RSD |
///
/// ## Output formats
///
/// | Method | Type | Stride | Fields |
/// |--------|------|--------|--------|
/// | `detect_circles` | `Float32Array` | 4 | `[x, y, radius, score, ...]` |
/// | `detect_circles_detailed` | `Float32Array` | 8 | `[x, y, r, score, ringness, coverage, degen, status, ...]` |
/// | `detect_circles_detailed_with` | `Float32Array` | 8 | same layout, for a chosen proposer |
/// | `frst_response` | `Float32Array` | 1 | row-major response values |
/// | `rsd_response` | `Float32Array` | 1 | row-major response values |
/// | `response_heatmap` | `Uint8Array` | 4 | RGBA pixels, row-major |
/// | `gradient_field` | `Float32Array` | 2 | `[gx, gy, ...]` per pixel |
/// | `extract_proposals` | `Float32Array` | 3 | `[x, y, score, ...]` per proposal |
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

    /// Create a processor from a JSON config.
    ///
    /// Fields missing from the JSON take their default values, so a partial
    /// document such as `{"radii": [8, 10], "polarity": "Bright"}` is valid.
    /// Throws on malformed JSON or ill-typed fields (unknown fields are ignored). See
    /// [`default_config_json`] for the shape.
    pub fn with_config_json(json: &str) -> Result<RadSymProcessor, JsValue> {
        let config = parse_config_json(json).map_err(|e| js_error(&e))?;
        Ok(Self {
            config,
            gray_buf: Vec::new(),
        })
    }

    /// Replace the whole configuration from a JSON document.
    ///
    /// Same semantics as [`with_config_json`](Self::with_config_json): missing
    /// fields take their defaults (they do *not* keep the previous values).
    /// On error the current configuration is left unchanged.
    pub fn set_config_json(&mut self, json: &str) -> Result<(), JsValue> {
        self.config = parse_config_json(json).map_err(|e| js_error(&e))?;
        Ok(())
    }

    /// The current configuration as a JSON string, including any changes made
    /// through the `set_*` methods.
    pub fn config_json(&self) -> Result<String, JsValue> {
        config_to_json(&self.config).map_err(|e| js_error(&e))
    }

    // -- Full pipeline methods ----------------------------------------------

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

    /// Run the full detection pipeline, returning detailed per-detection info.
    ///
    /// Returns a `Float32Array` with stride 8:
    /// `[x, y, radius, total_score, ringness, angular_coverage, is_degenerate, status, ...]`
    ///
    /// - `is_degenerate`: `0.0` = false, `1.0` = true.
    /// - `status`: `0.0` = Converged, `1.0` = MaxIterations, `2.0` = Degenerate,
    ///   `3.0` = OutOfBounds.
    ///
    /// The per-detection `ringness`, `angular_coverage`, and `is_degenerate`
    /// fields are sourced from the diagnostics channel
    /// (`detect_circles_with_diagnostics`), whose `score_breakdowns` vec is
    /// index-aligned with the detections.
    pub fn detect_circles_detailed(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let (detections, diagnostics) =
            detect_circles_with_diagnostics(&view, &self.config).map_err(to_js_err)?;

        let mut flat = Vec::with_capacity(detections.len() * 8);
        for (d, breakdown) in detections.iter().zip(&diagnostics.score_breakdowns) {
            flat.push(d.hypothesis.center.x);
            flat.push(d.hypothesis.center.y);
            flat.push(d.hypothesis.radius);
            flat.push(d.score.total);
            flat.push(breakdown.ringness);
            flat.push(breakdown.angular_coverage);
            flat.push(if breakdown.is_degenerate { 1.0 } else { 0.0 });
            flat.push(match d.status {
                RefinementStatus::Converged => 0.0,
                RefinementStatus::MaxIterations => 1.0,
                RefinementStatus::Degenerate => 2.0,
                RefinementStatus::OutOfBounds => 3.0,
                _ => -1.0,
            });
        }
        Ok(js_sys::Float32Array::from(&flat[..]))
    }

    /// Run detection using a specific proposal algorithm, returning detailed
    /// per-detection info (stride 8, identical layout to
    /// [`detect_circles_detailed`](Self::detect_circles_detailed)).
    ///
    /// `algorithm` is one of `"frst"`, `"frst_fused"`, `"rsd"`, or `"rsd_fused"`.
    ///
    /// - `"frst"` runs the canonical one-call pipeline
    ///   (`detect_circles_with_diagnostics`), which selects a per-proposal
    ///   radius from the multi-radius FRST scale map.
    /// - the other proposers compute their response, extract proposals via NMS,
    ///   then score and refine each proposal at the configured `radius_hint`
    ///   using the same scoring and refinement stages as the pipeline.
    ///
    /// This lets an algorithm selector drive the final detection, not just the
    /// response/proposal previews.
    pub fn detect_circles_detailed_with(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        algorithm: &str,
    ) -> Result<js_sys::Float32Array, JsValue> {
        if algorithm == "frst" {
            return self.detect_circles_detailed(pixels, width, height);
        }

        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let response = match algorithm {
            "frst_fused" => {
                frst_response_fused(&gradient, &self.frst_config()).map_err(to_js_err)?
            }
            "rsd" => rsd_response(&gradient, &self.rsd_config()).map_err(to_js_err)?,
            "rsd_fused" => rsd_response_fused(&gradient, &self.rsd_config()).map_err(to_js_err)?,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown algorithm \"{algorithm}\": expected \"frst\", \"frst_fused\", \"rsd\", or \"rsd_fused\""
                )));
            }
        };

        let flat = self.detect_from_response(&gradient, &response);
        Ok(js_sys::Float32Array::from(&flat[..]))
    }

    // -- Proposal algorithm methods -----------------------------------------

    /// Compute the FRST response map (multi-radius, separate accumulators).
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

        let response = frst_response(&gradient, &self.frst_config()).map_err(to_js_err)?;

        let data = response.response().data();
        Ok(js_sys::Float32Array::from(data))
    }

    /// Compute the fused FRST response map (single-pass multi-radius voting).
    ///
    /// Faster than `frst_response` for large radius sets. Returns a
    /// `Float32Array` of length `width * height` (row-major).
    pub fn frst_response_fused(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let response = frst_response_fused(&gradient, &self.frst_config()).map_err(to_js_err)?;

        let data = response.response().data();
        Ok(js_sys::Float32Array::from(data))
    }

    /// Compute the RSD response map (multi-radius, magnitude-only voting).
    ///
    /// ~2× faster than FRST but with lower discrimination. Returns a
    /// `Float32Array` of length `width * height` (row-major).
    pub fn rsd_response(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let rsd_cfg = self.rsd_config();
        let response = rsd_response(&gradient, &rsd_cfg).map_err(to_js_err)?;

        let data = response.response().data();
        Ok(js_sys::Float32Array::from(data))
    }

    /// Compute the fused RSD response map (single-pass multi-radius voting).
    ///
    /// Returns a `Float32Array` of length `width * height` (row-major).
    pub fn rsd_response_fused(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let rsd_cfg = self.rsd_config();
        let response = rsd_response_fused(&gradient, &rsd_cfg).map_err(to_js_err)?;

        let data = response.response().data();
        Ok(js_sys::Float32Array::from(data))
    }

    // -- Heatmap methods ----------------------------------------------------

    /// Compute a colorized response heatmap.
    ///
    /// `algorithm` must be one of `"frst"`, `"frst_fused"`, `"rsd"`, or
    /// `"rsd_fused"`.
    /// `colormap` must be one of `"jet"`, `"hot"`, or `"magma"`.
    ///
    /// Returns a `Uint8Array` of RGBA pixels (length `width * height * 4`).
    pub fn response_heatmap(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        algorithm: &str,
        colormap: &str,
    ) -> Result<Vec<u8>, JsValue> {
        let cmap = parse_colormap(colormap)?;

        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let response = match algorithm {
            "frst" => frst_response(&gradient, &self.frst_config()).map_err(to_js_err)?,
            "frst_fused" => {
                frst_response_fused(&gradient, &self.frst_config()).map_err(to_js_err)?
            }
            "rsd" => rsd_response(&gradient, &self.rsd_config()).map_err(to_js_err)?,
            "rsd_fused" => rsd_response_fused(&gradient, &self.rsd_config()).map_err(to_js_err)?,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown algorithm \"{algorithm}\": expected \"frst\", \"frst_fused\", \"rsd\", or \"rsd_fused\""
                )));
            }
        };

        let heatmap = response_heatmap(response.response(), cmap);
        Ok(heatmap.into_data())
    }

    // -- Proposal extraction ------------------------------------------------

    /// Extract seed proposals via NMS from a response map.
    ///
    /// `algorithm` must be one of `"frst"`, `"frst_fused"`, `"rsd"`, or
    /// `"rsd_fused"`.
    ///
    /// Returns a `Float32Array` with stride 3: `[x, y, score, ...]` per
    /// proposal, sorted by descending score.
    pub fn extract_proposals(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
        algorithm: &str,
    ) -> Result<js_sys::Float32Array, JsValue> {
        rgba_to_gray(pixels, width, height, &mut self.gray_buf)?;
        let view = ImageView::from_slice(&self.gray_buf, width, height).map_err(to_js_err)?;
        let gradient = compute_gradient(&view, self.config.gradient_operator).map_err(to_js_err)?;

        let response = match algorithm {
            "frst" => frst_response(&gradient, &self.frst_config()).map_err(to_js_err)?,
            "frst_fused" => {
                frst_response_fused(&gradient, &self.frst_config()).map_err(to_js_err)?
            }
            "rsd" => rsd_response(&gradient, &self.rsd_config()).map_err(to_js_err)?,
            "rsd_fused" => rsd_response_fused(&gradient, &self.rsd_config()).map_err(to_js_err)?,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown algorithm \"{algorithm}\": expected \"frst\", \"frst_fused\", \"rsd\", or \"rsd_fused\""
                )));
            }
        };

        let proposals =
            extract_proposals(&response, &self.config.advanced.nms, self.config.polarity);

        let mut flat = Vec::with_capacity(proposals.len() * 3);
        for p in &proposals {
            flat.push(p.seed.position.x);
            flat.push(p.seed.position.y);
            flat.push(p.seed.score);
        }
        Ok(js_sys::Float32Array::from(&flat[..]))
    }

    // -- Gradient field ------------------------------------------------------

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
        // The top-level `radii` is the single source of truth: both
        // `detect_circles` and the stage helpers (`frst_config`, `rsd_config`)
        // read it.
        self.config.radii = radii.to_vec();
    }

    /// Set the radial strictness exponent (alpha). Default: 2.0.
    ///
    /// Only affects FRST; RSD does not use alpha.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.config.advanced.frst.alpha = alpha;
    }

    /// Set the minimum gradient magnitude for voting. Default: 0.0.
    pub fn set_gradient_threshold(&mut self, threshold: f32) {
        self.config.advanced.frst.gradient_threshold = threshold;
    }

    /// Set the Gaussian smoothing factor (kn). Default: 0.5.
    pub fn set_smoothing_factor(&mut self, factor: f32) {
        self.config.advanced.frst.smoothing_factor = factor;
    }

    // -- NmsConfig setters --------------------------------------------------

    /// Set the NMS suppression radius in pixels. Default: 5.
    pub fn set_nms_radius(&mut self, radius: usize) {
        self.config.advanced.nms.radius = radius;
    }

    /// Set the NMS minimum response threshold. Default: 0.0.
    pub fn set_nms_threshold(&mut self, threshold: f32) {
        self.config.advanced.nms.threshold = threshold;
    }

    /// Set the maximum number of detections. Default: 1000.
    pub fn set_max_detections(&mut self, max: usize) {
        self.config.advanced.nms.max_detections = max;
    }

    // -- ScoringConfig setters ----------------------------------------------

    /// Set the number of angular samples around the annulus.
    pub fn set_num_angular_samples(&mut self, n: usize) {
        self.config.advanced.scoring.sampling.num_angular_samples = n;
    }

    /// Set the number of radial samples across the annulus width.
    pub fn set_num_radial_samples(&mut self, n: usize) {
        self.config.advanced.scoring.sampling.num_radial_samples = n;
    }

    /// Set the annulus margin as a fraction of radius. Default: 0.3.
    pub fn set_annulus_margin(&mut self, margin: f32) {
        self.config.advanced.scoring.annulus_margin = margin;
    }

    /// Set the minimum number of gradient samples. Default: 8.
    pub fn set_min_samples(&mut self, n: usize) {
        self.config.advanced.scoring.min_samples = n;
    }

    /// Set the weight of ringness in total score. Default: 0.6.
    pub fn set_weight_ringness(&mut self, w: f32) {
        self.config.advanced.scoring.weight_ringness = w;
    }

    /// Set the weight of angular coverage in total score. Default: 0.4.
    pub fn set_weight_coverage(&mut self, w: f32) {
        self.config.advanced.scoring.weight_coverage = w;
    }

    // -- CircleRefineConfig setters -----------------------------------------

    /// Set the maximum refinement iterations. Default: 10.
    pub fn set_max_iterations(&mut self, n: usize) {
        self.config.advanced.refinement.max_iterations = n;
    }

    /// Set the convergence tolerance in pixels. Default: 0.1.
    pub fn set_convergence_tol(&mut self, tol: f32) {
        self.config.advanced.refinement.convergence_tol = tol;
    }

    /// Set the maximum center drift as a fraction of radius. Default: 0.5.
    ///
    /// Refinement stops if the center moves farther than
    /// `max_center_drift * radius` from the initial position.
    pub fn set_max_center_drift(&mut self, drift: f32) {
        self.config.advanced.refinement.max_center_drift = drift;
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

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl RadSymProcessor {
    /// Score and refine the proposals from a precomputed response map, packing
    /// accepted detections into the stride-8 `detect_circles_detailed` layout.
    ///
    /// Mirrors the pipeline's score → filter → refine loop, but at the fixed
    /// `radius_hint` (proposers other than unfused FRST expose no scale map).
    fn detect_from_response(&self, gradient: &GradientField, response: &ResponseMap) -> Vec<f32> {
        let proposals =
            extract_proposals(response, &self.config.advanced.nms, self.config.polarity);
        let mut rows: Vec<[f32; 8]> = Vec::new();
        for p in &proposals {
            let circle = Circle::new(p.seed.position, self.config.radius_hint);
            let breakdown =
                score_circle_support_detailed(gradient, &circle, &self.config.advanced.scoring);
            if breakdown.is_degenerate || breakdown.total < self.config.min_score {
                continue;
            }
            if let Ok(refined) = refine_circle(gradient, &circle, &self.config.advanced.refinement)
            {
                let status = match refined.status {
                    RefinementStatus::Converged => 0.0,
                    RefinementStatus::MaxIterations => 1.0,
                    RefinementStatus::Degenerate => 2.0,
                    RefinementStatus::OutOfBounds => 3.0,
                    _ => -1.0,
                };
                rows.push([
                    refined.hypothesis.center.x,
                    refined.hypothesis.center.y,
                    refined.hypothesis.radius,
                    breakdown.total,
                    breakdown.ringness,
                    breakdown.angular_coverage,
                    if breakdown.is_degenerate { 1.0 } else { 0.0 },
                    status,
                ]);
            }
        }
        rows.sort_by(|a, b| b[3].partial_cmp(&a[3]).unwrap_or(std::cmp::Ordering::Equal));
        rows.into_iter().flatten().collect()
    }

    /// Build a full [`FrstConfig`] from the single-source-of-truth top-level
    /// `radii` and `polarity` plus the shared voting tuning (`advanced.frst`).
    fn frst_config(&self) -> FrstConfig {
        self.config
            .advanced
            .frst
            .to_frst_config(self.config.radii.clone(), self.config.polarity)
    }

    /// Build an [`RsdConfig`] from the shared FRST config fields.
    ///
    /// RSD uses the same radii, gradient threshold, polarity, and smoothing
    /// factor as FRST; it simply omits the alpha exponent.
    fn rsd_config(&self) -> RsdConfig {
        let mut config = RsdConfig::default();
        config.radii = self.config.radii.clone();
        config.gradient_threshold = self.config.advanced.frst.gradient_threshold;
        config.polarity = self.config.polarity;
        config.smoothing_factor = self.config.advanced.frst.smoothing_factor;
        config
    }
}

#[cfg(test)]
mod tests {
    //! Native tests for the JSON config path. The error branches that build a
    //! `JsValue` are not exercised here (they only work on wasm32); the parsing
    //! helpers they wrap are.

    use super::*;

    fn as_value(config: &DetectCirclesConfig) -> serde_json::Value {
        serde_json::to_value(config).unwrap()
    }

    #[test]
    fn default_config_json_round_trips() {
        let json = config_to_json(&DetectCirclesConfig::default()).unwrap();
        let parsed = parse_config_json(&json).unwrap();
        assert_eq!(as_value(&parsed), as_value(&DetectCirclesConfig::default()));
    }

    #[test]
    fn partial_json_takes_defaults() {
        let parsed = parse_config_json(r#"{"radii": [4, 6], "polarity": "Bright"}"#).unwrap();
        assert_eq!(parsed.radii, vec![4, 6]);
        assert_eq!(parsed.polarity, Polarity::Bright);
        assert_eq!(
            parsed.radius_hint,
            DetectCirclesConfig::default().radius_hint
        );
    }

    #[test]
    fn invalid_json_reports_a_useful_message() {
        let err = parse_config_json("{not json").unwrap_err();
        assert!(err.starts_with("invalid config JSON:"), "{err}");
        let err = parse_config_json(r#"{"polarity": "Sideways"}"#).unwrap_err();
        assert!(err.contains("Sideways") || err.contains("variant"), "{err}");
        let err = parse_config_json(r#"{"radii": "ten"}"#).unwrap_err();
        assert!(err.contains("invalid type"), "{err}");
    }

    #[test]
    fn setters_are_reflected_in_config_json() {
        let mut p = RadSymProcessor::new();
        p.set_radii(&[7, 9]);
        p.set_alpha(3.0);
        p.set_nms_radius(8);
        p.set_polarity("dark")
            .unwrap_or_else(|_| panic!("set_polarity"));
        let value: serde_json::Value = serde_json::from_str(&p.config_json().unwrap()).unwrap();
        assert_eq!(value["radii"], serde_json::json!([7, 9]));
        assert_eq!(value["polarity"], "Dark");
        assert_eq!(value["advanced"]["frst"]["alpha"], 3.0);
        assert_eq!(value["advanced"]["nms"]["radius"], 8);
    }

    #[test]
    fn config_json_drives_the_processor() {
        let json = r#"{"radii": [5], "advanced": {"nms": {"radius": 9}}}"#;
        let p = RadSymProcessor::with_config_json(json).unwrap();
        assert_eq!(p.config.radii, vec![5]);
        assert_eq!(p.config.advanced.nms.radius, 9);
        // Untouched sibling keeps its default.
        assert_eq!(
            p.config.advanced.nms.max_detections,
            DetectCirclesConfig::default().advanced.nms.max_detections
        );
    }
}
