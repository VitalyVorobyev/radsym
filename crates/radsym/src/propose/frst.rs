//! Fast Radial Symmetry Transform (FRST).
//!
//! Implements the gradient-offset center-voting algorithm from:
//!
//! - Loy, G. & Zelinsky, A. (2002). *A Fast Radial Symmetry Transform for
//!   Detecting Points of Interest.* ECCV 2002.
//! - Loy, G. & Zelinsky, A. (2003). *Fast Radial Symmetry for Detecting
//!   Points of Interest.* IEEE TPAMI 25(8).
//!
//! ## Algorithm summary
//!
//! For each tested radius `n`, every pixel with a gradient above a magnitude
//! threshold votes for two "affected" pixels along its gradient direction:
//!
//! - `p_+ = p + round(g/|g| * n)` (positive-affected)
//! - `p_- = p - round(g/|g| * n)` (negative-affected)
//!
//! Two accumulator images are maintained per radius:
//! - `O_n`: orientation projection (incremented by ±1)
//! - `M_n`: magnitude projection (incremented by ±|g|)
//!
//! These are combined as `F_n = |O_n_tilde|^alpha * M_n_tilde`, where `_tilde`
//! denotes clamping/normalization. The result is smoothed with a Gaussian and
//! summed across all radii.
//!
//! ## Fidelity
//!
//! This is a **reference + production** implementation. The core voting matches
//! the original paper; production additions include gradient magnitude
//! thresholding, polarity selection, and optional rayon parallelism.

use crate::core::error::Result;
use crate::core::gradient::GradientField;
use crate::core::image_view::OwnedImage;
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Configuration for FRST response computation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct FrstConfig {
    /// Set of discrete radii to test (in pixels).
    pub radii: Vec<u32>,
    /// Radial strictness exponent (alpha). Higher values require more
    /// consistent orientation evidence. Default: 2.0.
    pub alpha: Scalar,
    /// Minimum gradient magnitude to participate in voting. Pixels with
    /// `|g| < gradient_threshold` are skipped. Default: 0.0 (all pixels vote).
    pub gradient_threshold: Scalar,
    /// Which polarity to detect.
    pub polarity: Polarity,
    /// Standard deviation of the Gaussian smoothing kernel applied per-radius.
    /// Relative to the radius: `sigma = kn * n`. Default kn: 0.5.
    pub smoothing_factor: Scalar,
}

impl FrstConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> crate::core::error::Result<()> {
        use crate::core::error::RadSymError;
        if self.radii.is_empty() {
            return Err(RadSymError::InvalidConfig {
                reason: "radii must be non-empty",
            });
        }
        if self.radii.contains(&0) {
            return Err(RadSymError::InvalidConfig {
                reason: "all radii must be > 0",
            });
        }
        if self.alpha < 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "alpha must be >= 0",
            });
        }
        if self.smoothing_factor <= 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "smoothing_factor must be > 0",
            });
        }
        Ok(())
    }
}

impl Default for FrstConfig {
    fn default() -> Self {
        Self {
            radii: vec![3, 5, 7, 9, 11],
            alpha: 2.0,
            gradient_threshold: 0.0,
            polarity: Polarity::Both,
            smoothing_factor: 0.5,
        }
    }
}

/// Compute the FRST response map for a single radius.
///
/// Returns the smoothed per-radius contribution `S_n`.
pub fn frst_response_single(
    gradient: &GradientField,
    radius: u32,
    config: &FrstConfig,
) -> Result<OwnedImage<Scalar>> {
    config.validate()?;
    let w = gradient.width();
    let h = gradient.height();
    let n = radius as i32;

    // Accumulator images for this radius
    let mut o_n = OwnedImage::<Scalar>::zeros(w, h)?;
    let mut m_n = OwnedImage::<Scalar>::zeros(w, h)?;

    let o_data = o_n.data_mut();
    let m_data = m_n.data_mut();
    let gx_data = gradient.gx.data();
    let gy_data = gradient.gy.data();

    let thresh_sq = config.gradient_threshold * config.gradient_threshold;
    let vote_pos = config.polarity.votes_positive();
    let vote_neg = config.polarity.votes_negative();
    let n_f = n as Scalar;

    // Voting pass: each pixel casts votes to affected pixels
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let gx = gx_data[idx];
            let gy = gy_data[idx];
            let mag_sq = gx * gx + gy * gy;

            if mag_sq < thresh_sq {
                continue;
            }

            let mag = mag_sq.sqrt();
            let inv_mag = mag.recip();
            let dx = gx * inv_mag;
            let dy = gy * inv_mag;

            let offset_x = (dx * n_f).round() as i32;
            let offset_y = (dy * n_f).round() as i32;

            if vote_pos {
                let px = x as i32 + offset_x;
                let py = y as i32 + offset_y;
                if px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h {
                    let pidx = py as usize * w + px as usize;
                    o_data[pidx] += 1.0;
                    m_data[pidx] += mag;
                }
            }

            if vote_neg {
                let px = x as i32 - offset_x;
                let py = y as i32 - offset_y;
                if px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h {
                    let pidx = py as usize * w + px as usize;
                    o_data[pidx] -= 1.0;
                    m_data[pidx] += mag;
                }
            }
        }
    }

    // Combine: F_n = |O_n_tilde|^alpha * M_n_tilde
    // Clamp O_n to [-k_n, k_n] where k_n is a normalization constant.
    // Following the paper, k_n is typically the expected maximum vote count.
    // We use a simple normalization: divide by max(|O_n|) to get [-1, 1].
    let o_max = o_data
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, Scalar::max)
        .max(1.0); // avoid division by zero

    let m_max = m_data.iter().copied().fold(0.0f32, Scalar::max).max(1.0);

    let alpha = config.alpha;
    let mut f_n = OwnedImage::<Scalar>::zeros(w, h)?;
    let f_data = f_n.data_mut();

    // Special-case common alpha values to avoid expensive powf.
    // Alpha=2 is the paper's default; alpha=1 skips the exponent entirely.
    match alpha as u32 {
        1 if (alpha - 1.0).abs() < 1e-6 => {
            for i in 0..w * h {
                let o_abs = (o_data[i] / o_max).abs();
                f_data[i] = o_abs * (m_data[i] / m_max);
            }
        }
        2 if (alpha - 2.0).abs() < 1e-6 => {
            for i in 0..w * h {
                let o_abs = (o_data[i] / o_max).abs();
                f_data[i] = o_abs * o_abs * (m_data[i] / m_max);
            }
        }
        _ => {
            for i in 0..w * h {
                let o_abs = (o_data[i] / o_max).abs();
                f_data[i] = o_abs.powf(alpha) * (m_data[i] / m_max);
            }
        }
    }

    // Gaussian smoothing
    let sigma = config.smoothing_factor * radius as Scalar;
    if sigma > 0.5 {
        crate::core::blur::gaussian_blur_inplace(&mut f_n, sigma);
    }

    Ok(f_n)
}

/// Compute the full multi-radius FRST response map.
///
/// Returns the summed response across all configured radii: `S = sum(S_n)`,
/// wrapped in a [`ResponseMap`](super::extract::ResponseMap) tagged with
/// [`ProposalSource::Frst`](super::seed::ProposalSource::Frst).
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, FrstConfig, sobel_gradient, frst_response};
///
/// let size = 64usize;
/// let mut data = vec![0u8; size * size];
/// for y in 0..size {
///     for x in 0..size {
///         let dx = x as f32 - 32.0;
///         let dy = y as f32 - 32.0;
///         if (dx * dx + dy * dy).sqrt() <= 10.0 { data[y * size + x] = 255; }
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = sobel_gradient(&image).unwrap();
/// let config = FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() };
/// let response = frst_response(&grad, &config).unwrap();
/// assert_eq!(response.response().width(), size);
/// assert!(response.response().data().iter().any(|&v| v > 0.0));
/// ```
pub fn frst_response(
    gradient: &GradientField,
    config: &FrstConfig,
) -> Result<super::extract::ResponseMap> {
    config.validate()?;
    let w = gradient.width();
    let h = gradient.height();
    let mut response = OwnedImage::<Scalar>::zeros(w, h)?;

    #[cfg(feature = "rayon")]
    let per_radius = config
        .radii
        .par_iter()
        .map(|&radius| frst_response_single(gradient, radius, config))
        .collect::<Vec<_>>();

    #[cfg(not(feature = "rayon"))]
    let per_radius = config
        .radii
        .iter()
        .map(|&radius| frst_response_single(gradient, radius, config))
        .collect::<Vec<_>>();

    for s_n in per_radius {
        let s_n = s_n?;
        let resp_data = response.data_mut();
        let s_data = s_n.data();
        for i in 0..w * h {
            resp_data[i] += s_data[i];
        }
    }

    Ok(super::extract::ResponseMap::new(
        response,
        super::seed::ProposalSource::Frst,
    ))
}

/// Compute a fused multi-radius response map in a single pixel pass.
///
/// Unlike [`frst_response`], which processes each radius independently
/// (separate accumulator pair, separate Gaussian blur per radius), this
/// function fuses all radii into **one** pixel traversal with a single
/// shared accumulator and **one** final Gaussian blur.
///
/// The orientation accumulator and the [`FrstConfig::alpha`] exponent are
/// **ignored**; only gradient magnitude votes are accumulated. This yields
/// comparable peak locations to [`frst_response`] for typical circular
/// targets but runs significantly faster when many radii are tested:
///
/// - **One** image traversal (vs. N for standard FRST).
/// - **One** Gaussian blur with `sigma = smoothing_factor * median(radii)`
///   (vs. N per-radius blurs).
/// - **One** accumulator allocation (vs. 2N for FRST's O + M pairs).
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, FrstConfig, sobel_gradient};
/// use radsym::propose::frst::multiradius_response;
///
/// let size = 64usize;
/// let mut data = vec![0u8; size * size];
/// for y in 0..size {
///     for x in 0..size {
///         let dx = x as f32 - 32.0;
///         let dy = y as f32 - 32.0;
///         if (dx * dx + dy * dy).sqrt() <= 10.0 { data[y * size + x] = 255; }
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = sobel_gradient(&image).unwrap();
/// let config = FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() };
/// let response = multiradius_response(&grad, &config).unwrap();
/// assert_eq!(response.response().width(), size);
/// assert!(response.response().data().iter().any(|&v| v > 0.0));
/// ```
pub fn multiradius_response(
    gradient: &GradientField,
    config: &FrstConfig,
) -> Result<super::extract::ResponseMap> {
    config.validate()?;
    let accumulator = super::fused::fused_voting_pass(
        gradient,
        &config.radii,
        config.gradient_threshold,
        config.polarity,
        config.smoothing_factor,
    )?;
    Ok(super::extract::ResponseMap::new(
        accumulator,
        super::seed::ProposalSource::Frst,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;

    /// Generate a synthetic bright disk image (white circle on black background).
    fn make_bright_disk(size: usize, cx: usize, cy: usize, radius: f32) -> Vec<u8> {
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                if (dx * dx + dy * dy).sqrt() <= radius {
                    data[y * size + x] = 255;
                }
            }
        }
        data
    }

    /// Generate a synthetic ring image (bright annulus on black background).
    fn make_ring(
        size: usize,
        cx: usize,
        cy: usize,
        inner_radius: f32,
        outer_radius: f32,
    ) -> Vec<u8> {
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let r = (dx * dx + dy * dy).sqrt();
                if r >= inner_radius && r <= outer_radius {
                    data[y * size + x] = 255;
                }
            }
        }
        data
    }

    #[test]
    fn frst_detects_bright_disk_center() {
        let size = 64;
        let cx = 32;
        let cy = 32;
        let data = make_bright_disk(size, cx, cy, 10.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![9, 10, 11],
            alpha: 2.0,
            gradient_threshold: 1.0,
            polarity: Polarity::Bright,
            smoothing_factor: 0.5,
        };

        let response = frst_response(&grad, &config).unwrap();
        let resp_data = response.response().data();

        // Find the peak
        let (max_idx, &max_val) = resp_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_x = max_idx % size;
        let peak_y = max_idx / size;

        assert!(
            max_val > 0.0,
            "response should have a positive peak, got {max_val}"
        );
        assert!(
            (peak_x as f32 - cx as f32).abs() < 5.0,
            "peak x={peak_x} should be near center x={cx}"
        );
        assert!(
            (peak_y as f32 - cy as f32).abs() < 5.0,
            "peak y={peak_y} should be near center y={cy}"
        );
    }

    #[test]
    fn frst_detects_dark_disk() {
        let size = 64;
        let cx = 32;
        let cy = 32;
        // Dark disk: invert the bright disk
        let mut data = make_bright_disk(size, cx, cy, 10.0);
        for v in &mut data {
            *v = 255 - *v;
        }
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![9, 10, 11],
            polarity: Polarity::Dark,
            gradient_threshold: 1.0,
            ..FrstConfig::default()
        };

        let response = frst_response(&grad, &config).unwrap();
        let resp_data = response.response().data();

        let (max_idx, &max_val) = resp_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_x = max_idx % size;
        let peak_y = max_idx / size;

        assert!(max_val > 0.0);
        assert!((peak_x as f32 - cx as f32).abs() < 5.0);
        assert!((peak_y as f32 - cy as f32).abs() < 5.0);
    }

    #[test]
    fn frst_detects_ring_center() {
        let size = 80;
        let cx = 40;
        let cy = 40;
        let data = make_ring(size, cx, cy, 12.0, 16.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![11, 12, 13, 14, 15, 16, 17],
            alpha: 2.0,
            gradient_threshold: 1.0,
            polarity: Polarity::Both,
            smoothing_factor: 0.5,
        };

        let response = frst_response(&grad, &config).unwrap();
        let resp_data = response.response().data();

        let (max_idx, &max_val) = resp_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_x = max_idx % size;
        let peak_y = max_idx / size;

        assert!(max_val > 0.0, "ring should produce a positive response");
        assert!(
            (peak_x as f32 - cx as f32).abs() < 5.0,
            "peak x={peak_x} should be near ring center x={cx}"
        );
        assert!(
            (peak_y as f32 - cy as f32).abs() < 5.0,
            "peak y={peak_y} should be near ring center y={cy}"
        );
    }

    #[test]
    fn frst_multiple_targets() {
        let size = 128;
        let mut data = vec![0u8; size * size];

        // Draw two disks at different locations
        let targets = [(32, 32, 8.0), (90, 90, 10.0)];
        for &(cx, cy, r) in &targets {
            for y in 0..size {
                for x in 0..size {
                    let dx = x as f32 - cx as f32;
                    let dy = y as f32 - cy as f32;
                    if (dx * dx + dy * dy).sqrt() <= r {
                        data[y * size + x] = 255;
                    }
                }
            }
        }

        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![7, 8, 9, 10, 11],
            gradient_threshold: 1.0,
            polarity: Polarity::Bright,
            ..FrstConfig::default()
        };

        let response = frst_response(&grad, &config).unwrap();

        // Both target locations should have positive response
        for &(cx, cy, _) in &targets {
            let val = response.view().get(cx, cy).copied().unwrap_or(0.0);
            assert!(
                val > 0.0,
                "target at ({cx},{cy}) should have positive response, got {val}"
            );
        }
    }

    #[test]
    fn frst_response_dimensions_match_input() {
        let data = vec![128u8; 40 * 30];
        let image = ImageView::from_slice(&data, 40, 30).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        let config = FrstConfig::default();
        let response = frst_response(&grad, &config).unwrap();
        assert_eq!(response.response().width(), 40);
        assert_eq!(response.response().height(), 30);
    }

    #[test]
    fn default_config_passes_validation() {
        FrstConfig::default().validate().unwrap();
    }

    #[test]
    fn empty_radii_fails_validation() {
        let config = FrstConfig {
            radii: vec![],
            ..FrstConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn zero_radius_fails_validation() {
        let config = FrstConfig {
            radii: vec![5, 0, 3],
            ..FrstConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn negative_alpha_fails_validation() {
        let config = FrstConfig {
            alpha: -1.0,
            ..FrstConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn zero_smoothing_factor_fails_validation() {
        let config = FrstConfig {
            smoothing_factor: 0.0,
            ..FrstConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn frst_gradient_threshold_reduces_noise() {
        // Uniform image with very low gradients: high threshold should suppress all votes
        let data = vec![128u8; 32 * 32];
        let image = ImageView::from_slice(&data, 32, 32).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![5],
            gradient_threshold: 100.0, // very high
            ..FrstConfig::default()
        };

        let response = frst_response(&grad, &config).unwrap();
        assert!(
            response.response().data().iter().all(|&v| v == 0.0),
            "high threshold on uniform image should produce zero response"
        );
    }

    // --- multiradius_response tests ---

    #[test]
    fn multiradius_detects_bright_disk_center() {
        let size = 80;
        let data = make_bright_disk(size, 40, 40, 12.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![11, 12, 13],
            gradient_threshold: 1.0,
            polarity: Polarity::Bright,
            ..FrstConfig::default()
        };

        let response = multiradius_response(&grad, &config).unwrap();
        let resp_data = response.response().data();
        let (max_idx, _) = resp_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let peak_x = max_idx % size;
        let peak_y = max_idx / size;
        let dx = peak_x as f32 - 40.0;
        let dy = peak_y as f32 - 40.0;
        assert!(
            (dx * dx + dy * dy).sqrt() < 5.0,
            "peak at ({peak_x}, {peak_y}) too far from center (40, 40)"
        );
    }

    #[test]
    fn multiradius_detects_dark_disk() {
        let size = 80;
        let mut data = make_bright_disk(size, 40, 40, 12.0);
        for v in &mut data {
            *v = 255 - *v;
        }
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![11, 12, 13],
            polarity: Polarity::Dark,
            gradient_threshold: 1.0,
            ..FrstConfig::default()
        };

        let response = multiradius_response(&grad, &config).unwrap();
        let resp_data = response.response().data();
        let (max_idx, _) = resp_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let peak_x = max_idx % size;
        let peak_y = max_idx / size;
        let dx = peak_x as f32 - 40.0;
        let dy = peak_y as f32 - 40.0;
        assert!(
            (dx * dx + dy * dy).sqrt() < 5.0,
            "dark peak at ({peak_x}, {peak_y}) too far from center (40, 40)"
        );
    }

    #[test]
    fn multiradius_dimensions_match_input() {
        let data = vec![128u8; 40 * 30];
        let image = ImageView::from_slice(&data, 40, 30).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        let response = multiradius_response(&grad, &FrstConfig::default()).unwrap();
        assert_eq!(response.response().width(), 40);
        assert_eq!(response.response().height(), 30);
    }

    #[test]
    fn multiradius_gradient_threshold_suppresses_uniform() {
        let data = vec![128u8; 32 * 32];
        let image = ImageView::from_slice(&data, 32, 32).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![5],
            gradient_threshold: 100.0,
            ..FrstConfig::default()
        };

        let response = multiradius_response(&grad, &config).unwrap();
        assert!(
            response.response().data().iter().all(|&v| v == 0.0),
            "high threshold on uniform image should produce zero response"
        );
    }

    #[test]
    fn multiradius_matches_frst_peak_location() {
        let size = 100;
        let data = make_bright_disk(size, 50, 50, 16.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = FrstConfig {
            radii: vec![14, 15, 16, 17, 18],
            gradient_threshold: 1.0,
            polarity: Polarity::Bright,
            ..FrstConfig::default()
        };

        let frst = frst_response(&grad, &config).unwrap();
        let multi = multiradius_response(&grad, &config).unwrap();

        let find_peak = |data: &[f32], w: usize| -> (usize, usize) {
            let (idx, _) = data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (idx % w, idx / w)
        };

        let (fx, fy) = find_peak(frst.response().data(), size);
        let (mx, my) = find_peak(multi.response().data(), size);
        let dist = ((fx as f32 - mx as f32).powi(2) + (fy as f32 - my as f32).powi(2)).sqrt();
        assert!(
            dist < 3.0,
            "peak locations differ by {dist}px: frst=({fx},{fy}) multi=({mx},{my})"
        );
    }

    #[test]
    fn compute_median_radius_odd() {
        use crate::propose::fused::compute_median_radius;
        assert_eq!(compute_median_radius(&[3, 7, 5]), 5.0);
        assert_eq!(compute_median_radius(&[10]), 10.0);
        assert_eq!(compute_median_radius(&[1, 2, 3, 4, 5]), 3.0);
    }

    #[test]
    fn compute_median_radius_even() {
        use crate::propose::fused::compute_median_radius;
        assert_eq!(compute_median_radius(&[4, 8]), 6.0);
        assert_eq!(compute_median_radius(&[2, 4, 6, 8]), 5.0);
    }
}
