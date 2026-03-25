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

/// Configuration for FRST response computation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
            let dx = gx / mag;
            let dy = gy / mag;

            // Positive-affected pixel: p_+ = p + round(g/|g| * n)
            let px_pos = x as i32 + (dx * n as Scalar).round() as i32;
            let py_pos = y as i32 + (dy * n as Scalar).round() as i32;

            // Negative-affected pixel: p_- = p - round(g/|g| * n)
            let px_neg = x as i32 - (dx * n as Scalar).round() as i32;
            let py_neg = y as i32 - (dy * n as Scalar).round() as i32;

            let vote_pos = config.polarity.votes_positive();
            let vote_neg = config.polarity.votes_negative();

            if vote_pos
                && px_pos >= 0
                && (px_pos as usize) < w
                && py_pos >= 0
                && (py_pos as usize) < h
            {
                let pidx = py_pos as usize * w + px_pos as usize;
                o_data[pidx] += 1.0;
                m_data[pidx] += mag;
            }

            if vote_neg
                && px_neg >= 0
                && (px_neg as usize) < w
                && py_neg >= 0
                && (py_neg as usize) < h
            {
                let pidx = py_neg as usize * w + px_neg as usize;
                o_data[pidx] -= 1.0;
                m_data[pidx] += mag;
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

    for i in 0..w * h {
        let o_tilde = o_data[i] / o_max;
        let m_tilde = m_data[i] / m_max;
        f_data[i] = o_tilde.abs().powf(alpha) * m_tilde;
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
/// Returns the summed response across all configured radii: `S = sum(S_n)`.
pub fn frst_response(gradient: &GradientField, config: &FrstConfig) -> Result<OwnedImage<Scalar>> {
    let w = gradient.width();
    let h = gradient.height();
    let mut response = OwnedImage::<Scalar>::zeros(w, h)?;

    for &radius in &config.radii {
        let s_n = frst_response_single(gradient, radius, config)?;
        let resp_data = response.data_mut();
        let s_data = s_n.data();
        for i in 0..w * h {
            resp_data[i] += s_data[i];
        }
    }

    Ok(response)
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
        let resp_data = response.data();

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
        let resp_data = response.data();

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
        let resp_data = response.data();

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
        assert_eq!(response.width(), 40);
        assert_eq!(response.height(), 30);
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
            response.data().iter().all(|&v| v == 0.0),
            "high threshold on uniform image should produce zero response"
        );
    }
}
