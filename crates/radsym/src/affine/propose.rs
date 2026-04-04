//! GFRS-style affine-aware proposal generation.
//!
//! Extends FRST voting by warping gradient offset directions through a set
//! of affine maps, enabling detection of elliptical symmetry under
//! perspective distortion.
//!
//! Literature: Ni, K., Singh, M., Bahlmann, C. (2012). *Fast Radial Symmetry
//! Detection Under Affine Transformations.* CVPR 2012.
//!
//! ## Algorithm summary
//!
//! For each affine map `A` in a sampled set:
//!   1. Transform the gradient direction `g/|g|` by `A`.
//!   2. Compute the warped offset and vote into an accumulator.
//!   3. Smooth and record the peak response.
//!
//! The map producing the strongest peak gives both the center and an
//! estimate of the affine distortion (hence the ellipse parameters).

use crate::core::error::Result;
use crate::core::gradient::GradientField;
use crate::core::image_view::OwnedImage;
use crate::core::scalar::Scalar;

use super::transform::AffineMap;

/// Configuration for affine-aware FRST response.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineFrstConfig {
    /// Radius for voting (single radius for simplicity).
    pub radius: u32,
    /// Minimum gradient magnitude to vote.
    pub gradient_threshold: Scalar,
    /// Gaussian smoothing factor (sigma = factor * radius).
    pub smoothing_factor: Scalar,
    /// Affine maps to sample.
    pub affine_maps: Vec<AffineMap>,
}

impl Default for AffineFrstConfig {
    fn default() -> Self {
        Self {
            radius: 8,
            gradient_threshold: 0.0,
            smoothing_factor: 0.5,
            affine_maps: super::transform::sample_affine_maps(6, 3),
        }
    }
}

/// Result of affine-aware voting for a single affine map.
#[derive(Debug, Clone)]
pub struct AffineResponse {
    /// The response map for this affine map.
    pub response: OwnedImage<Scalar>,
    /// The affine map that produced this response.
    pub affine_map: AffineMap,
    /// Peak response value in this map.
    pub peak_value: Scalar,
}

/// Compute the affine FRST response for a single affine map.
///
/// Warps gradient directions by the affine map before computing the
/// voting offset, then accumulates magnitude votes.
pub fn affine_frst_response_single(
    gradient: &GradientField,
    radius: u32,
    affine: &AffineMap,
    gradient_threshold: Scalar,
    smoothing_factor: Scalar,
) -> Result<OwnedImage<Scalar>> {
    let w = gradient.width();
    let h = gradient.height();
    let n = radius as Scalar;

    let mut acc = OwnedImage::<Scalar>::zeros(w, h)?;
    let acc_data = acc.data_mut();
    let gx_data = gradient.gx.data();
    let gy_data = gradient.gy.data();

    let thresh_sq = gradient_threshold * gradient_threshold;

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

            // Warp the unit gradient direction by the affine map
            let wx = affine.a * dx + affine.b * dy;
            let wy = affine.c * dx + affine.d * dy;
            let wmag = (wx * wx + wy * wy).sqrt();
            if wmag < 1e-8 {
                continue;
            }
            let wx = wx / wmag;
            let wy = wy / wmag;

            // Vote at the warped offset position
            let px = x as i32 + (wx * n).round() as i32;
            let py = y as i32 + (wy * n).round() as i32;

            if px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h {
                acc_data[py as usize * w + px as usize] += mag;
            }
        }
    }

    // Gaussian smoothing
    let sigma = smoothing_factor * radius as Scalar;
    if sigma > 0.5 {
        crate::core::blur::gaussian_blur_inplace(&mut acc, sigma);
    }

    Ok(acc)
}

/// Compute affine FRST responses for all configured affine maps.
///
/// Returns a vector of [`AffineResponse`], one per affine map, sorted
/// by descending peak value.
pub fn affine_frst_responses(
    gradient: &GradientField,
    config: &AffineFrstConfig,
) -> Result<Vec<AffineResponse>> {
    let mut responses = Vec::with_capacity(config.affine_maps.len());

    for affine in &config.affine_maps {
        let response = affine_frst_response_single(
            gradient,
            config.radius,
            affine,
            config.gradient_threshold,
            config.smoothing_factor,
        )?;

        let peak_value = response.data().iter().copied().fold(0.0f32, Scalar::max);

        responses.push(AffineResponse {
            response,
            affine_map: *affine,
            peak_value,
        });
    }

    // Sort by descending peak value
    responses.sort_by(|a, b| {
        b.peak_value
            .partial_cmp(&a.peak_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(responses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;
    use crate::core::nms::{non_maximum_suppression, NmsConfig};

    fn make_ellipse_image(size: usize, cx: f32, cy: f32, a: f32, b: f32, angle: f32) -> Vec<u8> {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                // Rotate to ellipse-local frame
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;
                if (lx / a).powi(2) + (ly / b).powi(2) <= 1.0 {
                    data[y * size + x] = 255;
                }
            }
        }
        data
    }

    #[test]
    fn affine_responses_sorted_by_peak() {
        let size = 80;
        let data = make_ellipse_image(size, 40.0, 40.0, 12.0, 8.0, 0.3);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = AffineFrstConfig {
            radius: 10,
            affine_maps: super::super::transform::sample_affine_maps(4, 2),
            ..AffineFrstConfig::default()
        };

        let responses = affine_frst_responses(&grad, &config).unwrap();

        assert_eq!(responses.len(), 8);
        // Should be sorted by descending peak
        for pair in responses.windows(2) {
            assert!(pair[0].peak_value >= pair[1].peak_value);
        }
    }

    #[test]
    fn best_affine_finds_ellipse_center() {
        let size = 80;
        let cx = 40.0;
        let cy = 40.0;
        let data = make_ellipse_image(size, cx, cy, 12.0, 8.0, 0.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = AffineFrstConfig {
            radius: 10,
            ..AffineFrstConfig::default()
        };

        let responses = affine_frst_responses(&grad, &config).unwrap();
        assert!(!responses.is_empty());

        // The best response should have a peak near the ellipse center
        let best = &responses[0];
        let peaks = non_maximum_suppression(
            &best.response.view(),
            &NmsConfig {
                radius: 5,
                threshold: 0.0,
                max_detections: 1,
            },
        );

        assert!(!peaks.is_empty(), "should find a peak");
        let err = ((peaks[0].position.x - cx).powi(2) + (peaks[0].position.y - cy).powi(2)).sqrt();
        assert!(
            err < 8.0,
            "peak at ({}, {}) too far from center ({cx}, {cy}), error={err}",
            peaks[0].position.x,
            peaks[0].position.y
        );
    }
}
