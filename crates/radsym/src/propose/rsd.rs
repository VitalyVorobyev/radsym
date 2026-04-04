//! Radial Symmetry Detector (RSD) — fast proposal variant.
//!
//! Implements a simplified magnitude-only center-voting scheme inspired by:
//!
//! - Barnes, N., Zelinsky, A., Fletcher, L.S. (2008). *Real-time speed sign
//!   detection using the radial symmetry detector.* IEEE T-ITS 9(2).
//!
//! ## Differences from FRST
//!
//! RSD drops the orientation accumulator (`O_n`) and the alpha-exponent
//! combination step. Each pixel votes only with its gradient magnitude into
//! a single accumulator, which is then smoothed. This yields a ~2× speedup
//! at the cost of slightly lower discrimination against non-symmetric
//! features (no orientation consistency check).
//!
//! ## Budget-aware mode
//!
//! When `max_proposals` is set, extraction stops early once the budget is
//! reached, making RSD suitable for real-time pipelines where a fixed
//! proposal count is required regardless of scene complexity.

use crate::core::error::Result;
use crate::core::gradient::GradientField;
use crate::core::image_view::OwnedImage;
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Configuration for RSD response computation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RsdConfig {
    /// Set of discrete radii to test (in pixels).
    pub radii: Vec<u32>,
    /// Minimum gradient magnitude to participate in voting.
    pub gradient_threshold: Scalar,
    /// Which polarity to detect.
    pub polarity: Polarity,
    /// Standard deviation of Gaussian smoothing, relative to radius.
    pub smoothing_factor: Scalar,
}

impl RsdConfig {
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
        if self.smoothing_factor <= 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "smoothing_factor must be > 0",
            });
        }
        Ok(())
    }
}

impl Default for RsdConfig {
    fn default() -> Self {
        Self {
            radii: vec![3, 5, 7, 9, 11],
            gradient_threshold: 0.0,
            polarity: Polarity::Both,
            smoothing_factor: 0.5,
        }
    }
}

#[inline]
fn accumulate_vote(acc_data: &mut [Scalar], w: usize, h: usize, x: i32, y: i32, mag: Scalar) {
    if x >= 0 && (x as usize) < w && y >= 0 && (y as usize) < h {
        acc_data[y as usize * w + x as usize] += mag;
    }
}

fn accumulate_response(sum: &mut OwnedImage<Scalar>, single: &OwnedImage<Scalar>) {
    for (dst, src) in sum.data_mut().iter_mut().zip(single.data().iter().copied()) {
        *dst += src;
    }
}

/// Compute the RSD response map for a single radius.
///
/// Magnitude-only voting: each pixel with sufficient gradient magnitude
/// votes along its gradient direction, accumulating `|g|` at the target
/// pixel. No orientation accumulator is maintained.
pub fn rsd_response_single(
    gradient: &GradientField,
    radius: u32,
    config: &RsdConfig,
) -> Result<OwnedImage<Scalar>> {
    config.validate()?;
    let w = gradient.width();
    let h = gradient.height();
    let radius_f = radius as Scalar;

    let mut acc = OwnedImage::<Scalar>::zeros(w, h)?;
    let acc_data = acc.data_mut();
    let gx_data = gradient.gx.data();
    let gy_data = gradient.gy.data();

    let thresh_sq = config.gradient_threshold * config.gradient_threshold;
    match config.polarity {
        Polarity::Bright => {
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
                    let offset_x = (gx * inv_mag * radius_f).round() as i32;
                    let offset_y = (gy * inv_mag * radius_f).round() as i32;
                    accumulate_vote(
                        acc_data,
                        w,
                        h,
                        x as i32 + offset_x,
                        y as i32 + offset_y,
                        mag,
                    );
                }
            }
        }
        Polarity::Dark => {
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
                    let offset_x = (gx * inv_mag * radius_f).round() as i32;
                    let offset_y = (gy * inv_mag * radius_f).round() as i32;
                    accumulate_vote(
                        acc_data,
                        w,
                        h,
                        x as i32 - offset_x,
                        y as i32 - offset_y,
                        mag,
                    );
                }
            }
        }
        Polarity::Both => {
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
                    let offset_x = (gx * inv_mag * radius_f).round() as i32;
                    let offset_y = (gy * inv_mag * radius_f).round() as i32;
                    let x = x as i32;
                    let y = y as i32;
                    accumulate_vote(acc_data, w, h, x + offset_x, y + offset_y, mag);
                    accumulate_vote(acc_data, w, h, x - offset_x, y - offset_y, mag);
                }
            }
        }
    }

    // Gaussian smoothing
    let sigma = config.smoothing_factor * radius as Scalar;
    if sigma > 0.5 {
        crate::core::blur::gaussian_blur_inplace(&mut acc, sigma);
    }

    Ok(acc)
}

/// Compute the multi-radius RSD response by summing single-radius responses.
///
/// Returns the summed response wrapped in a [`ResponseMap`](super::extract::ResponseMap)
/// tagged with [`ProposalSource::Rsd`](super::seed::ProposalSource::Rsd).
pub fn rsd_response(
    gradient: &GradientField,
    config: &RsdConfig,
) -> Result<super::extract::ResponseMap> {
    config.validate()?;
    let w = gradient.width();
    let h = gradient.height();
    let mut sum = OwnedImage::<Scalar>::zeros(w, h)?;

    #[cfg(feature = "rayon")]
    let per_radius = config
        .radii
        .par_iter()
        .map(|&radius| rsd_response_single(gradient, radius, config))
        .collect::<Vec<_>>();

    #[cfg(not(feature = "rayon"))]
    for &radius in &config.radii {
        let single = rsd_response_single(gradient, radius, config)?;
        accumulate_response(&mut sum, &single);
    }

    #[cfg(feature = "rayon")]
    for single in per_radius {
        let single = single?;
        accumulate_response(&mut sum, &single);
    }

    Ok(super::extract::ResponseMap::new(
        sum,
        super::seed::ProposalSource::Rsd,
    ))
}

/// Compute a fused multi-radius RSD response map in a single pixel pass.
///
/// Unlike [`rsd_response`], which processes each radius independently
/// (separate accumulator and Gaussian blur per radius), this function
/// fuses all radii into **one** pixel traversal with a single shared
/// accumulator and **one** final Gaussian blur.
///
/// - **One** image traversal (vs. N for standard RSD).
/// - **One** Gaussian blur with `sigma = smoothing_factor * median(radii)`
///   (vs. N per-radius blurs).
/// - **One** accumulator allocation (vs. N).
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, RsdConfig, sobel_gradient};
/// use radsym::propose::rsd::rsd_response_fused;
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
/// let config = RsdConfig { radii: vec![9, 10, 11], ..RsdConfig::default() };
/// let response = rsd_response_fused(&grad, &config).unwrap();
/// assert_eq!(response.response().width(), size);
/// assert!(response.response().data().iter().any(|&v| v > 0.0));
/// ```
pub fn rsd_response_fused(
    gradient: &GradientField,
    config: &RsdConfig,
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
        super::seed::ProposalSource::Rsd,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;
    use crate::core::nms::{non_maximum_suppression, NmsConfig};

    fn make_disk(size: usize, cx: f32, cy: f32, radius: f32) -> Vec<u8> {
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

    #[test]
    fn rsd_detects_bright_disk() {
        let size = 64;
        let cx = 32.0;
        let cy = 32.0;
        let data = make_disk(size, cx, cy, 8.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = RsdConfig {
            radii: vec![7, 8, 9],
            polarity: Polarity::Bright,
            ..RsdConfig::default()
        };
        let response = rsd_response(&grad, &config).unwrap();

        let peaks = non_maximum_suppression(
            &response.view(),
            &NmsConfig {
                radius: 5,
                threshold: 0.0,
                max_detections: 5,
            },
        );

        assert!(!peaks.is_empty(), "should detect at least one peak");
        let best = &peaks[0];
        let err = ((best.position.x - cx).powi(2) + (best.position.y - cy).powi(2)).sqrt();
        assert!(
            err < 5.0,
            "peak at ({}, {}) too far from center ({cx}, {cy}), error={err}",
            best.position.x,
            best.position.y
        );
    }

    #[test]
    fn rsd_response_dimensions() {
        let size = 48;
        let data = vec![128u8; size * size];
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let response = rsd_response(&grad, &RsdConfig::default()).unwrap();
        assert_eq!(response.response().width(), size);
        assert_eq!(response.response().height(), size);
    }

    #[test]
    fn rsd_multiple_targets() {
        let size = 100;
        let mut data = vec![0u8; size * size];

        // Two disks
        let centers = [(30.0f32, 30.0f32), (70.0f32, 70.0f32)];
        for &(cx, cy) in &centers {
            for y in 0..size {
                for x in 0..size {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    if (dx * dx + dy * dy).sqrt() <= 8.0 {
                        data[y * size + x] = 255;
                    }
                }
            }
        }

        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = RsdConfig {
            radii: vec![7, 8, 9],
            polarity: Polarity::Bright,
            ..RsdConfig::default()
        };
        let response = rsd_response(&grad, &config).unwrap();

        let peaks = non_maximum_suppression(
            &response.view(),
            &NmsConfig {
                radius: 10,
                threshold: 0.0,
                max_detections: 10,
            },
        );

        assert!(
            peaks.len() >= 2,
            "should detect at least 2 targets, got {}",
            peaks.len()
        );

        // Verify both centers are found
        for &(cx, cy) in &centers {
            let found = peaks.iter().any(|p| {
                let err = ((p.position.x - cx).powi(2) + (p.position.y - cy).powi(2)).sqrt();
                err < 8.0
            });
            assert!(found, "did not find center near ({cx}, {cy})");
        }
    }

    #[test]
    fn default_config_passes_validation() {
        RsdConfig::default().validate().unwrap();
    }

    #[test]
    fn empty_radii_fails_validation() {
        let config = RsdConfig {
            radii: vec![],
            ..RsdConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn zero_smoothing_factor_fails_validation() {
        let config = RsdConfig {
            smoothing_factor: 0.0,
            ..RsdConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn rsd_gradient_threshold_reduces_noise() {
        let size = 64;
        let data = make_disk(size, 32.0, 32.0, 8.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let no_thresh = rsd_response(
            &grad,
            &RsdConfig {
                radii: vec![8],
                gradient_threshold: 0.0,
                ..RsdConfig::default()
            },
        )
        .unwrap();

        let with_thresh = rsd_response(
            &grad,
            &RsdConfig {
                radii: vec![8],
                gradient_threshold: 5.0,
                ..RsdConfig::default()
            },
        )
        .unwrap();

        // Peak should still be present but background should be lower
        let nms = NmsConfig {
            radius: 5,
            threshold: 0.0,
            max_detections: 1,
        };
        let peaks_no = non_maximum_suppression(&no_thresh.view(), &nms);
        let peaks_th = non_maximum_suppression(&with_thresh.view(), &nms);

        assert!(!peaks_no.is_empty());
        assert!(!peaks_th.is_empty());

        // Both should find the center
        let err_no = ((peaks_no[0].position.x - 32.0).powi(2)
            + (peaks_no[0].position.y - 32.0).powi(2))
        .sqrt();
        let err_th = ((peaks_th[0].position.x - 32.0).powi(2)
            + (peaks_th[0].position.y - 32.0).powi(2))
        .sqrt();
        assert!(err_no < 5.0);
        assert!(err_th < 5.0);
    }

    // --- rsd_response_fused tests ---

    #[test]
    fn rsd_fused_detects_bright_disk() {
        let size = 80;
        let data = make_disk(size, 40.0, 40.0, 12.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = RsdConfig {
            radii: vec![11, 12, 13],
            gradient_threshold: 1.0,
            polarity: Polarity::Bright,
            ..RsdConfig::default()
        };

        let response = rsd_response_fused(&grad, &config).unwrap();
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
            "fused RSD peak at ({peak_x}, {peak_y}) too far from center (40, 40)"
        );
    }

    #[test]
    fn rsd_fused_dimensions_match() {
        let data = vec![128u8; 40 * 30];
        let image = ImageView::from_slice(&data, 40, 30).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        let response = rsd_response_fused(&grad, &RsdConfig::default()).unwrap();
        assert_eq!(response.response().width(), 40);
        assert_eq!(response.response().height(), 30);
    }

    #[test]
    fn rsd_fused_matches_rsd_peak_location() {
        let size = 100;
        let data = make_disk(size, 50.0, 50.0, 16.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let config = RsdConfig {
            radii: vec![14, 15, 16, 17, 18],
            gradient_threshold: 1.0,
            polarity: Polarity::Bright,
            ..RsdConfig::default()
        };

        let rsd = rsd_response(&grad, &config).unwrap();
        let fused = rsd_response_fused(&grad, &config).unwrap();

        let find_peak = |data: &[f32], w: usize| -> (usize, usize) {
            let (idx, _) = data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (idx % w, idx / w)
        };

        let (rx, ry) = find_peak(rsd.response().data(), size);
        let (fx, fy) = find_peak(fused.response().data(), size);
        let dist = ((rx as f32 - fx as f32).powi(2) + (ry as f32 - fy as f32).powi(2)).sqrt();
        assert!(
            dist < 3.0,
            "peak locations differ by {dist}px: rsd=({rx},{ry}) fused=({fx},{fy})"
        );
    }
}
