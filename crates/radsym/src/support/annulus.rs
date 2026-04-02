//! Annulus sampling for local support extraction.
//!
//! Samples gradient vectors along circular or elliptical annuli around a
//! hypothesis center, producing [`SupportEvidence`] for scoring.

use crate::core::coords::PixelCoord;
use crate::core::geometry::Ellipse;
use crate::core::gradient::GradientField;
use crate::core::scalar::Scalar;

use super::evidence::{GradientSample, SupportEvidence};

/// Configuration for annulus sampling.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnnulusSamplingConfig {
    /// Number of angular samples around the annulus.
    pub num_angular_samples: usize,
    /// Number of radial samples across the annulus width.
    pub num_radial_samples: usize,
}

impl AnnulusSamplingConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> crate::core::error::Result<()> {
        use crate::core::error::RadSymError;
        if self.num_angular_samples < 4 {
            return Err(RadSymError::InvalidConfig {
                reason: "num_angular_samples must be >= 4",
            });
        }
        if self.num_radial_samples < 1 {
            return Err(RadSymError::InvalidConfig {
                reason: "num_radial_samples must be >= 1",
            });
        }
        Ok(())
    }
}

impl Default for AnnulusSamplingConfig {
    fn default() -> Self {
        Self {
            num_angular_samples: 64,
            num_radial_samples: 9,
        }
    }
}

/// Sample gradients along a circular annulus.
///
/// Samples at evenly spaced angles around the annulus, at multiple radial
/// offsets between `inner_radius` and `outer_radius`. For each sample,
/// records the gradient vector and its alignment with the radial direction.
pub fn sample_annulus(
    gradient: &GradientField,
    center: PixelCoord,
    inner_radius: Scalar,
    outer_radius: Scalar,
    config: &AnnulusSamplingConfig,
) -> SupportEvidence {
    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let mut samples = Vec::new();
    let mut alignment_sum = 0.0f32;

    let n_ang = config.num_angular_samples;
    let n_rad = config.num_radial_samples;

    for ai in 0..n_ang {
        let theta = 2.0 * std::f32::consts::PI * ai as Scalar / n_ang as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        for ri in 0..n_rad {
            let t = if n_rad <= 1 {
                0.5
            } else {
                ri as Scalar / (n_rad - 1) as Scalar
            };
            let r = inner_radius + t * (outer_radius - inner_radius);

            let sx = center.x + r * cos_t;
            let sy = center.y + r * sin_t;

            let Some(gx) = gx_view.sample(sx, sy) else {
                continue;
            };
            let Some(gy) = gy_view.sample(sx, sy) else {
                continue;
            };
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < 1e-8 {
                continue;
            }

            let alignment = ((gx * cos_t + gy * sin_t) / mag).abs();
            alignment_sum += alignment;

            samples.push(GradientSample {
                position: PixelCoord::new(sx, sy),
                gx,
                gy,
                radial_alignment: alignment,
            });
        }
    }

    let sample_count = samples.len();
    let mean_alignment = if sample_count > 0 {
        alignment_sum / sample_count as Scalar
    } else {
        0.0
    };

    let coverage = compute_circular_angular_coverage(&samples, center, config.num_angular_samples);
    SupportEvidence {
        gradient_samples: samples,
        angular_coverage: coverage,
        sample_count,
        mean_gradient_alignment: mean_alignment,
    }
}

/// Sample gradients along an elliptical annulus.
///
/// The ellipse is parameterized by semi-major/minor axes and angle. Samples
/// follow the ellipse boundary scaled between inner and outer factors.
pub fn sample_elliptical_annulus(
    gradient: &GradientField,
    ellipse: &Ellipse,
    inner_scale: Scalar,
    outer_scale: Scalar,
    config: &AnnulusSamplingConfig,
) -> SupportEvidence {
    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let mut samples = Vec::new();
    let mut alignment_sum = 0.0f32;

    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();

    let n_ang = config.num_angular_samples;
    let n_rad = config.num_radial_samples;

    for ai in 0..n_ang {
        let theta = 2.0 * std::f32::consts::PI * ai as Scalar / n_ang as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Ellipse point in local frame
        let ex = ellipse.semi_major * cos_t;
        let ey = ellipse.semi_minor * sin_t;

        for ri in 0..n_rad {
            let t = if n_rad <= 1 {
                0.5
            } else {
                ri as Scalar / (n_rad - 1) as Scalar
            };
            let scale = inner_scale + t * (outer_scale - inner_scale);

            // Scale the ellipse point
            let lx = scale * ex;
            let ly = scale * ey;

            // Rotate to image frame
            let sx = ellipse.center.x + lx * cos_a - ly * sin_a;
            let sy = ellipse.center.y + lx * sin_a + ly * cos_a;

            let Some(gx) = gx_view.sample(sx, sy) else {
                continue;
            };
            let Some(gy) = gy_view.sample(sx, sy) else {
                continue;
            };
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < 1e-8 {
                continue;
            }

            // True ellipse normal in local coordinates, scaled copies preserve direction.
            let nx_local = ellipse.semi_minor * cos_t;
            let ny_local = ellipse.semi_major * sin_t;
            let n_mag = (nx_local * nx_local + ny_local * ny_local).sqrt();
            if n_mag < 1e-8 {
                continue;
            }
            let nx = (nx_local * cos_a - ny_local * sin_a) / n_mag;
            let ny = (nx_local * sin_a + ny_local * cos_a) / n_mag;

            let alignment = ((gx * nx + gy * ny) / mag).abs();
            alignment_sum += alignment;

            samples.push(GradientSample {
                position: PixelCoord::new(sx, sy),
                gx,
                gy,
                radial_alignment: alignment,
            });
        }
    }

    let sample_count = samples.len();
    let mean_alignment = if sample_count > 0 {
        alignment_sum / sample_count as Scalar
    } else {
        0.0
    };

    let coverage =
        compute_elliptical_angular_coverage(&samples, ellipse, config.num_angular_samples);
    SupportEvidence {
        gradient_samples: samples,
        angular_coverage: coverage,
        sample_count,
        mean_gradient_alignment: mean_alignment,
    }
}

/// Estimate angular coverage from gradient samples.
///
/// Divides the annulus into `n_bins` angular bins and counts what fraction
/// of bins have at least one well-aligned sample (alignment > 0.5).
fn compute_circular_angular_coverage(
    samples: &[GradientSample],
    center: PixelCoord,
    n_bins: usize,
) -> Scalar {
    if samples.is_empty() || n_bins == 0 {
        return 0.0;
    }
    let mut bins = vec![false; n_bins];
    for s in samples {
        if s.radial_alignment > 0.5 {
            let angle = (s.position.y - center.y).atan2(s.position.x - center.x);
            let normalized = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let bin = (normalized * n_bins as Scalar) as usize;
            let bin = bin.min(n_bins - 1);
            bins[bin] = true;
        }
    }
    let filled = bins.iter().filter(|&&b| b).count();
    filled as Scalar / n_bins as Scalar
}

fn compute_elliptical_angular_coverage(
    samples: &[GradientSample],
    ellipse: &Ellipse,
    n_bins: usize,
) -> Scalar {
    if samples.is_empty() || n_bins == 0 || ellipse.semi_major <= 1e-6 || ellipse.semi_minor <= 1e-6
    {
        return 0.0;
    }

    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let mut bins = vec![false; n_bins];

    for s in samples {
        if s.radial_alignment <= 0.5 {
            continue;
        }

        let dx = s.position.x - ellipse.center.x;
        let dy = s.position.y - ellipse.center.y;
        let lx = dx * cos_a + dy * sin_a;
        let ly = -dx * sin_a + dy * cos_a;
        let angle = (ly / ellipse.semi_minor).atan2(lx / ellipse.semi_major);
        let normalized = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
        let bin = ((normalized * n_bins as Scalar) as usize).min(n_bins - 1);
        bins[bin] = true;
    }

    let filled = bins.iter().filter(|&&b| b).count();
    filled as Scalar / n_bins as Scalar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::geometry::Ellipse;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;

    fn make_ring_image(size: usize, cx: f32, cy: f32, r_inner: f32, r_outer: f32) -> Vec<u8> {
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                if r >= r_inner && r <= r_outer {
                    data[y * size + x] = 255;
                }
            }
        }
        data
    }

    #[test]
    fn sample_annulus_on_ring() {
        let size = 80;
        let cx = 40.0;
        let cy = 40.0;
        let data = make_ring_image(size, cx, cy, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let evidence = sample_annulus(
            &grad,
            PixelCoord::new(cx, cy),
            12.0,
            20.0,
            &AnnulusSamplingConfig::default(),
        );

        assert!(evidence.sample_count > 0, "should have samples");
        assert!(
            evidence.mean_gradient_alignment > 0.3,
            "gradients on a ring should be radially aligned, got {}",
            evidence.mean_gradient_alignment
        );
    }

    #[test]
    fn sample_annulus_off_center() {
        let size = 80;
        let data = make_ring_image(size, 40.0, 40.0, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        // Sample far from the actual center — should have poor alignment
        let evidence = sample_annulus(
            &grad,
            PixelCoord::new(20.0, 20.0),
            12.0,
            20.0,
            &AnnulusSamplingConfig::default(),
        );

        // Off-center sampling should yield worse alignment than on-center
        let on_center = sample_annulus(
            &grad,
            PixelCoord::new(40.0, 40.0),
            12.0,
            20.0,
            &AnnulusSamplingConfig::default(),
        );

        assert!(
            on_center.mean_gradient_alignment > evidence.mean_gradient_alignment,
            "on-center ({}) should beat off-center ({})",
            on_center.mean_gradient_alignment,
            evidence.mean_gradient_alignment
        );
    }

    #[test]
    fn sample_elliptical_annulus_on_circle() {
        // An elliptical annulus with equal semi-axes should behave like circular
        let size = 80;
        let data = make_ring_image(size, 40.0, 40.0, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let ellipse = Ellipse::new(PixelCoord::new(40.0, 40.0), 16.0, 16.0, 0.0);
        let evidence = sample_elliptical_annulus(
            &grad,
            &ellipse,
            0.75, // inner_scale = 12/16
            1.25, // outer_scale = 20/16
            &AnnulusSamplingConfig::default(),
        );

        assert!(evidence.sample_count > 0);
        assert!(
            evidence.mean_gradient_alignment > 0.3,
            "expected good alignment, got {}",
            evidence.mean_gradient_alignment
        );
    }

    #[test]
    fn empty_region_yields_zero_evidence() {
        let data = vec![0u8; 64 * 64];
        let image = ImageView::from_slice(&data, 64, 64).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let evidence = sample_annulus(
            &grad,
            PixelCoord::new(32.0, 32.0),
            10.0,
            15.0,
            &AnnulusSamplingConfig::default(),
        );

        assert_eq!(evidence.sample_count, 0);
        assert_eq!(evidence.mean_gradient_alignment, 0.0);
    }
}
