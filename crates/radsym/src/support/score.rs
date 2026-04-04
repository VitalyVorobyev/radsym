//! Support scoring for circle and ellipse hypotheses.
//!
//! Produces a structured [`SupportScore`] that quantifies how strongly local
//! image evidence supports a given geometric hypothesis.

use crate::core::coords::PixelCoord;
use crate::core::geometry::{Circle, Ellipse};
use crate::core::gradient::GradientField;
use crate::core::homography::Homography;
use crate::core::scalar::Scalar;
use nalgebra::Vector2;

use super::annulus::{AnnulusSamplingConfig, sample_annulus, sample_elliptical_annulus};
use super::coverage::{angular_coverage, ellipse_angular_coverage};

/// Structured support score with component breakdown.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SupportScore {
    /// Combined total score in `[0, 1]`.
    pub total: Scalar,
    /// Ring-like support strength (gradient alignment).
    pub ringness: Scalar,
    /// Angular coverage fraction in `[0, 1]`.
    pub angular_coverage: Scalar,
    /// True if the evidence is degenerate (insufficient samples).
    pub is_degenerate: bool,
}

/// Configuration for support scoring.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoringConfig {
    /// Annulus sampling configuration.
    pub sampling: AnnulusSamplingConfig,
    /// How much wider than the hypothesized radius to sample (as a fraction).
    /// E.g., 0.3 means sample from `0.7*r` to `1.3*r`.
    pub annulus_margin: Scalar,
    /// Minimum number of gradient samples to avoid degeneracy flag.
    pub min_samples: usize,
    /// Weight of ringness component in total score.
    pub weight_ringness: Scalar,
    /// Weight of angular coverage in total score.
    pub weight_coverage: Scalar,
}

impl ScoringConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> crate::core::error::Result<()> {
        use crate::core::error::RadSymError;
        if self.annulus_margin <= 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "annulus_margin must be > 0.0",
            });
        }
        if self.min_samples == 0 {
            return Err(RadSymError::InvalidConfig {
                reason: "min_samples must be > 0",
            });
        }
        if self.weight_ringness < 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "weight_ringness must be >= 0.0",
            });
        }
        if self.weight_coverage < 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "weight_coverage must be >= 0.0",
            });
        }
        Ok(())
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            sampling: AnnulusSamplingConfig::default(),
            annulus_margin: 0.3,
            min_samples: 8,
            weight_ringness: 0.6,
            weight_coverage: 0.4,
        }
    }
}

/// Score how strongly the local image evidence supports a circle hypothesis.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, Circle, ScoringConfig, sobel_gradient};
/// use radsym::support::score::score_circle_support;
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
/// let circle = Circle::new(radsym::PixelCoord::new(32.0, 32.0), 10.0);
/// let score = score_circle_support(&grad, &circle, &ScoringConfig::default());
/// assert!(!score.is_degenerate);
/// assert!(score.total > 0.0);
/// ```
pub fn score_circle_support(
    gradient: &GradientField,
    circle: &Circle,
    config: &ScoringConfig,
) -> SupportScore {
    let inner_r = circle.radius * (1.0 - config.annulus_margin);
    let outer_r = circle.radius * (1.0 + config.annulus_margin);

    let evidence = sample_annulus(
        gradient,
        circle.center,
        inner_r.max(1.0),
        outer_r,
        &config.sampling,
    );

    let is_degenerate = evidence.sample_count < config.min_samples;
    let ringness = evidence.mean_gradient_alignment;

    let cov = angular_coverage(
        gradient,
        circle.center,
        circle.radius,
        0.5,
        config.sampling.num_angular_samples,
    );

    let total = if is_degenerate {
        0.0
    } else {
        (config.weight_ringness * ringness + config.weight_coverage * cov).clamp(0.0, 1.0)
    };

    SupportScore {
        total,
        ringness,

        angular_coverage: cov,
        is_degenerate,
    }
}

/// Score how strongly the local image evidence supports an ellipse hypothesis.
pub fn score_ellipse_support(
    gradient: &GradientField,
    ellipse: &Ellipse,
    config: &ScoringConfig,
) -> SupportScore {
    let inner_scale = 1.0 - config.annulus_margin;
    let outer_scale = 1.0 + config.annulus_margin;

    let evidence = sample_elliptical_annulus(
        gradient,
        ellipse,
        inner_scale.max(0.1),
        outer_scale,
        &config.sampling,
    );

    let is_degenerate = evidence.sample_count < config.min_samples;
    let ringness = evidence.mean_gradient_alignment;

    let cov = ellipse_angular_coverage(gradient, ellipse, 0.5, config.sampling.num_angular_samples);

    let total = if is_degenerate {
        0.0
    } else {
        (config.weight_ringness * ringness + config.weight_coverage * cov).clamp(0.0, 1.0)
    };

    SupportScore {
        total,
        ringness,

        angular_coverage: cov,
        is_degenerate,
    }
}

/// Score support for a rectified-frame circle by sampling in rectified angle
/// and evaluating the pulled-back normal alignment in image space.
pub fn score_rectified_circle_support(
    gradient: &GradientField,
    rectified_circle: &Circle,
    homography: &Homography,
    config: &ScoringConfig,
) -> SupportScore {
    if rectified_circle.radius <= 1e-6 {
        return SupportScore {
            total: 0.0,
            ringness: 0.0,

            angular_coverage: 0.0,
            is_degenerate: true,
        };
    }

    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let inner_radius = rectified_circle.radius * (1.0 - config.annulus_margin);
    let outer_radius = rectified_circle.radius * (1.0 + config.annulus_margin);
    let n_ang = config.sampling.num_angular_samples.max(1);
    let n_rad = config.sampling.num_radial_samples.max(1);
    let mut sample_count = 0usize;
    let mut alignment_sum = 0.0;
    let mut bins = vec![false; n_ang];

    for (ai, bin) in bins.iter_mut().enumerate() {
        let theta = 2.0 * std::f32::consts::PI * ai as Scalar / n_ang as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let rectified_normal = Vector2::new(cos_t, sin_t);

        for ri in 0..n_rad {
            let t = if n_rad <= 1 {
                0.5
            } else {
                ri as Scalar / (n_rad - 1) as Scalar
            };
            let radius = inner_radius + t * (outer_radius - inner_radius);
            let rectified_point = PixelCoord::new(
                rectified_circle.center.x + radius * cos_t,
                rectified_circle.center.y + radius * sin_t,
            );
            let Some(image_point) = homography.map_rectified_to_image(rectified_point) else {
                continue;
            };
            let Some(gx) = gx_view.sample(image_point.x, image_point.y) else {
                continue;
            };
            let Some(gy) = gy_view.sample(image_point.x, image_point.y) else {
                continue;
            };
            let gradient_mag = (gx * gx + gy * gy).sqrt();
            if gradient_mag <= 1e-8 {
                continue;
            }
            let Some(image_normal) =
                homography.pullback_rectified_normal_to_image(rectified_point, rectified_normal)
            else {
                continue;
            };
            let normal_mag = image_normal.norm();
            if normal_mag <= 1e-8 {
                continue;
            }

            let alignment =
                ((gx * image_normal[0] + gy * image_normal[1]) / (gradient_mag * normal_mag)).abs();
            alignment_sum += alignment;
            sample_count += 1;
            if alignment > 0.5 {
                *bin = true;
            }
        }
    }

    let is_degenerate = sample_count < config.min_samples;
    let ringness = if sample_count > 0 {
        alignment_sum / sample_count as Scalar
    } else {
        0.0
    };
    let coverage = bins.iter().filter(|&&filled| filled).count() as Scalar / n_ang as Scalar;
    let total = if is_degenerate {
        0.0
    } else {
        (config.weight_ringness * ringness + config.weight_coverage * coverage).clamp(0.0, 1.0)
    };

    SupportScore {
        total,
        ringness,

        angular_coverage: coverage,
        is_degenerate,
    }
}

/// Score support for a circle hypothesis given a precomputed center.
///
/// Convenience function that constructs a [`Circle`] from center and radius.
pub fn score_at(
    gradient: &GradientField,
    center: PixelCoord,
    radius: Scalar,
    config: &ScoringConfig,
) -> SupportScore {
    let circle = Circle::new(center, radius);
    score_circle_support(gradient, &circle, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::homography::{Homography, rectified_circle_to_image_ellipse};
    use crate::core::image_view::ImageView;

    fn make_ring_u8(size: usize, cx: f32, cy: f32, r_inner: f32, r_outer: f32) -> Vec<u8> {
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
    fn high_score_for_centered_circle() {
        let size = 80;
        let cx = 40.0;
        let cy = 40.0;
        let data = make_ring_u8(size, cx, cy, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let circle = Circle::new(PixelCoord::new(cx, cy), 16.0);
        let score = score_circle_support(&grad, &circle, &ScoringConfig::default());

        assert!(!score.is_degenerate, "should not be degenerate");
        assert!(
            score.ringness > 0.3,
            "ringness should be high for centered circle, got {}",
            score.ringness
        );
        assert!(
            score.total > 0.2,
            "total should be positive, got {}",
            score.total
        );
    }

    #[test]
    fn low_score_for_offset_circle() {
        let size = 80;
        let data = make_ring_u8(size, 40.0, 40.0, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let centered = Circle::new(PixelCoord::new(40.0, 40.0), 16.0);
        let offset = Circle::new(PixelCoord::new(20.0, 20.0), 16.0);

        let score_centered = score_circle_support(&grad, &centered, &ScoringConfig::default());
        let score_offset = score_circle_support(&grad, &offset, &ScoringConfig::default());

        assert!(
            score_centered.total > score_offset.total,
            "centered ({}) should score higher than offset ({})",
            score_centered.total,
            score_offset.total
        );
    }

    #[test]
    fn degenerate_on_empty_image() {
        let data = vec![0u8; 64 * 64];
        let image = ImageView::from_slice(&data, 64, 64).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let circle = Circle::new(PixelCoord::new(32.0, 32.0), 10.0);
        let score = score_circle_support(&grad, &circle, &ScoringConfig::default());

        assert!(score.is_degenerate, "empty image should be degenerate");
        assert_eq!(score.total, 0.0);
    }

    #[test]
    fn ellipse_score_on_circle() {
        let size = 80;
        let cx = 40.0;
        let cy = 40.0;
        let data = make_ring_u8(size, cx, cy, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let ellipse = Ellipse::new(PixelCoord::new(cx, cy), 16.0, 16.0, 0.0);
        let score = score_ellipse_support(&grad, &ellipse, &ScoringConfig::default());

        assert!(!score.is_degenerate);
        assert!(
            score.total > 0.2,
            "ellipse on circle should score well, got {}",
            score.total
        );
    }

    #[test]
    fn score_at_convenience() {
        let size = 80;
        let data = make_ring_u8(size, 40.0, 40.0, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let score = score_at(
            &grad,
            PixelCoord::new(40.0, 40.0),
            16.0,
            &ScoringConfig::default(),
        );
        assert!(score.total > 0.2);
    }

    #[test]
    fn high_score_for_rectified_circle_under_homography() {
        let homography = Homography::new([
            [1.1, 0.06, 16.0],
            [0.03, 0.98, 12.0],
            [0.0011, -0.0007, 1.0],
        ])
        .unwrap();
        let rectified_circle = Circle::new(PixelCoord::new(64.0, 60.0), 16.0);
        let ellipse = rectified_circle_to_image_ellipse(&homography, &rectified_circle).unwrap();
        let size = 128;
        let cos_a = ellipse.angle.cos();
        let sin_a = ellipse.angle.sin();
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - ellipse.center.x;
                let dy = y as f32 - ellipse.center.y;
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;
                if (lx / ellipse.semi_major).powi(2) + (ly / ellipse.semi_minor).powi(2) <= 1.0 {
                    data[y * size + x] = 255;
                }
            }
        }
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let score = score_rectified_circle_support(
            &gradient,
            &rectified_circle,
            &homography,
            &ScoringConfig::default(),
        );
        assert!(!score.is_degenerate);
        assert!(
            score.total > 0.2,
            "expected positive homography-aware score, got {}",
            score.total
        );
        assert!(
            score.angular_coverage > 0.5,
            "expected usable rectified coverage, got {}",
            score.angular_coverage
        );
    }
}
