//! Support scoring for circle and ellipse hypotheses.
//!
//! Produces a structured [`SupportScore`] that quantifies how strongly local
//! image evidence supports a given geometric hypothesis.

use crate::core::coords::PixelCoord;
use crate::core::geometry::{Circle, Ellipse};
use crate::core::gradient::GradientField;
use crate::core::scalar::Scalar;

use super::annulus::{sample_annulus, sample_elliptical_annulus, AnnulusSamplingConfig};
use super::coverage::angular_coverage;

/// Structured support score with component breakdown.
#[derive(Debug, Clone, Copy)]
pub struct SupportScore {
    /// Combined total score in `[0, 1]`.
    pub total: Scalar,
    /// Ring-like support strength (gradient alignment).
    pub ringness: Scalar,
    /// Polarity consistency (not yet fully implemented; reserved).
    pub polarity_consistency: Scalar,
    /// Angular coverage fraction in `[0, 1]`.
    pub angular_coverage: Scalar,
    /// True if the evidence is degenerate (insufficient samples).
    pub is_degenerate: bool,
}

/// Configuration for support scoring.
#[derive(Debug, Clone)]
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
        polarity_consistency: 1.0, // TODO: implement polarity analysis
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

    // For ellipse, estimate angular coverage using the mean radius
    let mean_radius = (ellipse.semi_major + ellipse.semi_minor) / 2.0;
    let cov = angular_coverage(
        gradient,
        ellipse.center,
        mean_radius,
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
        polarity_consistency: 1.0,
        angular_coverage: cov,
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
}
