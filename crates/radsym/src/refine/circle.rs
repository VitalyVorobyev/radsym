//! Iterative circle refinement.
//!
//! Refines a circle hypothesis (center + radius) using gradient evidence
//! from an annular sampling region. Each iteration re-samples the annulus,
//! updates the center via the radial-center method, and re-estimates the
//! radius from the gradient peak distribution.

use crate::core::error::Result;
use crate::core::geometry::Circle;
use crate::core::gradient::GradientField;
use crate::core::scalar::Scalar;
use crate::support::annulus::{AnnulusSamplingConfig, sample_annulus};

use super::radial_center::RadialCenterConfig;
use super::radial_center::radial_center_refine_from_gradient;
use super::result::{RefinementResult, RefinementStatus};

/// Configuration for iterative circle refinement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CircleRefineConfig {
    /// Maximum number of refinement iterations.
    pub max_iterations: usize,
    /// Convergence tolerance: stop when center shift < this (pixels).
    pub convergence_tol: Scalar,
    /// Fractional annulus margin around the hypothesized radius.
    pub annulus_margin: Scalar,
    /// Radial center config for center refinement sub-step.
    pub radial_center: RadialCenterConfig,
    /// Annulus sampling config for radius estimation.
    pub sampling: AnnulusSamplingConfig,
}

impl CircleRefineConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> crate::core::error::Result<()> {
        use crate::core::error::RadSymError;
        if self.max_iterations == 0 {
            return Err(RadSymError::InvalidConfig {
                reason: "max_iterations must be > 0",
            });
        }
        if self.convergence_tol <= 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "convergence_tol must be > 0.0",
            });
        }
        if self.annulus_margin <= 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "annulus_margin must be > 0.0",
            });
        }
        Ok(())
    }
}

impl Default for CircleRefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            convergence_tol: 0.1,
            annulus_margin: 0.3,
            radial_center: RadialCenterConfig::default(),
            sampling: AnnulusSamplingConfig::default(),
        }
    }
}

/// Iteratively refine a circle hypothesis.
///
/// Each iteration:
/// 1. Refines the center using the radial center method on the gradient field.
/// 2. Samples the annulus around the current estimate.
/// 3. Re-estimates the radius from the peak of the radial gradient distribution.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, Circle, PixelCoord, sobel_gradient, refine_circle};
/// use radsym::refine::circle::CircleRefineConfig;
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
/// let circle = Circle::new(PixelCoord::new(33.0, 33.0), 10.0);
/// let result = refine_circle(&grad, &circle, &CircleRefineConfig::default()).unwrap();
/// let dx = result.hypothesis.center.x - 32.0;
/// let dy = result.hypothesis.center.y - 32.0;
/// assert!((dx * dx + dy * dy).sqrt() < 3.0, "refined center should be near (32, 32)");
/// ```
pub fn refine_circle(
    gradient: &GradientField,
    initial: &Circle,
    config: &CircleRefineConfig,
) -> Result<RefinementResult<Circle>> {
    config.validate()?;
    let mut center = initial.center;
    let mut radius = initial.radius;

    for iter in 0..config.max_iterations {
        // Step 1: Refine center
        let center_result =
            radial_center_refine_from_gradient(gradient, center, &config.radial_center)?;

        match center_result.status {
            RefinementStatus::Converged => {}
            RefinementStatus::Degenerate => {
                return Ok(RefinementResult {
                    hypothesis: Circle::new(center, radius),
                    status: RefinementStatus::Degenerate,
                    residual: center_result.residual,
                    iterations: iter,
                });
            }
            RefinementStatus::OutOfBounds => {
                return Ok(RefinementResult {
                    hypothesis: Circle::new(center, radius),
                    status: RefinementStatus::OutOfBounds,
                    residual: center_result.residual,
                    iterations: iter,
                });
            }
            RefinementStatus::MaxIterations => {}
        }

        let new_center = center_result.hypothesis;

        // Step 2: Re-estimate radius from gradient evidence
        let inner_r = (radius * (1.0 - config.annulus_margin)).max(1.0);
        let outer_r = radius * (1.0 + config.annulus_margin);
        let evidence = sample_annulus(gradient, new_center, inner_r, outer_r, &config.sampling);

        if evidence.sample_count > 4 {
            // Estimate radius as the mean distance of well-aligned gradient samples
            let mut r_sum = 0.0;
            let mut r_count = 0;
            for s in &evidence.gradient_samples {
                if s.radial_alignment > 0.3 {
                    let dx = s.position.x - new_center.x;
                    let dy = s.position.y - new_center.y;
                    r_sum += (dx * dx + dy * dy).sqrt();
                    r_count += 1;
                }
            }
            if r_count > 0 {
                radius = r_sum / r_count as Scalar;
            }
        }

        // Check convergence
        let shift = ((new_center.x - center.x).powi(2) + (new_center.y - center.y).powi(2)).sqrt();
        center = new_center;

        if shift < config.convergence_tol {
            return Ok(RefinementResult {
                hypothesis: Circle::new(center, radius),
                status: RefinementStatus::Converged,
                residual: shift,
                iterations: iter + 1,
            });
        }
    }

    Ok(RefinementResult {
        hypothesis: Circle::new(center, radius),
        status: RefinementStatus::MaxIterations,
        residual: 0.0,
        iterations: config.max_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::coords::PixelCoord;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;

    fn make_ring(size: usize, cx: f32, cy: f32, r_inner: f32, r_outer: f32) -> Vec<u8> {
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
    fn refine_from_noisy_initial() {
        let size = 100;
        let true_cx = 50.0;
        let true_cy = 50.0;
        let true_r = 20.0;
        let data = make_ring(size, true_cx, true_cy, 18.0, 22.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        // Start with a noisy estimate; use large patch to cover ring radius
        let initial = Circle::new(PixelCoord::new(52.0, 48.0), 18.0);
        let config = CircleRefineConfig {
            radial_center: RadialCenterConfig {
                patch_radius: 25,
                ..RadialCenterConfig::default()
            },
            ..CircleRefineConfig::default()
        };
        let result = refine_circle(&grad, &initial, &config).unwrap();

        assert!(
            result.converged() || result.status == RefinementStatus::MaxIterations,
            "unexpected status: {:?}",
            result.status
        );

        let c = &result.hypothesis;
        let center_err = ((c.center.x - true_cx).powi(2) + (c.center.y - true_cy).powi(2)).sqrt();
        assert!(
            center_err < 3.0,
            "center error {center_err} too large, refined=({}, {})",
            c.center.x,
            c.center.y
        );
        assert!(
            (c.radius - true_r).abs() < 3.0,
            "radius error {} too large, refined={}",
            (c.radius - true_r).abs(),
            c.radius
        );
    }

    #[test]
    fn default_config_passes_validation() {
        CircleRefineConfig::default().validate().unwrap();
    }

    #[test]
    fn zero_max_iterations_fails_validation() {
        let config = CircleRefineConfig {
            max_iterations: 0,
            ..CircleRefineConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn zero_convergence_tol_fails_validation() {
        let config = CircleRefineConfig {
            convergence_tol: 0.0,
            ..CircleRefineConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn degenerate_on_empty() {
        let data = vec![0u8; 100 * 100];
        let image = ImageView::from_slice(&data, 100, 100).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let initial = Circle::new(PixelCoord::new(50.0, 50.0), 15.0);
        let result = refine_circle(&grad, &initial, &CircleRefineConfig::default()).unwrap();
        assert_eq!(result.status, RefinementStatus::Degenerate);
    }
}
