//! Iterative ellipse refinement.
//!
//! Refines an ellipse hypothesis (center, semi-axes, orientation) using
//! gradient evidence from an elliptical annular sampling region. Uses
//! the radial center method for center refinement, then fits ellipse
//! parameters to the gradient sample distribution.

use crate::core::coords::PixelCoord;
use crate::core::geometry::Ellipse;
use crate::core::gradient::GradientField;
use crate::core::scalar::Scalar;
use crate::support::annulus::{sample_elliptical_annulus, AnnulusSamplingConfig};
use crate::support::evidence::GradientSample;

use super::radial_center::{radial_center_refine_from_gradient, RadialCenterConfig};
use super::result::{RefinementResult, RefinementStatus};

/// Configuration for iterative ellipse refinement.
#[derive(Debug, Clone)]
pub struct EllipseRefineConfig {
    /// Maximum number of refinement iterations.
    pub max_iterations: usize,
    /// Convergence tolerance: stop when center shift < this (pixels).
    pub convergence_tol: Scalar,
    /// Fractional annulus margin around the hypothesized ellipse.
    pub annulus_margin: Scalar,
    /// Radial center config for center refinement.
    pub radial_center: RadialCenterConfig,
    /// Annulus sampling config.
    pub sampling: AnnulusSamplingConfig,
    /// Minimum alignment for a sample to contribute to ellipse fitting.
    pub min_alignment: Scalar,
}

impl Default for EllipseRefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            convergence_tol: 0.1,
            annulus_margin: 0.3,
            radial_center: RadialCenterConfig::default(),
            sampling: AnnulusSamplingConfig::default(),
            min_alignment: 0.3,
        }
    }
}

/// Fit ellipse parameters (semi-major, semi-minor, angle) to gradient samples.
///
/// Uses a covariance-based approach: computes the second-moment matrix of
/// well-aligned gradient sample positions relative to the center, then
/// extracts semi-axes and orientation from eigenvalues/eigenvectors.
fn fit_ellipse_from_samples(
    samples: &[GradientSample],
    center: PixelCoord,
    min_alignment: Scalar,
) -> Option<(Scalar, Scalar, Scalar)> {
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    let mut count = 0u32;

    for s in samples {
        if s.radial_alignment < min_alignment {
            continue;
        }
        let dx = (s.position.x - center.x) as f64;
        let dy = (s.position.y - center.y) as f64;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        count += 1;
    }

    if count < 5 {
        return None;
    }

    let n = count as f64;
    let cxx = sxx / n;
    let cyy = syy / n;
    let cxy = sxy / n;

    // Eigenvalues of the 2x2 covariance matrix
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (trace * trace - 4.0 * det).max(0.0);
    let sqrt_disc = disc.sqrt();

    let lambda1 = (trace + sqrt_disc) / 2.0;
    let lambda2 = (trace - sqrt_disc) / 2.0;

    if lambda1 < 1e-8 || lambda2 < 1e-8 {
        return None;
    }

    // Semi-axes are proportional to sqrt of eigenvalues
    let semi_major = (lambda1.sqrt()) as Scalar;
    let semi_minor = (lambda2.sqrt()) as Scalar;

    // Orientation from eigenvector of larger eigenvalue
    let angle = if cxy.abs() > 1e-10 {
        (lambda1 - cxx).atan2(cxy) as Scalar
    } else if cxx >= cyy {
        0.0
    } else {
        std::f32::consts::FRAC_PI_2
    };

    Some((semi_major, semi_minor, angle))
}

/// Iteratively refine an ellipse hypothesis.
///
/// Each iteration:
/// 1. Refines the center using the radial center method.
/// 2. Samples the elliptical annulus around the current estimate.
/// 3. Re-fits ellipse parameters from the gradient sample distribution.
pub fn refine_ellipse(
    gradient: &GradientField,
    initial: &Ellipse,
    config: &EllipseRefineConfig,
) -> RefinementResult<Ellipse> {
    let mut center = initial.center;
    let mut semi_major = initial.semi_major;
    let mut semi_minor = initial.semi_minor;
    let mut angle = initial.angle;

    for iter in 0..config.max_iterations {
        // Step 1: Refine center
        let center_result =
            radial_center_refine_from_gradient(gradient, center, &config.radial_center);

        match center_result.status {
            RefinementStatus::Converged => {}
            RefinementStatus::Degenerate => {
                return RefinementResult {
                    hypothesis: Ellipse::new(center, semi_major, semi_minor, angle),
                    status: RefinementStatus::Degenerate,
                    residual: center_result.residual,
                    iterations: iter,
                };
            }
            RefinementStatus::OutOfBounds => {
                return RefinementResult {
                    hypothesis: Ellipse::new(center, semi_major, semi_minor, angle),
                    status: RefinementStatus::OutOfBounds,
                    residual: center_result.residual,
                    iterations: iter,
                };
            }
            RefinementStatus::MaxIterations => {}
        }

        let new_center = center_result.hypothesis;

        // Step 2: Sample elliptical annulus
        let current = Ellipse::new(new_center, semi_major, semi_minor, angle);
        let inner_scale = (1.0 - config.annulus_margin).max(0.1);
        let outer_scale = 1.0 + config.annulus_margin;
        let evidence = sample_elliptical_annulus(
            gradient,
            &current,
            inner_scale,
            outer_scale,
            &config.sampling,
        );

        // Step 3: Fit ellipse from gradient samples
        if evidence.sample_count > 8 {
            if let Some((new_a, new_b, new_angle)) = fit_ellipse_from_samples(
                &evidence.gradient_samples,
                new_center,
                config.min_alignment,
            ) {
                semi_major = new_a;
                semi_minor = new_b;
                angle = new_angle;
            }
        }

        // Check convergence
        let shift = ((new_center.x - center.x).powi(2) + (new_center.y - center.y).powi(2)).sqrt();
        center = new_center;

        if shift < config.convergence_tol {
            return RefinementResult {
                hypothesis: Ellipse::new(center, semi_major, semi_minor, angle),
                status: RefinementStatus::Converged,
                residual: shift,
                iterations: iter + 1,
            };
        }
    }

    RefinementResult {
        hypothesis: Ellipse::new(center, semi_major, semi_minor, angle),
        status: RefinementStatus::MaxIterations,
        residual: 0.0,
        iterations: config.max_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn refine_circle_as_ellipse() {
        let size = 100;
        let true_cx = 50.0;
        let true_cy = 50.0;
        let data = make_ring(size, true_cx, true_cy, 18.0, 22.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let initial = Ellipse::new(PixelCoord::new(52.0, 48.0), 20.0, 20.0, 0.0);
        let config = EllipseRefineConfig {
            radial_center: RadialCenterConfig {
                patch_radius: 25,
                ..RadialCenterConfig::default()
            },
            ..EllipseRefineConfig::default()
        };
        let result = refine_ellipse(&grad, &initial, &config);

        assert!(
            result.converged() || result.status == RefinementStatus::MaxIterations,
            "unexpected status: {:?}",
            result.status
        );

        let e = &result.hypothesis;
        let center_err = ((e.center.x - true_cx).powi(2) + (e.center.y - true_cy).powi(2)).sqrt();
        assert!(center_err < 3.0, "center error {center_err} too large");

        // For a circle, semi-major and semi-minor should be roughly equal
        let axis_ratio = e.semi_minor / e.semi_major;
        assert!(
            axis_ratio > 0.7,
            "expected near-circular, got ratio {axis_ratio}"
        );
    }

    #[test]
    fn degenerate_on_empty() {
        let data = vec![0u8; 100 * 100];
        let image = ImageView::from_slice(&data, 100, 100).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let initial = Ellipse::new(PixelCoord::new(50.0, 50.0), 15.0, 15.0, 0.0);
        let result = refine_ellipse(&grad, &initial, &EllipseRefineConfig::default());
        assert_eq!(result.status, RefinementStatus::Degenerate);
    }
}
