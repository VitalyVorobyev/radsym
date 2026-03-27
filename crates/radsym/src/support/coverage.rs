//! Angular support coverage analysis.
//!
//! Estimates what fraction of a circle/ellipse has consistent gradient support,
//! which is critical for distinguishing complete rings from partial arcs.

use crate::core::coords::PixelCoord;
use crate::core::geometry::Ellipse;
use crate::core::gradient::GradientField;
use crate::core::scalar::Scalar;

/// Estimate the angular coverage of gradient support around a center at a
/// given radius.
///
/// Divides the full circle into `num_bins` angular sectors, samples the
/// gradient at each angle, and counts the fraction of sectors where the
/// gradient is well-aligned with the radial direction (alignment > `tolerance`).
///
/// Returns a value in `[0, 1]`.
pub fn angular_coverage(
    gradient: &GradientField,
    center: PixelCoord,
    radius: Scalar,
    tolerance: Scalar,
    num_bins: usize,
) -> Scalar {
    if num_bins == 0 {
        return 0.0;
    }

    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let mut bins = vec![false; num_bins];

    for (bi, bin) in bins.iter_mut().enumerate() {
        let theta = 2.0 * std::f32::consts::PI * bi as Scalar / num_bins as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let sx = center.x + radius * cos_t;
        let sy = center.y + radius * sin_t;

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
        if alignment > tolerance {
            *bin = true;
        }
    }

    let filled = bins.iter().filter(|&&b| b).count();
    filled as Scalar / num_bins as Scalar
}

/// Estimate angular coverage of gradient support along an ellipse boundary.
///
/// Samples the boundary at evenly spaced ellipse-relative angles and checks
/// whether the local gradient aligns with the true ellipse normal.
pub fn ellipse_angular_coverage(
    gradient: &GradientField,
    ellipse: &Ellipse,
    tolerance: Scalar,
    num_bins: usize,
) -> Scalar {
    if num_bins == 0 || ellipse.semi_major <= 1e-6 || ellipse.semi_minor <= 1e-6 {
        return 0.0;
    }

    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let mut bins = vec![false; num_bins];

    for (bi, bin) in bins.iter_mut().enumerate() {
        let theta = 2.0 * std::f32::consts::PI * bi as Scalar / num_bins as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let ex = ellipse.semi_major * cos_t;
        let ey = ellipse.semi_minor * sin_t;
        let sx = ellipse.center.x + ex * cos_a - ey * sin_a;
        let sy = ellipse.center.y + ex * sin_a + ey * cos_a;

        let nx_local = ellipse.semi_minor * cos_t;
        let ny_local = ellipse.semi_major * sin_t;
        let n_mag = (nx_local * nx_local + ny_local * ny_local).sqrt();
        if n_mag < 1e-8 {
            continue;
        }
        let nx = (nx_local * cos_a - ny_local * sin_a) / n_mag;
        let ny = (nx_local * sin_a + ny_local * cos_a) / n_mag;

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
        let alignment = ((gx * nx + gy * ny) / mag).abs();
        if alignment > tolerance {
            *bin = true;
        }
    }

    let filled = bins.iter().filter(|&&b| b).count();
    filled as Scalar / num_bins as Scalar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::geometry::Ellipse;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;

    #[test]
    fn full_circle_has_high_coverage() {
        let size = 80;
        let cx = 40.0f32;
        let cy = 40.0f32;
        let radius = 15.0f32;

        // Full bright disk
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

        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let cov = angular_coverage(&grad, PixelCoord::new(cx, cy), radius, 0.5, 64);

        assert!(
            cov > 0.7,
            "full circle should have high coverage, got {cov}"
        );
    }

    #[test]
    fn partial_arc_has_lower_coverage() {
        let size = 80;
        let cx = 40.0f32;
        let cy = 40.0f32;
        let radius = 15.0f32;

        // Half-circle (only right half)
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);
                if r <= radius && angle.abs() < std::f32::consts::FRAC_PI_2 {
                    data[y * size + x] = 255;
                }
            }
        }

        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let cov = angular_coverage(&grad, PixelCoord::new(cx, cy), radius, 0.5, 64);

        assert!(
            cov < 0.7,
            "half circle should have lower coverage than full circle, got {cov}"
        );
    }

    #[test]
    fn empty_image_zero_coverage() {
        let data = vec![0u8; 64 * 64];
        let image = ImageView::from_slice(&data, 64, 64).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let cov = angular_coverage(&grad, PixelCoord::new(32.0, 32.0), 10.0, 0.5, 64);
        assert_eq!(cov, 0.0);
    }

    #[test]
    fn ellipse_boundary_has_high_coverage() {
        let size = 120;
        let cx = 60.0f32;
        let cy = 60.0f32;
        let a = 24.0f32;
        let b = 16.0f32;
        let angle = 0.35f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;
                if (lx / a).powi(2) + (ly / b).powi(2) <= 1.0 {
                    data[y * size + x] = 255;
                }
            }
        }

        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        let ellipse = Ellipse::new(PixelCoord::new(cx, cy), a, b, angle);
        let cov = ellipse_angular_coverage(&grad, &ellipse, 0.5, 64);
        assert!(cov > 0.65, "expected high ellipse coverage, got {cov}");
    }
}
