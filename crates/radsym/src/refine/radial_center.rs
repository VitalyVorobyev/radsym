//! Radial center refinement (Parthasarathy, Nature Methods 2012).
//!
//! Computes the subpixel center of radial symmetry by finding the
//! least-squares intersection of gradient-direction lines. The algorithm
//! is non-iterative and typically achieves subpixel accuracy in a single
//! pass.
//!
//! Two variants are provided:
//!
//! - **Reference**: Roberts-cross gradient on the half-pixel grid, matching
//!   the original paper exactly.
//! - **Production**: Uses the precomputed Sobel gradient field, which is
//!   faster when the gradient is already available.
//!
//! Literature: Parthasarathy, R. "Rapid, accurate particle tracking by
//! calculation of radial symmetry centers." Nature Methods 9, 724–726 (2012).

use crate::core::coords::PixelCoord;
use crate::core::error::Result;
use crate::core::gradient::GradientField;
use crate::core::image_view::ImageView;
use crate::core::scalar::Scalar;

use super::result::{RefinementResult, RefinementStatus};

/// Configuration for radial center refinement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RadialCenterConfig {
    /// Half-width of the patch around the seed (in pixels).
    /// The full patch is `(2*patch_radius+1)²`.
    pub patch_radius: usize,
    /// Minimum gradient magnitude to include a pixel in the fit.
    pub gradient_threshold: Scalar,
}

impl RadialCenterConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> crate::core::error::Result<()> {
        use crate::core::error::RadSymError;
        if self.patch_radius == 0 {
            return Err(RadSymError::InvalidConfig {
                reason: "patch_radius must be > 0",
            });
        }
        Ok(())
    }
}

impl Default for RadialCenterConfig {
    fn default() -> Self {
        Self {
            patch_radius: 12,
            gradient_threshold: 1e-4,
        }
    }
}

/// Refine a seed center using the reference Parthasarathy algorithm.
///
/// Computes Roberts-cross gradients on the half-pixel grid from the raw
/// image patch, then solves the weighted least-squares intersection.
///
/// The returned center is in image coordinates (not patch-local).
pub fn radial_center_refine(
    image: &ImageView<'_, u8>,
    seed: PixelCoord,
    config: &RadialCenterConfig,
) -> Result<RefinementResult<PixelCoord>> {
    config.validate()?;
    let pr = config.patch_radius as i32;
    let cx = seed.x.round() as i32;
    let cy = seed.y.round() as i32;
    let w = image.width() as i32;
    let h = image.height() as i32;

    // Check bounds
    if cx - pr < 0 || cx + pr >= w || cy - pr < 0 || cy + pr >= h {
        return Ok(RefinementResult {
            hypothesis: seed,
            status: RefinementStatus::OutOfBounds,
            residual: 0.0,
            iterations: 0,
        });
    }

    // Roberts-cross gradient on half-pixel grid:
    //   dI/dx(i+0.5, j+0.5) = [I(i+1,j) - I(i,j) + I(i+1,j+1) - I(i,j+1)] / 2
    //   dI/dy(i+0.5, j+0.5) = [I(i,j+1) - I(i,j) + I(i+1,j+1) - I(i,j+1)] / 2
    //
    // We accumulate the weighted LS system: minimize sum_k w_k * (d_k - n_k . c)^2
    // where n_k is the unit gradient direction, c is the center, and d_k = n_k . p_k.

    // f64 accumulation for numerical stability in weighted least-squares:
    // summing many squared gradient magnitudes in f32 can lose precision,
    // causing the 2x2 system determinant to underflow and produce wrong centers.
    let mut swx2 = 0.0f64;
    let mut swy2 = 0.0f64;
    let mut swxy = 0.0f64;
    let mut swxd = 0.0f64;
    let mut swyd = 0.0f64;

    let x_min = cx - pr;
    let y_min = cy - pr;
    let x_max = cx + pr; // exclusive for Roberts cross: need (x+1, y+1)
    let y_max = cy + pr;

    for iy in y_min..y_max {
        for ix in x_min..x_max {
            let p00 = *image.get(ix as usize, iy as usize).unwrap_or(&0) as f64;
            let p10 = *image.get((ix + 1) as usize, iy as usize).unwrap_or(&0) as f64;
            let p01 = *image.get(ix as usize, (iy + 1) as usize).unwrap_or(&0) as f64;
            let p11 = *image
                .get((ix + 1) as usize, (iy + 1) as usize)
                .unwrap_or(&0) as f64;

            let gx = (p10 - p00 + p11 - p01) * 0.5;
            let gy = (p01 - p00 + p11 - p10) * 0.5;

            let mag = (gx * gx + gy * gy).sqrt();
            if mag < config.gradient_threshold as f64 {
                continue;
            }

            // Unit gradient direction
            let nx = gx / mag;
            let ny = gy / mag;

            // Half-pixel position relative to seed
            let px = (ix as f64 + 0.5) - seed.x as f64;
            let py = (iy as f64 + 0.5) - seed.y as f64;

            // Weight: gradient magnitude (Parthasarathy uses |grad|^2 weighting
            // to reduce bias from weak gradients)
            let w = mag * mag;

            // d = n . p  (signed distance from origin to the gradient line)
            let d = nx * px + ny * py;

            swx2 += w * nx * nx;
            swy2 += w * ny * ny;
            swxy += w * nx * ny;
            swxd += w * nx * d;
            swyd += w * ny * d;
        }
    }

    // Solve 2x2 system: [swx2, swxy; swxy, swy2] * [cx, cy] = [swxd, swyd]
    let det = swx2 * swy2 - swxy * swxy;
    if det.abs() < 1e-12 {
        return Ok(RefinementResult {
            hypothesis: seed,
            status: RefinementStatus::Degenerate,
            residual: 0.0,
            iterations: 0,
        });
    }

    let dcx = (swy2 * swxd - swxy * swyd) / det;
    let dcy = (swx2 * swyd - swxy * swxd) / det;

    let refined = PixelCoord::new(seed.x + dcx as Scalar, seed.y + dcy as Scalar);

    // Check if refined center is still within the patch
    let shift = ((dcx * dcx + dcy * dcy).sqrt()) as Scalar;
    let status = if shift > config.patch_radius as Scalar {
        RefinementStatus::OutOfBounds
    } else {
        RefinementStatus::Converged
    };

    Ok(RefinementResult {
        hypothesis: refined,
        status,
        residual: shift,
        iterations: 1,
    })
}

/// Refine a seed center using a precomputed Sobel gradient field.
///
/// This is the production variant — faster when the gradient field is already
/// available (e.g., from proposal generation). Uses gradient magnitude squared
/// as weights, matching the Parthasarathy weighting scheme.
pub fn radial_center_refine_from_gradient(
    gradient: &GradientField,
    seed: PixelCoord,
    config: &RadialCenterConfig,
) -> Result<RefinementResult<PixelCoord>> {
    config.validate()?;
    let pr = config.patch_radius as i32;
    let cx = seed.x.round() as i32;
    let cy = seed.y.round() as i32;
    let w = gradient.width() as i32;
    let h = gradient.height() as i32;

    if cx - pr < 0 || cx + pr >= w || cy - pr < 0 || cy + pr >= h {
        return Ok(RefinementResult {
            hypothesis: seed,
            status: RefinementStatus::OutOfBounds,
            residual: 0.0,
            iterations: 0,
        });
    }

    // f64 accumulation for numerical stability in weighted least-squares:
    // summing many squared gradient magnitudes in f32 can lose precision,
    // causing the 2x2 system determinant to underflow and produce wrong centers.
    let mut swx2 = 0.0f64;
    let mut swy2 = 0.0f64;
    let mut swxy = 0.0f64;
    let mut swxd = 0.0f64;
    let mut swyd = 0.0f64;

    let x_min = (cx - pr).max(0) as usize;
    let y_min = (cy - pr).max(0) as usize;
    let x_max = ((cx + pr) as usize).min(gradient.width());
    let y_max = ((cy + pr) as usize).min(gradient.height());

    for iy in y_min..y_max {
        for ix in x_min..x_max {
            if let Some((gx_f, gy_f)) = gradient.get(ix, iy) {
                let gx = gx_f as f64;
                let gy = gy_f as f64;
                let mag = (gx * gx + gy * gy).sqrt();
                if mag < config.gradient_threshold as f64 {
                    continue;
                }

                let nx = gx / mag;
                let ny = gy / mag;

                // Pixel position relative to seed
                let px = ix as f64 - seed.x as f64;
                let py = iy as f64 - seed.y as f64;

                let w = mag * mag;
                let d = nx * px + ny * py;

                swx2 += w * nx * nx;
                swy2 += w * ny * ny;
                swxy += w * nx * ny;
                swxd += w * nx * d;
                swyd += w * ny * d;
            }
        }
    }

    let det = swx2 * swy2 - swxy * swxy;
    if det.abs() < 1e-12 {
        return Ok(RefinementResult {
            hypothesis: seed,
            status: RefinementStatus::Degenerate,
            residual: 0.0,
            iterations: 0,
        });
    }

    let dcx = (swy2 * swxd - swxy * swyd) / det;
    let dcy = (swx2 * swyd - swxy * swxd) / det;

    let refined = PixelCoord::new(seed.x + dcx as Scalar, seed.y + dcy as Scalar);

    let shift = ((dcx * dcx + dcy * dcy).sqrt()) as Scalar;
    let status = if shift > config.patch_radius as Scalar {
        RefinementStatus::OutOfBounds
    } else {
        RefinementStatus::Converged
    };

    Ok(RefinementResult {
        hypothesis: refined,
        status,
        residual: shift,
        iterations: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;

    /// Create a synthetic Gaussian blob centered at (cx, cy) with given sigma.
    fn make_gaussian_blob(size: usize, cx: f32, cy: f32, sigma: f32) -> Vec<u8> {
        let mut data = vec![0u8; size * size];
        let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let val = (255.0 * (-inv_2s2 * (dx * dx + dy * dy)).exp()) as u8;
                data[y * size + x] = val;
            }
        }
        data
    }

    /// Create a synthetic ring image.
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
    fn reference_on_gaussian_blob() {
        let size = 80;
        let true_cx = 40.3;
        let true_cy = 39.7;
        let data = make_gaussian_blob(size, true_cx, true_cy, 8.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();

        let seed = PixelCoord::new(40.0, 40.0);
        let result = radial_center_refine(&image, seed, &RadialCenterConfig::default()).unwrap();

        assert!(
            result.converged(),
            "should converge, got {:?}",
            result.status
        );
        let err_x = (result.hypothesis.x - true_cx).abs();
        let err_y = (result.hypothesis.y - true_cy).abs();
        assert!(
            err_x < 0.5,
            "x error {err_x} too large (refined={}, true={})",
            result.hypothesis.x,
            true_cx
        );
        assert!(
            err_y < 0.5,
            "y error {err_y} too large (refined={}, true={})",
            result.hypothesis.y,
            true_cy
        );
    }

    #[test]
    fn reference_on_ring() {
        let size = 80;
        let true_cx = 40.0;
        let true_cy = 40.0;
        let data = make_ring(size, true_cx, true_cy, 14.0, 18.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();

        // Use a larger patch to capture the ring edges
        let config = RadialCenterConfig {
            patch_radius: 20,
            ..RadialCenterConfig::default()
        };
        let seed = PixelCoord::new(41.0, 39.0);
        let result = radial_center_refine(&image, seed, &config).unwrap();

        assert!(result.converged());
        let err = ((result.hypothesis.x - true_cx).powi(2)
            + (result.hypothesis.y - true_cy).powi(2))
        .sqrt();
        assert!(
            err < 3.0,
            "refinement error {err} too large, refined=({}, {})",
            result.hypothesis.x,
            result.hypothesis.y
        );
    }

    #[test]
    fn production_on_gaussian_blob() {
        let size = 80;
        let true_cx = 40.3;
        let true_cy = 39.7;
        let data = make_gaussian_blob(size, true_cx, true_cy, 8.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let seed = PixelCoord::new(40.0, 40.0);
        let result =
            radial_center_refine_from_gradient(&grad, seed, &RadialCenterConfig::default())
                .unwrap();

        assert!(result.converged());
        let err = ((result.hypothesis.x - true_cx).powi(2)
            + (result.hypothesis.y - true_cy).powi(2))
        .sqrt();
        // Sobel gradient is coarser than Roberts-cross, so tolerance is larger
        assert!(
            err < 1.5,
            "production error {err} too large, refined=({}, {})",
            result.hypothesis.x,
            result.hypothesis.y
        );
    }

    #[test]
    fn degenerate_on_uniform() {
        let data = vec![128u8; 64 * 64];
        let image = ImageView::from_slice(&data, 64, 64).unwrap();

        let result = radial_center_refine(
            &image,
            PixelCoord::new(32.0, 32.0),
            &RadialCenterConfig::default(),
        )
        .unwrap();
        assert_eq!(result.status, RefinementStatus::Degenerate);
    }

    #[test]
    fn out_of_bounds_near_edge() {
        let size = 40;
        let data = make_gaussian_blob(size, 5.0, 5.0, 4.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();

        let result = radial_center_refine(
            &image,
            PixelCoord::new(5.0, 5.0),
            &RadialCenterConfig::default(),
        )
        .unwrap();
        assert_eq!(result.status, RefinementStatus::OutOfBounds);
    }

    #[test]
    fn default_config_passes_validation() {
        RadialCenterConfig::default().validate().unwrap();
    }

    #[test]
    fn zero_patch_radius_fails_validation() {
        let config = RadialCenterConfig {
            patch_radius: 0,
            ..RadialCenterConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::core::error::RadSymError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn reference_vs_production_consistency() {
        let size = 80;
        let true_cx = 40.0;
        let true_cy = 40.0;
        let data = make_gaussian_blob(size, true_cx, true_cy, 8.0);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let seed = PixelCoord::new(42.0, 38.0);
        let ref_result =
            radial_center_refine(&image, seed, &RadialCenterConfig::default()).unwrap();
        let prod_result =
            radial_center_refine_from_gradient(&grad, seed, &RadialCenterConfig::default())
                .unwrap();

        assert!(ref_result.converged());
        assert!(prod_result.converged());

        // Both should refine toward the true center (within ~1px of each other)
        let diff = ((ref_result.hypothesis.x - prod_result.hypothesis.x).powi(2)
            + (ref_result.hypothesis.y - prod_result.hypothesis.y).powi(2))
        .sqrt();
        assert!(
            diff < 1.5,
            "reference and production differ by {diff} pixels"
        );
    }
}
