//! Gradient computation for image data.
//!
//! Provides Sobel and Scharr gradient computation and a [`GradientField`] type
//! that stores per-pixel gradient vectors. Use [`compute_gradient`] to select
//! the operator at runtime, or call [`sobel_gradient`] / [`scharr_gradient`]
//! directly.

use super::error::Result;
use super::image_view::{ImageView, OwnedImage};
use super::scalar::Scalar;

/// A pixel type that can be read as a scalar intensity for gradient computation.
///
/// Implemented for `u8`, `u16`, and `f32` so the detection pipeline can ingest
/// 8-bit, 10/12/16-bit, and floating-point imagery without a lossy
/// pre-conversion. The gradient hot loops cast each pixel through
/// [`SourcePixel::to_scalar`].
///
/// **Threshold note:** gradient magnitudes scale with the input's value range,
/// so *absolute* thresholds such as [`FrstConfig::gradient_threshold`] and
/// [`RadialCenterConfig::gradient_threshold`] are bit-depth dependent — a
/// threshold tuned for `u8` is ~256× too small for raw `u16`. Scale such
/// thresholds to the input range, or normalize against
/// [`GradientField::max_magnitude`].
///
/// [`FrstConfig::gradient_threshold`]: crate::FrstConfig
/// [`RadialCenterConfig::gradient_threshold`]: crate::RadialCenterConfig
pub trait SourcePixel: Copy {
    /// Convert this pixel to a scalar intensity.
    fn to_scalar(self) -> Scalar;
}

impl SourcePixel for u8 {
    #[inline]
    fn to_scalar(self) -> Scalar {
        self as Scalar
    }
}

impl SourcePixel for u16 {
    #[inline]
    fn to_scalar(self) -> Scalar {
        self as Scalar
    }
}

impl SourcePixel for f32 {
    #[inline]
    fn to_scalar(self) -> Scalar {
        self
    }
}

/// Per-pixel gradient field storing separate x and y gradient components.
pub struct GradientField {
    /// Horizontal gradient (dI/dx). Positive = intensity increases rightward.
    pub(crate) gx: OwnedImage<Scalar>,
    /// Vertical gradient (dI/dy). Positive = intensity increases downward.
    pub(crate) gy: OwnedImage<Scalar>,
}

impl GradientField {
    /// Image width.
    #[inline]
    pub fn width(&self) -> usize {
        self.gx.width()
    }

    /// Image height.
    #[inline]
    pub fn height(&self) -> usize {
        self.gx.height()
    }

    /// Borrowed view of the horizontal gradient component.
    #[inline]
    pub fn gx(&self) -> ImageView<'_, Scalar> {
        self.gx.view()
    }

    /// Borrowed view of the vertical gradient component.
    #[inline]
    pub fn gy(&self) -> ImageView<'_, Scalar> {
        self.gy.view()
    }

    /// Gradient vector at pixel `(x, y)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<(Scalar, Scalar)> {
        let gx = self.gx.get(x, y)?;
        let gy = self.gy.get(x, y)?;
        Some((gx, gy))
    }

    /// Gradient magnitude at pixel `(x, y)`.
    #[inline]
    pub fn magnitude(&self, x: usize, y: usize) -> Option<Scalar> {
        let (gx, gy) = self.get(x, y)?;
        Some((gx * gx + gy * gy).sqrt())
    }

    /// Maximum gradient magnitude across the entire field.
    pub fn max_magnitude(&self) -> Scalar {
        let gx = self.gx.data();
        let gy = self.gy.data();
        gx.iter()
            .zip(gy.iter())
            .map(|(&x, &y)| x * x + y * y)
            .fold(0.0f32, Scalar::max)
            .sqrt()
    }
}

/// Compute the gradient field of a grayscale `u8` image using a 3x3 Sobel operator.
///
/// The output has the same dimensions as the input. Border pixels are set to zero.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, sobel_gradient};
///
/// let size = 16usize;
/// let mut data = vec![0u8; size * size];
/// // Create a vertical step edge at x = 8
/// for y in 0..size {
///     for x in 8..size {
///         data[y * size + x] = 255;
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = sobel_gradient(&image).unwrap();
/// assert_eq!(grad.width(), size);
/// assert_eq!(grad.height(), size);
/// // Strong horizontal gradient at the step edge
/// let (gx, _) = grad.get(8, size / 2).unwrap();
/// assert!(gx > 20.0, "expected strong gx at step edge, got {gx}");
/// ```
pub fn sobel_gradient<P: SourcePixel>(image: &ImageView<'_, P>) -> Result<GradientField> {
    let w = image.width();
    let h = image.height();
    let stride = image.stride();
    let src = image.as_slice();
    let mut gx = OwnedImage::<Scalar>::zeros(w, h)?;
    let mut gy = OwnedImage::<Scalar>::zeros(w, h)?;

    let gx_data = gx.data_mut();
    let gy_data = gy.data_mut();

    for y in 1..h - 1 {
        let row_prev = (y - 1) * stride;
        let row_curr = y * stride;
        let row_next = (y + 1) * stride;
        for x in 1..w - 1 {
            // Direct slice access — loop bounds guarantee all neighbors are in-bounds.
            let p00 = src[row_prev + x - 1].to_scalar();
            let p10 = src[row_prev + x].to_scalar();
            let p20 = src[row_prev + x + 1].to_scalar();
            let p01 = src[row_curr + x - 1].to_scalar();
            let p21 = src[row_curr + x + 1].to_scalar();
            let p02 = src[row_next + x - 1].to_scalar();
            let p12 = src[row_next + x].to_scalar();
            let p22 = src[row_next + x + 1].to_scalar();

            let dx = (-p00 + p20 - 2.0 * p01 + 2.0 * p21 - p02 + p22) / 8.0;
            let dy = (-p00 - 2.0 * p10 - p20 + p02 + 2.0 * p12 + p22) / 8.0;

            let idx = y * w + x;
            gx_data[idx] = dx;
            gy_data[idx] = dy;
        }
    }

    Ok(GradientField { gx, gy })
}

/// Compute the gradient field of a float image using a 3x3 Sobel operator.
///
/// Thin wrapper over the generic [`sobel_gradient`]; kept for explicit `f32`
/// call sites.
pub fn sobel_gradient_f32(image: &ImageView<'_, f32>) -> Result<GradientField> {
    sobel_gradient(image)
}

/// Compute gradient magnitude image from a gradient field.
pub fn gradient_magnitude(field: &GradientField) -> Result<OwnedImage<Scalar>> {
    let w = field.width();
    let h = field.height();
    let mut mag = OwnedImage::<Scalar>::zeros(w, h)?;
    let mag_data = mag.data_mut();
    let gx_data = field.gx.data();
    let gy_data = field.gy.data();

    for i in 0..w * h {
        let gx = gx_data[i];
        let gy = gy_data[i];
        mag_data[i] = (gx * gx + gy * gy).sqrt();
    }

    Ok(mag)
}

/// Thin a gradient field to single-pixel-wide ridges via non-maximum
/// suppression along the gradient direction (Canny-style edge thinning).
///
/// Each interior pixel's gradient orientation is quantized to one of four
/// directions — 0°, 45°, 90°, 135° — using ratio tests on `(gx, gy)` (no
/// `atan2`, no trigonometry). Its squared magnitude is compared against the two
/// 8-neighbours straddling that direction; pixels that are not a local maximum
/// along the gradient are set to `(0, 0)`. Multi-pixel-wide gradient bands
/// collapse to a single ridge pixel, which sharpens center-voting responses and
/// — with a positive gradient threshold — reduces the number of voting pixels.
///
/// Output dimensions match the input; the 1-pixel border stays zero, matching
/// [`sobel_gradient`] / [`scharr_gradient`]. Images smaller than 3×3 in either
/// axis have no interior and yield an all-zero field.
///
/// This is distinct from [`non_maximum_suppression`](crate::NmsConfig), which
/// suppresses peaks of a scalar *response* map in a square window; this thins a
/// *gradient* field along the gradient normal.
///
/// # Threshold note
///
/// Suppressed pixels are set to exactly zero, and the RSD/FRST voting gate skips
/// a pixel only when `mag² < gradient_threshold²`. At a gradient threshold of
/// `0.0`, a zeroed pixel is therefore *not* skipped (`0.0 < 0.0` is false):
/// thinning still sharpens the response but does not reduce voting iterations.
/// Use a small positive gradient threshold to reclaim the voting-cost reduction.
///
/// # References
///
/// Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE TPAMI
/// 8(6), 679–698 — the non-maximum-suppression stage.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, scharr_gradient, thin_gradient};
///
/// let size = 32usize;
/// let mut data = vec![0u8; size * size];
/// for y in 0..size {
///     for x in 16..size {
///         data[y * size + x] = 255; // vertical step edge at x = 16
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = scharr_gradient(&image).unwrap();
/// let thin = thin_gradient(&grad).unwrap();
/// assert_eq!((thin.width(), thin.height()), (size, size));
///
/// // Thinning keeps no more active pixels than the raw field.
/// let active = |g: &radsym::GradientField| {
///     (0..g.height())
///         .flat_map(|y| (0..g.width()).map(move |x| (x, y)))
///         .filter(|&(x, y)| g.magnitude(x, y).unwrap() > 0.0)
///         .count()
/// };
/// assert!(active(&thin) <= active(&grad));
/// ```
pub fn thin_gradient(field: &GradientField) -> Result<GradientField> {
    let w = field.width();
    let h = field.height();

    let gx_in = field.gx.data();
    let gy_in = field.gy.data();

    // Read-only squared-magnitude scratch: never mutated during the pass, so the
    // result is independent of traversal order (fully deterministic).
    let mut mag_sq = vec![0.0f32; w * h];
    for i in 0..w * h {
        mag_sq[i] = gx_in[i] * gx_in[i] + gy_in[i] * gy_in[i];
    }

    let mut out_gx = OwnedImage::<Scalar>::zeros(w, h)?;
    let mut out_gy = OwnedImage::<Scalar>::zeros(w, h)?;
    let out_gx_data = out_gx.data_mut();
    let out_gy_data = out_gy.data_mut();

    // tan(22.5°) = √2 − 1, tan(67.5°) = √2 + 1.
    const LO: Scalar = std::f32::consts::SQRT_2 - 1.0;
    const HI: Scalar = std::f32::consts::SQRT_2 + 1.0;

    // Interior only; borders stay zero. saturating_sub keeps tiny images safe.
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let m = mag_sq[idx];
            if m == 0.0 {
                continue; // no direction; nothing to keep
            }

            let gx = gx_in[idx];
            let gy = gy_in[idx];
            let ax = gx.abs();
            let ay = gy.abs();

            // Step toward the +gradient neighbour. sign picks, for the diagonal
            // sector, the same-sign "\" vs opposite-sign "/" diagonal.
            let sx: isize = if gx >= 0.0 { 1 } else { -1 };
            let sy: isize = if gy >= 0.0 { 1 } else { -1 };
            let (fx, fy): (isize, isize) = if ay < LO * ax {
                (sx, 0) // horizontal gradient → compare left/right
            } else if ay > HI * ax {
                (0, sy) // vertical gradient → compare up/down
            } else {
                (sx, sy) // diagonal gradient
            };

            let fwd = mag_sq[(y as isize + fy) as usize * w + (x as isize + fx) as usize];
            let bwd = mag_sq[(y as isize - fy) as usize * w + (x as isize - fx) as usize];

            // Asymmetric tie-break (`>=` forward, `>` backward) so an equal-value
            // band along the gradient reduces to exactly one pixel.
            if m >= fwd && m > bwd {
                out_gx_data[idx] = gx;
                out_gy_data[idx] = gy;
            }
        }
    }

    Ok(GradientField {
        gx: out_gx,
        gy: out_gy,
    })
}

/// Choice of 3x3 gradient operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum GradientOperator {
    /// Sobel 3x3 (weights 1-2-1). Standard, widely used.
    #[default]
    Sobel,
    /// Scharr 3x3 (weights 3-10-3). Better rotational symmetry than Sobel.
    Scharr,
}

/// Compute the gradient field of a grayscale `u8` image using a 3x3 Scharr operator.
///
/// The Scharr operator provides better rotational isotropy than Sobel,
/// which can improve detection quality for circular structures.
/// Border pixels are set to zero.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, scharr_gradient};
///
/// let size = 16usize;
/// let mut data = vec![0u8; size * size];
/// // Create a vertical step edge at x = 8
/// for y in 0..size {
///     for x in 8..size {
///         data[y * size + x] = 255;
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = scharr_gradient(&image).unwrap();
/// assert_eq!(grad.width(), size);
/// assert_eq!(grad.height(), size);
/// // Strong horizontal gradient at the step edge
/// let (gx, _) = grad.get(8, size / 2).unwrap();
/// assert!(gx > 20.0, "expected strong gx at step edge, got {gx}");
/// ```
pub fn scharr_gradient<P: SourcePixel>(image: &ImageView<'_, P>) -> Result<GradientField> {
    let w = image.width();
    let h = image.height();
    let stride = image.stride();
    let src = image.as_slice();
    let mut gx = OwnedImage::<Scalar>::zeros(w, h)?;
    let mut gy = OwnedImage::<Scalar>::zeros(w, h)?;

    let gx_data = gx.data_mut();
    let gy_data = gy.data_mut();

    for y in 1..h - 1 {
        let row_prev = (y - 1) * stride;
        let row_curr = y * stride;
        let row_next = (y + 1) * stride;
        for x in 1..w - 1 {
            let p00 = src[row_prev + x - 1].to_scalar();
            let p10 = src[row_prev + x].to_scalar();
            let p20 = src[row_prev + x + 1].to_scalar();
            let p01 = src[row_curr + x - 1].to_scalar();
            let p21 = src[row_curr + x + 1].to_scalar();
            let p02 = src[row_next + x - 1].to_scalar();
            let p12 = src[row_next + x].to_scalar();
            let p22 = src[row_next + x + 1].to_scalar();

            let dx =
                (-3.0 * p00 + 3.0 * p20 - 10.0 * p01 + 10.0 * p21 - 3.0 * p02 + 3.0 * p22) / 32.0;
            let dy =
                (-3.0 * p00 - 10.0 * p10 - 3.0 * p20 + 3.0 * p02 + 10.0 * p12 + 3.0 * p22) / 32.0;

            let idx = y * w + x;
            gx_data[idx] = dx;
            gy_data[idx] = dy;
        }
    }

    Ok(GradientField { gx, gy })
}

/// Compute the gradient field of a float image using a 3x3 Scharr operator.
///
/// Thin wrapper over the generic [`scharr_gradient`]; kept for explicit `f32`
/// call sites.
pub fn scharr_gradient_f32(image: &ImageView<'_, f32>) -> Result<GradientField> {
    scharr_gradient(image)
}

/// Compute the gradient field using the specified operator.
///
/// Dispatches to [`sobel_gradient`] or [`scharr_gradient`] based on `operator`.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, compute_gradient, GradientOperator};
///
/// let size = 16usize;
/// let mut data = vec![0u8; size * size];
/// // Create a vertical step edge at x = 8
/// for y in 0..size {
///     for x in 8..size {
///         data[y * size + x] = 255;
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = compute_gradient(&image, GradientOperator::Scharr).unwrap();
/// assert_eq!(grad.width(), size);
/// assert_eq!(grad.height(), size);
/// // Strong horizontal gradient at the step edge
/// let (gx, _) = grad.get(8, size / 2).unwrap();
/// assert!(gx > 20.0, "expected strong gx at step edge, got {gx}");
/// ```
pub fn compute_gradient<P: SourcePixel>(
    image: &ImageView<'_, P>,
    operator: GradientOperator,
) -> Result<GradientField> {
    match operator {
        GradientOperator::Sobel => sobel_gradient(image),
        GradientOperator::Scharr => scharr_gradient(image),
    }
}

/// Compute the gradient field of a float image using the specified operator.
///
/// Thin wrapper over the generic [`compute_gradient`]; kept for explicit `f32`
/// call sites.
pub fn compute_gradient_f32(
    image: &ImageView<'_, f32>,
    operator: GradientOperator,
) -> Result<GradientField> {
    compute_gradient(image, operator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_of_horizontal_step() {
        // 5x3 image with vertical edge at column 2
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            0, 0, 255, 255, 255,
            0, 0, 255, 255, 255,
            0, 0, 255, 255, 255,
        ];
        let image = ImageView::from_slice(&data, 5, 3).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        // At the step edge (x=2, y=1), gx should be strongly positive
        let (gx, gy) = grad.get(2, 1).unwrap();
        assert!(
            gx > 30.0,
            "expected strong horizontal gradient, got gx={gx}"
        );
        assert!(
            gy.abs() < 1e-6,
            "expected zero vertical gradient, got gy={gy}"
        );
    }

    #[test]
    fn gradient_of_vertical_step() {
        // 3x5 image with horizontal edge at row 2
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            0, 0, 0,
            0, 0, 0,
            255, 255, 255,
            255, 255, 255,
            255, 255, 255,
        ];
        let image = ImageView::from_slice(&data, 3, 5).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let (gx, gy) = grad.get(1, 2).unwrap();
        assert!(
            gx.abs() < 1e-6,
            "expected zero horizontal gradient, got gx={gx}"
        );
        assert!(gy > 30.0, "expected strong vertical gradient, got gy={gy}");
    }

    #[test]
    fn gradient_magnitude_computation() {
        let data: Vec<u8> = vec![0; 9];
        let image = ImageView::from_slice(&data, 3, 3).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        let mag = gradient_magnitude(&grad).unwrap();
        // Uniform image: all gradients should be zero
        assert!(mag.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gradient_field_dimensions() {
        let data: Vec<u8> = vec![128; 20];
        let image = ImageView::from_slice(&data, 5, 4).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        assert_eq!(grad.width(), 5);
        assert_eq!(grad.height(), 4);
    }

    /// Build a gradient field from raw component vectors (test-only).
    fn field_from(w: usize, h: usize, gx: Vec<Scalar>, gy: Vec<Scalar>) -> GradientField {
        GradientField {
            gx: OwnedImage::from_vec(gx, w, h).unwrap(),
            gy: OwnedImage::from_vec(gy, w, h).unwrap(),
        }
    }

    #[test]
    fn thin_reduces_horizontal_band_to_single_column() {
        // 5x5 field, purely horizontal gradient (gy=0), triangular ridge across
        // columns: magnitude 0,1,2,1,0 in every row. NMS along the gradient
        // (left/right) must keep only the peak column (x=2) for interior rows.
        let w = 5;
        let h = 5;
        let row = [0.0, 1.0, 2.0, 1.0, 0.0];
        let mut gx = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                gx[y * w + x] = row[x];
            }
        }
        let field = field_from(w, h, gx, vec![0.0; w * h]);
        let thin = thin_gradient(&field).unwrap();

        for y in 0..h {
            for x in 0..w {
                let kept = thin.magnitude(x, y).unwrap() > 0.0;
                // Only interior rows (1..=3) at the peak column x=2 survive.
                let expect = (1..=3).contains(&y) && x == 2;
                assert_eq!(
                    kept, expect,
                    "pixel ({x},{y}) kept={kept}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn thin_reduces_vertical_band_to_single_row() {
        // Transpose of the horizontal case: gx=0, gy ridge down the rows.
        let w = 5;
        let h = 5;
        let col = [0.0, 1.0, 2.0, 1.0, 0.0];
        let mut gy = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                gy[y * w + x] = col[y];
            }
        }
        let field = field_from(w, h, vec![0.0; w * h], gy);
        let thin = thin_gradient(&field).unwrap();

        for y in 0..h {
            for x in 0..w {
                let kept = thin.magnitude(x, y).unwrap() > 0.0;
                let expect = y == 2 && (1..=3).contains(&x);
                assert_eq!(
                    kept, expect,
                    "pixel ({x},{y}) kept={kept}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn thin_preserves_surviving_gradient_values() {
        // The kept peak must retain its exact (gx, gy), not a modified value.
        let w = 5;
        let h = 3;
        let row = [0.0, 1.0, 2.0, 1.0, 0.0];
        let mut gx = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                gx[y * w + x] = row[x];
            }
        }
        let field = field_from(w, h, gx, vec![0.0; w * h]);
        let thin = thin_gradient(&field).unwrap();
        let (gx_k, gy_k) = thin.get(2, 1).unwrap();
        assert_eq!((gx_k, gy_k), (2.0, 0.0));
    }

    #[test]
    fn thin_borders_stay_zero() {
        // A sobel gradient of a disk, then confirm the 1px border is all zero.
        let size = 32;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - 16.0;
                let dy = y as f32 - 16.0;
                if dx * dx + dy * dy <= 64.0 {
                    data[y * size + x] = 255;
                }
            }
        }
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let thin = thin_gradient(&sobel_gradient(&image).unwrap()).unwrap();
        for x in 0..size {
            assert_eq!(thin.magnitude(x, 0).unwrap(), 0.0);
            assert_eq!(thin.magnitude(x, size - 1).unwrap(), 0.0);
        }
        for y in 0..size {
            assert_eq!(thin.magnitude(0, y).unwrap(), 0.0);
            assert_eq!(thin.magnitude(size - 1, y).unwrap(), 0.0);
        }
    }

    #[test]
    fn thin_is_deterministic_and_idempotent() {
        // Deterministic pseudo-pattern field with mixed directions.
        let w = 16;
        let h = 12;
        let mut gx = vec![0.0f32; w * h];
        let mut gy = vec![0.0f32; w * h];
        for i in 0..w * h {
            gx[i] = ((i * 7) % 11) as f32 - 5.0;
            gy[i] = ((i * 13) % 9) as f32 - 4.0;
        }
        let field = field_from(w, h, gx, gy);

        let a = thin_gradient(&field).unwrap();
        let b = thin_gradient(&field).unwrap();
        assert_eq!(a.gx.data(), b.gx.data(), "gx not deterministic");
        assert_eq!(a.gy.data(), b.gy.data(), "gy not deterministic");

        // Thinning an already-thinned field is stable.
        let c = thin_gradient(&a).unwrap();
        assert_eq!(a.gx.data(), c.gx.data(), "not idempotent (gx)");
        assert_eq!(a.gy.data(), c.gy.data(), "not idempotent (gy)");
    }

    #[test]
    fn thin_tiny_image_no_panic() {
        for (w, h) in [(2usize, 2usize), (1, 5), (5, 1), (3, 1), (1, 1)] {
            let field = field_from(w, h, vec![1.0; w * h], vec![1.0; w * h]);
            let thin = thin_gradient(&field).unwrap();
            // No interior → everything suppressed.
            assert!(thin.gx.data().iter().all(|&v| v == 0.0));
            assert!(thin.gy.data().iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn scharr_gradient_of_horizontal_step() {
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            0, 0, 255, 255, 255,
            0, 0, 255, 255, 255,
            0, 0, 255, 255, 255,
        ];
        let image = ImageView::from_slice(&data, 5, 3).unwrap();
        let grad = scharr_gradient(&image).unwrap();

        let (gx, gy) = grad.get(2, 1).unwrap();
        assert!(
            gx > 30.0,
            "expected strong horizontal gradient, got gx={gx}"
        );
        assert!(
            gy.abs() < 1e-6,
            "expected zero vertical gradient, got gy={gy}"
        );
    }

    #[test]
    fn scharr_gradient_zeros_on_uniform() {
        let data: Vec<u8> = vec![128; 25];
        let image = ImageView::from_slice(&data, 5, 5).unwrap();
        let grad = scharr_gradient(&image).unwrap();
        assert!(grad.gx().as_slice().iter().all(|&v| v == 0.0));
        assert!(grad.gy().as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scharr_dimensions_match() {
        let data: Vec<u8> = vec![128; 20];
        let image = ImageView::from_slice(&data, 5, 4).unwrap();
        let grad = scharr_gradient(&image).unwrap();
        assert_eq!(grad.width(), 5);
        assert_eq!(grad.height(), 4);
    }

    #[test]
    fn scharr_gradient_f32_matches_u8() {
        // 5x3 step-edge: left half 0, right half at max
        #[rustfmt::skip]
        let data_u8: Vec<u8> = vec![
            0, 0, 255, 255, 255,
            0, 0, 255, 255, 255,
            0, 0, 255, 255, 255,
        ];
        let data_f32: Vec<f32> = data_u8.iter().map(|&v| v as f32).collect();

        let img_u8 = ImageView::from_slice(&data_u8, 5, 3).unwrap();
        let img_f32 = ImageView::from_slice(&data_f32, 5, 3).unwrap();

        let grad_u8 = scharr_gradient(&img_u8).unwrap();
        let grad_f32 = scharr_gradient_f32(&img_f32).unwrap();

        let (gx_u8, gy_u8) = grad_u8.get(2, 1).unwrap();
        let (gx_f32, gy_f32) = grad_f32.get(2, 1).unwrap();

        assert!(
            (gx_u8 - gx_f32).abs() < 1e-4,
            "gx mismatch: u8={gx_u8} f32={gx_f32}"
        );
        assert!(
            (gy_u8 - gy_f32).abs() < 1e-4,
            "gy mismatch: u8={gy_u8} f32={gy_f32}"
        );
    }

    #[test]
    fn compute_gradient_f32_dispatches() {
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            0.0, 0.0, 255.0, 255.0, 255.0,
            0.0, 0.0, 255.0, 255.0, 255.0,
            0.0, 0.0, 255.0, 255.0, 255.0,
        ];
        let image = ImageView::from_slice(&data, 5, 3).unwrap();

        let sobel = compute_gradient_f32(&image, GradientOperator::Sobel).unwrap();
        let scharr = compute_gradient_f32(&image, GradientOperator::Scharr).unwrap();

        let (sobel_gx, _) = sobel.get(2, 1).unwrap();
        let (scharr_gx, _) = scharr.get(2, 1).unwrap();

        assert!(
            sobel_gx.abs() > 0.1,
            "Sobel f32 gx should be nonzero: {sobel_gx}"
        );
        assert!(
            scharr_gx.abs() > 0.1,
            "Scharr f32 gx should be nonzero: {scharr_gx}"
        );
    }

    #[test]
    fn compute_gradient_dispatches_correctly() {
        // Asymmetric pattern: corner pixel differs, making Sobel and Scharr
        // produce different gx values due to different corner weights
        // (Sobel: 1, Scharr: 3).
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            100,   0,   0,
              0,   0,   0,
              0,   0,   0,
        ];
        let image = ImageView::from_slice(&data, 3, 3).unwrap();
        let sobel = compute_gradient(&image, GradientOperator::Sobel).unwrap();
        let scharr = compute_gradient(&image, GradientOperator::Scharr).unwrap();

        let (sx, _) = sobel.get(1, 1).unwrap();
        let (cx, _) = scharr.get(1, 1).unwrap();
        // Both should be nonzero
        assert!(sx.abs() > 0.1, "Sobel gx should be nonzero: {sx}");
        assert!(cx.abs() > 0.1, "Scharr gx should be nonzero: {cx}");
        // Different weights on corner pixel means different values
        // Sobel: -100 * 1 / 8 = -12.5; Scharr: -100 * 3 / 32 = -9.375
        assert!(
            (sx - cx).abs() > 0.1,
            "Sobel gx ({sx}) and Scharr gx ({cx}) should differ"
        );
    }
}
