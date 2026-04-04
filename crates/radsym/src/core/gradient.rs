//! Gradient computation for image data.
//!
//! Provides Sobel and Scharr gradient computation and a [`GradientField`] type
//! that stores per-pixel gradient vectors. Use [`compute_gradient`] to select
//! the operator at runtime, or call [`sobel_gradient`] / [`scharr_gradient`]
//! directly.

use super::error::Result;
use super::image_view::{ImageView, OwnedImage};
use super::scalar::Scalar;

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
pub fn sobel_gradient(image: &ImageView<'_, u8>) -> Result<GradientField> {
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
            let p00 = src[row_prev + x - 1] as Scalar;
            let p10 = src[row_prev + x] as Scalar;
            let p20 = src[row_prev + x + 1] as Scalar;
            let p01 = src[row_curr + x - 1] as Scalar;
            let p21 = src[row_curr + x + 1] as Scalar;
            let p02 = src[row_next + x - 1] as Scalar;
            let p12 = src[row_next + x] as Scalar;
            let p22 = src[row_next + x + 1] as Scalar;

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
pub fn sobel_gradient_f32(image: &ImageView<'_, f32>) -> Result<GradientField> {
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
            let p00 = src[row_prev + x - 1];
            let p10 = src[row_prev + x];
            let p20 = src[row_prev + x + 1];
            let p01 = src[row_curr + x - 1];
            let p21 = src[row_curr + x + 1];
            let p02 = src[row_next + x - 1];
            let p12 = src[row_next + x];
            let p22 = src[row_next + x + 1];

            let dx = (-p00 + p20 - 2.0 * p01 + 2.0 * p21 - p02 + p22) / 8.0;
            let dy = (-p00 - 2.0 * p10 - p20 + p02 + 2.0 * p12 + p22) / 8.0;

            let idx = y * w + x;
            gx_data[idx] = dx;
            gy_data[idx] = dy;
        }
    }

    Ok(GradientField { gx, gy })
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
pub fn scharr_gradient(image: &ImageView<'_, u8>) -> Result<GradientField> {
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
            let p00 = src[row_prev + x - 1] as Scalar;
            let p10 = src[row_prev + x] as Scalar;
            let p20 = src[row_prev + x + 1] as Scalar;
            let p01 = src[row_curr + x - 1] as Scalar;
            let p21 = src[row_curr + x + 1] as Scalar;
            let p02 = src[row_next + x - 1] as Scalar;
            let p12 = src[row_next + x] as Scalar;
            let p22 = src[row_next + x + 1] as Scalar;

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
pub fn scharr_gradient_f32(image: &ImageView<'_, f32>) -> Result<GradientField> {
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
            let p00 = src[row_prev + x - 1];
            let p10 = src[row_prev + x];
            let p20 = src[row_prev + x + 1];
            let p01 = src[row_curr + x - 1];
            let p21 = src[row_curr + x + 1];
            let p02 = src[row_next + x - 1];
            let p12 = src[row_next + x];
            let p22 = src[row_next + x + 1];

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
pub fn compute_gradient(
    image: &ImageView<'_, u8>,
    operator: GradientOperator,
) -> Result<GradientField> {
    match operator {
        GradientOperator::Sobel => sobel_gradient(image),
        GradientOperator::Scharr => scharr_gradient(image),
    }
}

/// Compute the gradient field of a float image using the specified operator.
pub fn compute_gradient_f32(
    image: &ImageView<'_, f32>,
    operator: GradientOperator,
) -> Result<GradientField> {
    match operator {
        GradientOperator::Sobel => sobel_gradient_f32(image),
        GradientOperator::Scharr => scharr_gradient_f32(image),
    }
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
