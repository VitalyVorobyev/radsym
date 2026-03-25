//! Gradient computation for image data.
//!
//! Provides Sobel-based gradient computation and a [`GradientField`] type that
//! stores per-pixel gradient vectors.

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
}

/// Compute the gradient field of a grayscale `u8` image using a 3x3 Sobel operator.
///
/// The output has the same dimensions as the input. Border pixels are set to zero.
pub fn sobel_gradient(image: &ImageView<'_, u8>) -> Result<GradientField> {
    let w = image.width();
    let h = image.height();
    let mut gx = OwnedImage::<Scalar>::zeros(w, h)?;
    let mut gy = OwnedImage::<Scalar>::zeros(w, h)?;

    let gx_data = gx.data_mut();
    let gy_data = gy.data_mut();

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            // Sobel 3x3 kernel
            let p00 = image.get(x - 1, y - 1).copied().unwrap_or(0) as Scalar;
            let p10 = image.get(x, y - 1).copied().unwrap_or(0) as Scalar;
            let p20 = image.get(x + 1, y - 1).copied().unwrap_or(0) as Scalar;
            let p01 = image.get(x - 1, y).copied().unwrap_or(0) as Scalar;
            let p21 = image.get(x + 1, y).copied().unwrap_or(0) as Scalar;
            let p02 = image.get(x - 1, y + 1).copied().unwrap_or(0) as Scalar;
            let p12 = image.get(x, y + 1).copied().unwrap_or(0) as Scalar;
            let p22 = image.get(x + 1, y + 1).copied().unwrap_or(0) as Scalar;

            // Sobel X: [-1 0 1; -2 0 2; -1 0 1] / 8
            let dx = (-p00 + p20 - 2.0 * p01 + 2.0 * p21 - p02 + p22) / 8.0;
            // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1] / 8
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
    let mut gx = OwnedImage::<Scalar>::zeros(w, h)?;
    let mut gy = OwnedImage::<Scalar>::zeros(w, h)?;

    let gx_data = gx.data_mut();
    let gy_data = gy.data_mut();

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p00 = *image.get(x - 1, y - 1).unwrap_or(&0.0);
            let p10 = *image.get(x, y - 1).unwrap_or(&0.0);
            let p20 = *image.get(x + 1, y - 1).unwrap_or(&0.0);
            let p01 = *image.get(x - 1, y).unwrap_or(&0.0);
            let p21 = *image.get(x + 1, y).unwrap_or(&0.0);
            let p02 = *image.get(x - 1, y + 1).unwrap_or(&0.0);
            let p12 = *image.get(x, y + 1).unwrap_or(&0.0);
            let p22 = *image.get(x + 1, y + 1).unwrap_or(&0.0);

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
}
