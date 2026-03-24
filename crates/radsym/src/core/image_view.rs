//! Image view abstractions.
//!
//! [`ImageView`] is a borrowed 2D view into a flat buffer with explicit stride.
//! The stride counts elements between row starts, so a stride larger than width
//! represents padded rows. ROI slices are zero-copy views into the same backing
//! slice.
//!
//! [`OwnedImage`] stores a contiguous image buffer and can produce an
//! [`ImageView`] into it.

use super::error::{RadSymError, Result};
use super::scalar::Scalar;

// ---------------------------------------------------------------------------
// ImageView
// ---------------------------------------------------------------------------

/// Borrowed 2D image view with explicit stride.
///
/// Generic over the element type `T`. For grayscale `u8` images, use
/// `ImageView<'_, u8>`; for float accumulators and response maps, use
/// `ImageView<'_, f32>`.
#[derive(Copy, Clone)]
pub struct ImageView<'a, T> {
    data: &'a [T],
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a, T> ImageView<'a, T> {
    /// Creates a contiguous view with `stride == width`.
    pub fn from_slice(data: &'a [T], width: usize, height: usize) -> Result<Self> {
        Self::new(data, width, height, width)
    }

    /// Creates a view with an explicit stride.
    pub fn new(data: &'a [T], width: usize, height: usize, stride: usize) -> Result<Self> {
        let needed = required_len(width, height, stride)?;
        if data.len() < needed {
            return Err(RadSymError::BufferTooSmall {
                needed,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Stride in elements between row starts.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The full backing slice (including any padding).
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// Element at `(x, y)`, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<&'a T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get(y * self.stride + x)
    }

    /// Contiguous slice for row `y` with length `width`.
    #[inline]
    pub fn row(&self, y: usize) -> Option<&'a [T]> {
        if y >= self.height {
            return None;
        }
        let start = y * self.stride;
        self.data.get(start..start + self.width)
    }

    /// Zero-copy ROI view into the same backing buffer.
    pub fn roi(&self, x: usize, y: usize, width: usize, height: usize) -> Result<ImageView<'a, T>> {
        if width == 0 || height == 0 {
            return Err(RadSymError::InvalidDimensions { width, height });
        }
        let end_x = x.checked_add(width).ok_or(RadSymError::InvalidDimensions {
            width: self.width,
            height: self.height,
        })?;
        let end_y = y
            .checked_add(height)
            .ok_or(RadSymError::InvalidDimensions {
                width: self.width,
                height: self.height,
            })?;
        if end_x > self.width || end_y > self.height {
            return Err(RadSymError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        let start = y * self.stride + x;
        let data = &self.data[start..];
        ImageView::new(data, width, height, self.stride)
    }
}

// ---------------------------------------------------------------------------
// Bilinear sampling (f32 specialization)
// ---------------------------------------------------------------------------

impl ImageView<'_, f32> {
    /// Bilinear interpolation at subpixel position `(x, y)`.
    ///
    /// Returns `None` if `(x, y)` is outside the image bounds (with a 1-pixel
    /// margin for the interpolation footprint).
    #[inline]
    pub fn sample(&self, x: Scalar, y: Scalar) -> Option<Scalar> {
        if x < 0.0 || y < 0.0 {
            return None;
        }
        let ix = x as usize;
        let iy = y as usize;
        if ix + 1 >= self.width || iy + 1 >= self.height {
            return None;
        }
        let fx = x - ix as Scalar;
        let fy = y - iy as Scalar;
        let v00 = self.data[iy * self.stride + ix];
        let v10 = self.data[iy * self.stride + ix + 1];
        let v01 = self.data[(iy + 1) * self.stride + ix];
        let v11 = self.data[(iy + 1) * self.stride + ix + 1];
        let top = v00 + fx * (v10 - v00);
        let bot = v01 + fx * (v11 - v01);
        Some(top + fy * (bot - top))
    }
}

impl ImageView<'_, u8> {
    /// Bilinear interpolation at subpixel position, returning `f32`.
    #[inline]
    pub fn sample(&self, x: Scalar, y: Scalar) -> Option<Scalar> {
        if x < 0.0 || y < 0.0 {
            return None;
        }
        let ix = x as usize;
        let iy = y as usize;
        if ix + 1 >= self.width || iy + 1 >= self.height {
            return None;
        }
        let fx = x - ix as Scalar;
        let fy = y - iy as Scalar;
        let v00 = self.data[iy * self.stride + ix] as Scalar;
        let v10 = self.data[iy * self.stride + ix + 1] as Scalar;
        let v01 = self.data[(iy + 1) * self.stride + ix] as Scalar;
        let v11 = self.data[(iy + 1) * self.stride + ix + 1] as Scalar;
        let top = v00 + fx * (v10 - v00);
        let bot = v01 + fx * (v11 - v01);
        Some(top + fy * (bot - top))
    }
}

// ---------------------------------------------------------------------------
// OwnedImage
// ---------------------------------------------------------------------------

/// Owned contiguous image buffer.
///
/// Generic over element type `T`. Stride always equals width (contiguous rows).
pub struct OwnedImage<T> {
    data: Vec<T>,
    width: usize,
    height: usize,
}

impl<T> OwnedImage<T> {
    /// Creates an owned image from an existing buffer.
    pub fn from_vec(data: Vec<T>, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(RadSymError::InvalidDimensions { width, height });
        }
        let needed = width
            .checked_mul(height)
            .ok_or(RadSymError::InvalidDimensions { width, height })?;
        if data.len() < needed {
            return Err(RadSymError::BufferTooSmall {
                needed,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Borrows this image as an [`ImageView`].
    pub fn view(&self) -> ImageView<'_, T> {
        ImageView {
            data: &self.data,
            width: self.width,
            height: self.height,
            stride: self.width,
        }
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The backing buffer in row-major order.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to the backing buffer.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Clone + Default> OwnedImage<T> {
    /// Creates a zero-initialized image.
    pub fn zeros(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(RadSymError::InvalidDimensions { width, height });
        }
        let len = width
            .checked_mul(height)
            .ok_or(RadSymError::InvalidDimensions { width, height })?;
        Ok(Self {
            data: vec![T::default(); len],
            width,
            height,
        })
    }
}

impl OwnedImage<Scalar> {
    /// Mutable element access at `(x, y)`.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Scalar> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get_mut(y * self.width + x)
    }

    /// Element access at `(x, y)`.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<Scalar> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get(y * self.width + x).copied()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn required_len(width: usize, height: usize, stride: usize) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(RadSymError::InvalidDimensions { width, height });
    }
    if stride < width {
        return Err(RadSymError::InvalidStride { width, stride });
    }
    let needed = (height - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(width))
        .ok_or(RadSymError::InvalidDimensions { width, height })?;
    Ok(needed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_from_slice() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let view = ImageView::from_slice(&data, 4, 3).unwrap();
        assert_eq!(view.width(), 4);
        assert_eq!(view.height(), 3);
        assert_eq!(*view.get(2, 1).unwrap(), 6.0);
    }

    #[test]
    fn view_with_stride() {
        // 3 wide, stride 4 (1 padding element per row), 2 rows
        let data = [1.0f32, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0];
        let view = ImageView::new(&data, 3, 2, 4).unwrap();
        assert_eq!(*view.get(0, 1).unwrap(), 4.0);
        assert_eq!(*view.get(2, 1).unwrap(), 6.0);
    }

    #[test]
    fn view_out_of_bounds() {
        let data = [0.0f32; 4];
        let view = ImageView::from_slice(&data, 2, 2).unwrap();
        assert!(view.get(2, 0).is_none());
        assert!(view.get(0, 2).is_none());
    }

    #[test]
    fn bilinear_sample_f32() {
        // 2x2 image: [[0, 1], [2, 3]]
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let view = ImageView::from_slice(&data, 2, 2).unwrap();
        // Center of the 2x2 image
        let val = view.sample(0.5, 0.5).unwrap();
        assert!((val - 1.5).abs() < 1e-6, "expected 1.5, got {val}");
    }

    #[test]
    fn bilinear_sample_u8() {
        let data = [0u8, 100, 200, 50];
        let view = ImageView::from_slice(&data, 2, 2).unwrap();
        let val = view.sample(0.0, 0.0).unwrap();
        assert!((val - 0.0).abs() < 1e-3);
    }

    #[test]
    fn owned_image_zeros() {
        let img: OwnedImage<f32> = OwnedImage::zeros(8, 8).unwrap();
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
        assert_eq!(img.data().len(), 64);
        assert!(img.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn owned_image_view_roundtrip() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let img = OwnedImage::from_vec(data, 3, 2).unwrap();
        let view = img.view();
        assert_eq!(*view.get(1, 0).unwrap(), 1.0);
        assert_eq!(*view.get(2, 1).unwrap(), 5.0);
    }

    #[test]
    fn roi_extraction() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let view = ImageView::from_slice(&data, 4, 4).unwrap();
        let sub = view.roi(1, 1, 2, 2).unwrap();
        assert_eq!(sub.width(), 2);
        assert_eq!(sub.height(), 2);
        assert_eq!(*sub.get(0, 0).unwrap(), 5.0); // row 1, col 1 of original
        assert_eq!(*sub.get(1, 1).unwrap(), 10.0); // row 2, col 2 of original
    }

    #[test]
    fn buffer_too_small() {
        let data = [0.0f32; 3];
        let result = ImageView::from_slice(&data, 2, 2);
        assert!(result.is_err());
    }
}
