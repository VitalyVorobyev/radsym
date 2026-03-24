//! Coordinate types and newtype wrappers.
//!
//! ## Convention
//!
//! All pixel coordinates use **(x = column, y = row)**, matching
//! `nalgebra::Point2<f32>`. x increases rightward, y increases downward.

use nalgebra::Point2;

use super::scalar::Scalar;

/// Subpixel coordinate in the image plane.
///
/// `x` is the column (horizontal), `y` is the row (vertical).
pub type PixelCoord = Point2<Scalar>;

/// Integer pixel index for accumulator addressing.
pub type PixelIndex = Point2<i32>;

/// Newtype wrapper for image-space coordinates.
///
/// Distinguishes raw image coordinates from coordinates in other frames
/// (e.g. a rectified working frame or a world frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePoint(pub PixelCoord);

impl ImagePoint {
    /// Create a new image point from (x, y) coordinates.
    #[inline]
    pub fn new(x: Scalar, y: Scalar) -> Self {
        Self(PixelCoord::new(x, y))
    }

    /// x coordinate (column).
    #[inline]
    pub fn x(&self) -> Scalar {
        self.0.x
    }

    /// y coordinate (row).
    #[inline]
    pub fn y(&self) -> Scalar {
        self.0.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_point_construction() {
        let p = ImagePoint::new(10.5, 20.3);
        assert_eq!(p.x(), 10.5);
        assert_eq!(p.y(), 20.3);
    }

    #[test]
    fn pixel_coord_is_nalgebra_point2() {
        let p: PixelCoord = PixelCoord::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }
}
