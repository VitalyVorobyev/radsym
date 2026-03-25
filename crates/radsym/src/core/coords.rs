//! Coordinate types.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_coord_is_nalgebra_point2() {
        let p: PixelCoord = PixelCoord::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }
}
