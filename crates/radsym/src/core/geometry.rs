//! Geometric primitives for radial symmetry analysis.

use super::coords::PixelCoord;
use super::scalar::Scalar;

/// A circle in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Circle {
    /// Center position (x = col, y = row).
    pub center: PixelCoord,
    /// Radius in pixels.
    pub radius: Scalar,
}

impl Circle {
    /// Create a new circle.
    #[inline]
    pub fn new(center: PixelCoord, radius: Scalar) -> Self {
        Self { center, radius }
    }
}

/// An ellipse in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ellipse {
    /// Center position (x = col, y = row).
    pub center: PixelCoord,
    /// Semi-major axis length in pixels.
    pub semi_major: Scalar,
    /// Semi-minor axis length in pixels.
    pub semi_minor: Scalar,
    /// Orientation angle of the semi-major axis in radians, measured
    /// counter-clockwise from the positive x-axis.
    pub angle: Scalar,
}

impl Ellipse {
    /// Create a new ellipse.
    #[inline]
    pub fn new(center: PixelCoord, semi_major: Scalar, semi_minor: Scalar, angle: Scalar) -> Self {
        Self {
            center,
            semi_major,
            semi_minor,
            angle,
        }
    }

    /// The eccentricity of the ellipse, in `[0, 1)`.
    #[inline]
    pub fn eccentricity(&self) -> Scalar {
        let a = self.semi_major;
        let b = self.semi_minor;
        if a <= 0.0 {
            return 0.0;
        }
        (1.0 - (b * b) / (a * a)).sqrt()
    }
}

impl From<Circle> for Ellipse {
    fn from(c: Circle) -> Self {
        Self {
            center: c.center,
            semi_major: c.radius,
            semi_minor: c.radius,
            angle: 0.0,
        }
    }
}

/// A circular annulus (ring) in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Annulus {
    /// Center position.
    pub center: PixelCoord,
    /// Inner radius in pixels.
    pub inner_radius: Scalar,
    /// Outer radius in pixels.
    pub outer_radius: Scalar,
}

impl Annulus {
    /// Create a new annulus. Panics if `inner_radius > outer_radius`.
    #[inline]
    pub fn new(center: PixelCoord, inner_radius: Scalar, outer_radius: Scalar) -> Self {
        debug_assert!(
            inner_radius <= outer_radius,
            "inner_radius ({inner_radius}) must be <= outer_radius ({outer_radius})"
        );
        Self {
            center,
            inner_radius,
            outer_radius,
        }
    }

    /// Radial thickness of the annulus.
    #[inline]
    pub fn thickness(&self) -> Scalar {
        self.outer_radius - self.inner_radius
    }

    /// Mean radius of the annulus.
    #[inline]
    pub fn mean_radius(&self) -> Scalar {
        0.5 * (self.inner_radius + self.outer_radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_construction() {
        let c = Circle::new(PixelCoord::new(100.0, 200.0), 15.0);
        assert_eq!(c.center.x, 100.0);
        assert_eq!(c.radius, 15.0);
    }

    #[test]
    fn ellipse_from_circle() {
        let c = Circle::new(PixelCoord::new(50.0, 50.0), 10.0);
        let e: Ellipse = c.into();
        assert_eq!(e.semi_major, 10.0);
        assert_eq!(e.semi_minor, 10.0);
        assert_eq!(e.angle, 0.0);
        assert!(e.eccentricity() < 1e-6);
    }

    #[test]
    fn ellipse_eccentricity() {
        let e = Ellipse::new(PixelCoord::new(0.0, 0.0), 10.0, 6.0, 0.0);
        let expected = (1.0 - 36.0 / 100.0_f32).sqrt();
        assert!((e.eccentricity() - expected).abs() < 1e-6);
    }

    #[test]
    fn annulus_properties() {
        let a = Annulus::new(PixelCoord::new(0.0, 0.0), 5.0, 15.0);
        assert_eq!(a.thickness(), 10.0);
        assert_eq!(a.mean_radius(), 10.0);
    }
}
