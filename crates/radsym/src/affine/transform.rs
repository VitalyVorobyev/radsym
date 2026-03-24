//! Affine transformation types for perspective-aware symmetry detection.
//!
//! Provides a 2D affine map that can be used to warp gradient directions
//! before voting, enabling detection of elliptical symmetry under
//! perspective distortion.

use crate::core::coords::PixelCoord;
use crate::core::scalar::Scalar;

/// A 2×2 affine map (linear part only, no translation).
///
/// Represents the linear component of an affine transformation used to
/// warp gradient offset directions. The full affine is:
///
/// ```text
/// [x']   [a  b] [x]
/// [y'] = [c  d] [y]
/// ```
///
/// For GFRS-style voting, a set of affine maps is sampled from a
/// discrete parameter space, and each map produces a separate
/// accumulator. Peaks in different accumulators indicate different
/// elliptical symmetry orientations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMap {
    /// Element (0,0) of the 2×2 matrix.
    pub a: Scalar,
    /// Element (0,1) of the 2×2 matrix.
    pub b: Scalar,
    /// Element (1,0) of the 2×2 matrix.
    pub c: Scalar,
    /// Element (1,1) of the 2×2 matrix.
    pub d: Scalar,
}

impl AffineMap {
    /// Create an affine map from matrix elements.
    #[inline]
    pub fn new(a: Scalar, b: Scalar, c: Scalar, d: Scalar) -> Self {
        Self { a, b, c, d }
    }

    /// The identity transformation.
    #[inline]
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    /// Create from rotation angle (radians).
    #[inline]
    pub fn rotation(angle: Scalar) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self::new(c, -s, s, c)
    }

    /// Create from anisotropic scaling.
    #[inline]
    pub fn scale(sx: Scalar, sy: Scalar) -> Self {
        Self::new(sx, 0.0, 0.0, sy)
    }

    /// Apply this affine map to a point.
    #[inline]
    pub fn apply(&self, p: PixelCoord) -> PixelCoord {
        PixelCoord::new(self.a * p.x + self.b * p.y, self.c * p.x + self.d * p.y)
    }

    /// Determinant of the 2×2 matrix.
    #[inline]
    pub fn determinant(&self) -> Scalar {
        self.a * self.d - self.b * self.c
    }

    /// Inverse of the affine map, if it exists.
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self::new(
            self.d * inv_det,
            -self.b * inv_det,
            -self.c * inv_det,
            self.a * inv_det,
        ))
    }

    /// Compose two affine maps: self * other.
    #[inline]
    pub fn compose(&self, other: &AffineMap) -> AffineMap {
        AffineMap::new(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )
    }
}

/// Generate a discrete set of affine maps for GFRS-style parameter sampling.
///
/// Samples `num_angles` rotation angles and `num_scales` aspect ratios,
/// producing `num_angles * num_scales` affine maps. Each map is a
/// rotation composed with anisotropic scaling.
pub fn sample_affine_maps(num_angles: usize, num_scales: usize) -> Vec<AffineMap> {
    let mut maps = Vec::with_capacity(num_angles * num_scales);

    for ai in 0..num_angles {
        let angle = std::f32::consts::PI * ai as Scalar / num_angles as Scalar;

        for si in 0..num_scales {
            // Aspect ratio from 0.5 to 1.0 (no need to go > 1 due to angle coverage)
            let aspect = if num_scales <= 1 {
                1.0
            } else {
                0.5 + 0.5 * si as Scalar / (num_scales - 1) as Scalar
            };

            let rot = AffineMap::rotation(angle);
            let scl = AffineMap::scale(1.0, aspect);
            maps.push(rot.compose(&scl));
        }
    }

    maps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_preserves_point() {
        let id = AffineMap::identity();
        let p = PixelCoord::new(3.0, 4.0);
        let q = id.apply(p);
        assert!((q.x - 3.0).abs() < 1e-6);
        assert!((q.y - 4.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_90() {
        let rot = AffineMap::rotation(std::f32::consts::FRAC_PI_2);
        let p = PixelCoord::new(1.0, 0.0);
        let q = rot.apply(p);
        assert!((q.x - 0.0).abs() < 1e-5);
        assert!((q.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inverse_roundtrip() {
        let m = AffineMap::new(2.0, 1.0, 0.5, 3.0);
        let inv = m.inverse().unwrap();
        let id = m.compose(&inv);
        assert!((id.a - 1.0).abs() < 1e-5);
        assert!((id.b).abs() < 1e-5);
        assert!((id.c).abs() < 1e-5);
        assert!((id.d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sample_affine_maps_count() {
        let maps = sample_affine_maps(6, 3);
        assert_eq!(maps.len(), 18);
    }

    #[test]
    fn determinant_of_rotation() {
        let rot = AffineMap::rotation(0.7);
        assert!((rot.determinant() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn singular_has_no_inverse() {
        let m = AffineMap::new(1.0, 2.0, 0.5, 3.0);
        assert!(m.inverse().is_some());

        let singular = AffineMap::new(1.0, 2.0, 2.0, 4.0);
        assert!(singular.inverse().is_none());
    }
}
