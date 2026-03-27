//! Homography utilities for projective rectification workflows.
//!
//! The public contract uses an **image -> rectified** homography `H`, where:
//!
//! ```text
//! x_R ~ H x_I
//! ```
//!
//! `x_I` is a point in source image coordinates and `x_R` is a point in a
//! caller-defined rectified pixel frame where the target rim is circular.

use nalgebra::{Matrix2, Matrix3, SymmetricEigen, Vector2, Vector3};

use super::coords::PixelCoord;
use super::error::{RadSymError, Result};
use super::geometry::{Circle, Ellipse};
use super::scalar::Scalar;

const HOMOGRAPHY_EPS: Scalar = 1e-6;

/// Caller-defined rectified raster domain for projective voting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RectifiedGrid {
    /// Rectified width in pixels.
    pub width: usize,
    /// Rectified height in pixels.
    pub height: usize,
}

impl RectifiedGrid {
    /// Create a new rectified raster domain.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(RadSymError::InvalidDimensions { width, height });
        }
        Ok(Self { width, height })
    }
}

/// Validated image-to-rectified homography with cached inverse.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Homography {
    matrix: Matrix3<Scalar>,
    inverse: Matrix3<Scalar>,
}

impl Homography {
    /// Construct from a row-major 3x3 matrix.
    pub fn new(matrix: [[Scalar; 3]; 3]) -> Result<Self> {
        let flat = [
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        ];
        Self::from_flat(flat)
    }

    /// Construct from a row-major 9-element array.
    pub fn from_flat(values: [Scalar; 9]) -> Result<Self> {
        if !values.iter().all(|v| v.is_finite()) {
            return Err(RadSymError::InvalidConfig {
                reason: "homography contains NaN or Inf",
            });
        }

        let mut matrix = Matrix3::from_row_slice(&values);
        let scale = if matrix[(2, 2)].abs() > HOMOGRAPHY_EPS {
            matrix[(2, 2)]
        } else {
            matrix.norm()
        };
        if !scale.is_finite() || scale.abs() <= HOMOGRAPHY_EPS {
            return Err(RadSymError::InvalidConfig {
                reason: "homography has zero scale",
            });
        }
        matrix /= scale;

        let Some(inverse) = matrix.try_inverse() else {
            return Err(RadSymError::InvalidConfig {
                reason: "homography is singular",
            });
        };

        if !inverse.iter().all(|v| v.is_finite()) {
            return Err(RadSymError::InvalidConfig {
                reason: "homography inverse contains NaN or Inf",
            });
        }

        Ok(Self { matrix, inverse })
    }

    /// The identity projective map.
    #[inline]
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
            inverse: Matrix3::identity(),
        }
    }

    /// Borrow the image-to-rectified 3x3 matrix.
    #[inline]
    pub fn matrix(&self) -> &Matrix3<Scalar> {
        &self.matrix
    }

    /// Borrow the rectified-to-image inverse 3x3 matrix.
    #[inline]
    pub fn inverse_matrix(&self) -> &Matrix3<Scalar> {
        &self.inverse
    }

    /// Return the row-major matrix coefficients.
    #[inline]
    pub fn to_flat(&self) -> [Scalar; 9] {
        [
            self.matrix[(0, 0)],
            self.matrix[(0, 1)],
            self.matrix[(0, 2)],
            self.matrix[(1, 0)],
            self.matrix[(1, 1)],
            self.matrix[(1, 2)],
            self.matrix[(2, 0)],
            self.matrix[(2, 1)],
            self.matrix[(2, 2)],
        ]
    }

    /// Map an image-frame point into the rectified frame.
    #[inline]
    pub fn map_image_to_rectified(&self, point: PixelCoord) -> Option<PixelCoord> {
        map_homogeneous(&self.matrix, point)
    }

    /// Map a rectified-frame point back into the source image.
    #[inline]
    pub fn map_rectified_to_image(&self, point: PixelCoord) -> Option<PixelCoord> {
        map_homogeneous(&self.inverse, point)
    }

    /// Local Jacobian of the image -> rectified inhomogeneous mapping.
    pub fn jacobian_image_to_rectified(&self, point: PixelCoord) -> Option<Matrix2<Scalar>> {
        jacobian_at(&self.matrix, point)
    }

    /// Local Jacobian of the rectified -> image inhomogeneous mapping.
    pub fn jacobian_rectified_to_image(&self, point: PixelCoord) -> Option<Matrix2<Scalar>> {
        jacobian_at(&self.inverse, point)
    }

    /// Transform an image gradient covector into the rectified frame.
    pub fn transform_gradient_image_to_rectified(
        &self,
        image_point: PixelCoord,
        gradient: Vector2<Scalar>,
    ) -> Option<Vector2<Scalar>> {
        let jacobian = self.jacobian_image_to_rectified(image_point)?;
        let inv = jacobian.try_inverse()?;
        let result = inv.transpose() * gradient;
        if result.iter().all(|v| v.is_finite()) {
            Some(result)
        } else {
            None
        }
    }

    /// Pull back a rectified normal covector into image coordinates.
    pub fn pullback_rectified_normal_to_image(
        &self,
        rectified_point: PixelCoord,
        rectified_normal: Vector2<Scalar>,
    ) -> Option<Vector2<Scalar>> {
        let jacobian = self.jacobian_rectified_to_image(rectified_point)?;
        let result = jacobian.transpose() * rectified_normal;
        if result.iter().all(|v| v.is_finite()) {
            Some(result)
        } else {
            None
        }
    }
}

fn map_homogeneous(matrix: &Matrix3<Scalar>, point: PixelCoord) -> Option<PixelCoord> {
    let homogeneous = matrix * Vector3::new(point.x, point.y, 1.0);
    let w = homogeneous[2];
    if !w.is_finite() || w.abs() <= HOMOGRAPHY_EPS {
        return None;
    }
    let x = homogeneous[0] / w;
    let y = homogeneous[1] / w;
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some(PixelCoord::new(x, y))
}

fn jacobian_at(matrix: &Matrix3<Scalar>, point: PixelCoord) -> Option<Matrix2<Scalar>> {
    let x = point.x;
    let y = point.y;
    let u = matrix[(0, 0)] * x + matrix[(0, 1)] * y + matrix[(0, 2)];
    let v = matrix[(1, 0)] * x + matrix[(1, 1)] * y + matrix[(1, 2)];
    let w = matrix[(2, 0)] * x + matrix[(2, 1)] * y + matrix[(2, 2)];
    if !w.is_finite() || w.abs() <= HOMOGRAPHY_EPS {
        return None;
    }
    let w2 = w * w;
    let j = Matrix2::new(
        (matrix[(0, 0)] * w - u * matrix[(2, 0)]) / w2,
        (matrix[(0, 1)] * w - u * matrix[(2, 1)]) / w2,
        (matrix[(1, 0)] * w - v * matrix[(2, 0)]) / w2,
        (matrix[(1, 1)] * w - v * matrix[(2, 1)]) / w2,
    );
    if !j.iter().all(|v| v.is_finite()) || j.determinant().abs() <= HOMOGRAPHY_EPS {
        return None;
    }
    Some(j)
}

fn circle_conic(circle: &Circle) -> Matrix3<Scalar> {
    let cx = circle.center.x;
    let cy = circle.center.y;
    let r2 = circle.radius * circle.radius;
    Matrix3::new(
        1.0,
        0.0,
        -cx,
        0.0,
        1.0,
        -cy,
        -cx,
        -cy,
        cx * cx + cy * cy - r2,
    )
}

fn conic_to_ellipse(conic: &Matrix3<Scalar>) -> Option<Ellipse> {
    let q = 0.5 * (conic + conic.transpose());
    let a = q[(0, 0)];
    let b = 2.0 * q[(0, 1)];
    let c = q[(1, 1)];
    let d = 2.0 * q[(0, 2)];
    let e = 2.0 * q[(1, 2)];
    let f = q[(2, 2)];

    let linear = Matrix2::new(2.0 * a, b, b, 2.0 * c);
    let rhs = Vector2::new(-d, -e);
    let center = linear.try_inverse()? * rhs;
    if !center.iter().all(|v| v.is_finite()) {
        return None;
    }

    let s = Matrix2::new(a, 0.5 * b, 0.5 * b, c);
    let cvec = Vector2::new(center[0], center[1]);
    let translated_constant = cvec.dot(&(s * cvec)) + d * cvec[0] + e * cvec[1] + f;
    if !translated_constant.is_finite() || translated_constant >= -HOMOGRAPHY_EPS {
        return None;
    }

    let eigen = SymmetricEigen::new(s);
    let l0 = eigen.eigenvalues[0];
    let l1 = eigen.eigenvalues[1];
    if l0 <= HOMOGRAPHY_EPS || l1 <= HOMOGRAPHY_EPS {
        return None;
    }

    let axis0 = (-translated_constant / l0).sqrt();
    let axis1 = (-translated_constant / l1).sqrt();
    if !axis0.is_finite() || !axis1.is_finite() {
        return None;
    }

    let v0 = eigen.eigenvectors.column(0);
    let v1 = eigen.eigenvectors.column(1);
    let (semi_major, semi_minor, axis) = if axis0 >= axis1 {
        (axis0, axis1, v0)
    } else {
        (axis1, axis0, v1)
    };
    let angle = axis[1].atan2(axis[0]);

    if semi_major <= HOMOGRAPHY_EPS || semi_minor <= HOMOGRAPHY_EPS {
        return None;
    }

    Some(Ellipse::new(
        PixelCoord::new(center[0], center[1]),
        semi_major,
        semi_minor,
        angle,
    ))
}

/// Transport a rectified-frame circle back to an image-space ellipse.
pub fn rectified_circle_to_image_ellipse(
    homography: &Homography,
    circle: &Circle,
) -> Result<Ellipse> {
    if !circle.radius.is_finite() || circle.radius <= HOMOGRAPHY_EPS {
        return Err(RadSymError::DegenerateHypothesis {
            reason: "rectified circle radius must be positive",
        });
    }

    let rectified_conic = circle_conic(circle);
    let image_conic = homography.matrix.transpose() * rectified_conic * homography.matrix;
    conic_to_ellipse(&image_conic).ok_or(RadSymError::RefinementFailed {
        reason: "projective circle transport did not yield a valid ellipse",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ellipse_level(ellipse: &Ellipse, point: PixelCoord) -> Scalar {
        let cos_a = ellipse.angle.cos();
        let sin_a = ellipse.angle.sin();
        let dx = point.x - ellipse.center.x;
        let dy = point.y - ellipse.center.y;
        let lx = dx * cos_a + dy * sin_a;
        let ly = -dx * sin_a + dy * cos_a;
        (lx / ellipse.semi_major).powi(2) + (ly / ellipse.semi_minor).powi(2)
    }

    #[test]
    fn identity_roundtrip_point() {
        let h = Homography::identity();
        let p = PixelCoord::new(12.5, 24.0);
        let q = h.map_image_to_rectified(p).unwrap();
        let r = h.map_rectified_to_image(q).unwrap();
        assert!((r.x - p.x).abs() < 1e-6);
        assert!((r.y - p.y).abs() < 1e-6);
    }

    #[test]
    fn jacobian_matches_affine_case() {
        let h = Homography::new([[2.0, 0.5, 3.0], [0.25, 1.5, -1.0], [0.0, 0.0, 1.0]]).unwrap();
        let j = h
            .jacobian_image_to_rectified(PixelCoord::new(10.0, 20.0))
            .unwrap();
        assert!((j[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((j[(0, 1)] - 0.5).abs() < 1e-6);
        assert!((j[(1, 0)] - 0.25).abs() < 1e-6);
        assert!((j[(1, 1)] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn transported_circle_matches_sampled_boundary() {
        let homography =
            Homography::new([[1.2, 0.1, 15.0], [0.05, 0.9, -8.0], [0.0015, -0.0008, 1.0]]).unwrap();
        let circle = Circle::new(PixelCoord::new(80.0, 60.0), 24.0);
        let ellipse = rectified_circle_to_image_ellipse(&homography, &circle).unwrap();

        for i in 0..64 {
            let theta = 2.0 * std::f32::consts::PI * i as Scalar / 64.0;
            let rectified = PixelCoord::new(
                circle.center.x + circle.radius * theta.cos(),
                circle.center.y + circle.radius * theta.sin(),
            );
            let image = homography.map_rectified_to_image(rectified).unwrap();
            let level = ellipse_level(&ellipse, image);
            assert!(
                (level - 1.0).abs() < 2e-2,
                "ellipse level mismatch: {level}"
            );
        }
    }

    #[test]
    fn projective_transport_does_not_preserve_circle_center() {
        let homography =
            Homography::new([[1.1, 0.08, 10.0], [0.02, 0.95, 5.0], [0.0018, -0.0011, 1.0]])
                .unwrap();
        let circle = Circle::new(PixelCoord::new(90.0, 70.0), 18.0);
        let ellipse = rectified_circle_to_image_ellipse(&homography, &circle).unwrap();
        let mapped_center = homography.map_rectified_to_image(circle.center).unwrap();
        let delta = ((ellipse.center.x - mapped_center.x).powi(2)
            + (ellipse.center.y - mapped_center.y).powi(2))
        .sqrt();
        assert!(
            delta > 0.5,
            "expected a projective center mismatch, got {delta}"
        );
    }
}
