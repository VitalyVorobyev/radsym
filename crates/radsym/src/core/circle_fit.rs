//! Algebraic circle fitting via the Kasa method.
//!
//! Fits a circle to a set of 2D points by minimizing the algebraic distance
//! to the implicit circle equation `x^2 + y^2 + Dx + Ey + F = 0`.
//!
//! Reference: Kasa, I. (1976). *A circle fitting procedure and its error
//! analysis.* IEEE Transactions on Instrumentation and Measurement, 25(1), 8-14.

use nalgebra::{SMatrix, SVector};

use super::coords::PixelCoord;
use super::geometry::Circle;
use super::scalar::Scalar;

/// Fit a circle to weighted 2D points using the Kasa method.
///
/// Each point contributes to the normal equations proportionally to its
/// weight. Weights are clamped to a minimum of `1e-3` to avoid
/// degeneracies.
///
/// Returns `None` if fewer than 3 points are provided, if the normal
/// equations are singular, or if the solution yields a degenerate circle.
pub fn fit_circle_weighted(points: &[PixelCoord], weights: &[Scalar]) -> Option<Circle> {
    if points.len() < 3 || points.len() != weights.len() {
        return None;
    }

    let mut h = SMatrix::<Scalar, 3, 3>::zeros();
    let mut g = SVector::<Scalar, 3>::zeros();

    for (point, &weight) in points.iter().zip(weights) {
        let row = SVector::<Scalar, 3>::new(point.x, point.y, 1.0);
        let rhs = -(point.x * point.x + point.y * point.y);
        let w = weight.max(1e-3);
        h += row * row.transpose() * w;
        g += row * (rhs * w);
    }

    circle_from_normal_equations(h, g)
}

/// Fit a circle to 2D points using the Kasa method (uniform weights).
///
/// Returns `None` if fewer than 3 points are provided, if the normal
/// equations are singular, or if the solution yields a degenerate circle.
pub fn fit_circle(points: &[PixelCoord]) -> Option<Circle> {
    if points.len() < 3 {
        return None;
    }

    let mut h = SMatrix::<Scalar, 3, 3>::zeros();
    let mut g = SVector::<Scalar, 3>::zeros();

    for point in points {
        let row = SVector::<Scalar, 3>::new(point.x, point.y, 1.0);
        let rhs = -(point.x * point.x + point.y * point.y);
        h += row * row.transpose();
        g += row * rhs;
    }

    circle_from_normal_equations(h, g)
}

fn circle_from_normal_equations(h: SMatrix<Scalar, 3, 3>, g: SVector<Scalar, 3>) -> Option<Circle> {
    let solution = h.lu().solve(&g)?;
    let center = PixelCoord::new(-0.5 * solution[0], -0.5 * solution[1]);
    let radius2 = center.x * center.x + center.y * center.y - solution[2];
    if !center.x.is_finite() || !center.y.is_finite() || !radius2.is_finite() || radius2 <= 1e-6 {
        return None;
    }
    Some(Circle::new(center, radius2.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn circle_points(cx: f32, cy: f32, r: f32, n: usize) -> Vec<PixelCoord> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * PI * i as f32 / n as f32;
                PixelCoord::new(cx + r * angle.cos(), cy + r * angle.sin())
            })
            .collect()
    }

    #[test]
    fn fit_circle_from_exact_points() {
        let points = circle_points(50.0, 50.0, 20.0, 36);
        let c = fit_circle(&points).unwrap();
        assert!((c.center.x - 50.0).abs() < 0.01);
        assert!((c.center.y - 50.0).abs() < 0.01);
        assert!((c.radius - 20.0).abs() < 0.01);
    }

    #[test]
    fn fit_circle_weighted_uniform_matches_unweighted() {
        let points = circle_points(30.0, 40.0, 15.0, 24);
        let weights = vec![1.0; points.len()];
        let c1 = fit_circle(&points).unwrap();
        let c2 = fit_circle_weighted(&points, &weights).unwrap();
        assert!((c1.center.x - c2.center.x).abs() < 1e-4);
        assert!((c1.center.y - c2.center.y).abs() < 1e-4);
        assert!((c1.radius - c2.radius).abs() < 1e-4);
    }

    #[test]
    fn fit_circle_too_few_points() {
        let points = vec![PixelCoord::new(0.0, 0.0), PixelCoord::new(1.0, 0.0)];
        assert!(fit_circle(&points).is_none());
    }

    #[test]
    fn fit_circle_weighted_length_mismatch() {
        let points = circle_points(0.0, 0.0, 10.0, 10);
        let weights = vec![1.0; 5];
        assert!(fit_circle_weighted(&points, &weights).is_none());
    }
}
