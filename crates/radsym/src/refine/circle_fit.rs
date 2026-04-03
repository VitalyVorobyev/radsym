use std::cmp::Ordering;
use std::f32::consts::PI;

use nalgebra::{SMatrix, SVector};

use crate::core::coords::PixelCoord;
use crate::core::geometry::{Circle, Ellipse};
use crate::core::homography::Homography;
use crate::core::scalar::Scalar;

use super::homography::HomographyEllipseRefineConfig;

// f64 for numerical stability in weighted least-squares circle fitting:
// the 3x3 normal matrix can lose precision in f32 when point coordinates
// are large (e.g., far from origin), leading to inaccurate radius estimates.
type Matrix3x3 = SMatrix<f64, 3, 3>;
type Vector3f = SVector<f64, 3>;

const TRIM_KEEP_FRACTION: Scalar = 0.75;
const MIN_RADIUS_FRACTION: Scalar = 0.55;
const MAX_RADIUS_FRACTION: Scalar = 1.6;
const SOLVER_MAX_STEPS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub(super) struct CircleFitEvaluation {
    pub objective: Scalar,
    pub threshold: Scalar,
    pub coverage: Scalar,
    pub trimmed_mean_edge_score: Scalar,
    pub inlier_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CircleFitOutcome {
    pub circle: Circle,
    pub evaluation: CircleFitEvaluation,
    pub converged: bool,
}

pub(super) fn fit_circle_initial<T, FPoint, FScore>(
    observations: &[T],
    point_of: FPoint,
    score_of: FScore,
) -> Option<Circle>
where
    FPoint: Fn(&T) -> PixelCoord,
    FScore: Fn(&T) -> Scalar,
{
    if observations.len() < 3 {
        return None;
    }

    let mut h = SMatrix::<Scalar, 3, 3>::zeros();
    let mut g = SVector::<Scalar, 3>::zeros();

    for observation in observations {
        let point = point_of(observation);
        let row = SVector::<Scalar, 3>::new(point.x, point.y, 1.0);
        let rhs = -(point.x * point.x + point.y * point.y);
        let weight = score_of(observation).max(1e-3);
        h += row * row.transpose() * weight;
        g += row * (rhs * weight);
    }

    let solution = h.lu().solve(&g)?;
    let center = PixelCoord::new(-0.5 * solution[0], -0.5 * solution[1]);
    let radius2 = center.x * center.x + center.y * center.y - solution[2];
    if !center.x.is_finite() || !center.y.is_finite() || !radius2.is_finite() || radius2 <= 1e-6 {
        return None;
    }
    Some(Circle::new(center, radius2.sqrt()))
}

pub(super) fn fit_circle_from_points(points: &[PixelCoord]) -> Option<Circle> {
    crate::core::circle_fit::fit_circle(points)
}

pub(super) fn approximate_rectified_circle_from_image_ellipse(
    homography: &Homography,
    image_ellipse: &Ellipse,
    samples: usize,
) -> Option<Circle> {
    let sample_count = samples.max(24);
    let cos_a = image_ellipse.angle.cos();
    let sin_a = image_ellipse.angle.sin();
    let mut points = Vec::with_capacity(sample_count);

    for index in 0..sample_count {
        let theta = 2.0 * PI * index as Scalar / sample_count as Scalar;
        let ex = image_ellipse.semi_major * theta.cos();
        let ey = image_ellipse.semi_minor * theta.sin();
        let image_point = PixelCoord::new(
            image_ellipse.center.x + ex * cos_a - ey * sin_a,
            image_ellipse.center.y + ex * sin_a + ey * cos_a,
        );
        let rectified_point = homography.map_image_to_rectified(image_point)?;
        points.push(rectified_point);
    }

    fit_circle_from_points(&points)
}

pub(super) fn circle_residual_and_jacobian(
    circle: &Circle,
    point: PixelCoord,
) -> Option<(f64, Vector3f)> {
    let dx = point.x as f64 - circle.center.x as f64;
    let dy = point.y as f64 - circle.center.y as f64;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist <= 1e-8 || !dist.is_finite() || circle.radius <= 1e-8 {
        return None;
    }
    let residual = dist - circle.radius as f64;
    let jacobian = Vector3f::new(-dx / dist, -dy / dist, -(circle.radius as f64));
    Some((residual, jacobian))
}

pub(super) fn evaluate_circle_fit<T, FPoint, FScore, FSector>(
    observations: &[T],
    circle: &Circle,
    ray_count: usize,
    point_of: FPoint,
    score_of: FScore,
    sector_of: FSector,
) -> Option<CircleFitEvaluation>
where
    FPoint: Fn(&T) -> PixelCoord,
    FScore: Fn(&T) -> Scalar,
    FSector: Fn(&T) -> usize,
{
    if observations.len() < 3 {
        return None;
    }

    let mut residuals = Vec::with_capacity(observations.len());
    for obs in observations {
        let (residual, _) = circle_residual_and_jacobian(circle, point_of(obs))?;
        residuals.push(residual as Scalar);
    }

    let mut sorted = residuals
        .iter()
        .map(|value| value.abs())
        .collect::<Vec<_>>();
    sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    let keep = ((TRIM_KEEP_FRACTION * residuals.len() as Scalar).ceil() as usize)
        .clamp(min_inlier_count(ray_count), residuals.len());
    let threshold = sorted[keep - 1].max(1e-6);

    let mut sectors = vec![false; ray_count];
    let mut weighted_sum = 0.0f64;
    let mut weight_total = 0.0f64;
    let mut edge_sum = 0.0f64;
    let mut inlier_count = 0usize;

    for (obs, residual) in observations.iter().zip(&residuals) {
        if residual.abs() > threshold {
            continue;
        }
        let weight = score_of(obs).max(1e-3) as f64;
        weighted_sum += weight * (*residual as f64) * (*residual as f64);
        weight_total += weight;
        edge_sum += score_of(obs) as f64;
        sectors[sector_of(obs)] = true;
        inlier_count += 1;
    }

    if inlier_count == 0 || weight_total <= 0.0 {
        return None;
    }

    Some(CircleFitEvaluation {
        objective: (weighted_sum / weight_total).sqrt() as Scalar,
        threshold,
        coverage: sectors.iter().filter(|&&used| used).count() as Scalar / ray_count as Scalar,
        trimmed_mean_edge_score: (edge_sum / inlier_count as f64) as Scalar,
        inlier_count,
    })
}

pub(super) fn min_inlier_count(ray_count: usize) -> usize {
    (ray_count / 8).max(8)
}

pub(super) fn guard_circle(
    candidate: &Circle,
    seed_circle: &Circle,
    config: &HomographyEllipseRefineConfig,
) -> bool {
    if !candidate.center.x.is_finite()
        || !candidate.center.y.is_finite()
        || !candidate.radius.is_finite()
        || candidate.radius < MIN_RADIUS_FRACTION * seed_circle.radius
        || candidate.radius > MAX_RADIUS_FRACTION * seed_circle.radius
    {
        return false;
    }

    let dx = candidate.center.x - seed_circle.center.x;
    let dy = candidate.center.y - seed_circle.center.y;
    if (dx * dx + dy * dy).sqrt() > config.max_center_shift_fraction * seed_circle.radius {
        return false;
    }

    if (candidate.radius - seed_circle.radius).abs()
        > config.max_radius_change_fraction * seed_circle.radius
    {
        return false;
    }

    true
}

pub(super) fn compare_circle_fit_quality(
    candidate: &CircleFitOutcome,
    best: &CircleFitOutcome,
    seed_radius: Scalar,
) -> Ordering {
    if (candidate.evaluation.coverage - best.evaluation.coverage).abs() > 1e-4 {
        return candidate
            .evaluation
            .coverage
            .partial_cmp(&best.evaluation.coverage)
            .unwrap_or(Ordering::Equal);
    }
    if (candidate.evaluation.objective - best.evaluation.objective).abs() > 1e-6 {
        return best
            .evaluation
            .objective
            .partial_cmp(&candidate.evaluation.objective)
            .unwrap_or(Ordering::Equal);
    }
    let candidate_radius_error = (candidate.circle.radius - seed_radius).abs();
    let best_radius_error = (best.circle.radius - seed_radius).abs();
    if (candidate_radius_error - best_radius_error).abs() > 0.25 {
        return best_radius_error
            .partial_cmp(&candidate_radius_error)
            .unwrap_or(Ordering::Equal);
    }
    if (candidate.evaluation.trimmed_mean_edge_score - best.evaluation.trimmed_mean_edge_score)
        .abs()
        > 1e-6
    {
        return candidate
            .evaluation
            .trimmed_mean_edge_score
            .partial_cmp(&best.evaluation.trimmed_mean_edge_score)
            .unwrap_or(Ordering::Equal);
    }
    candidate
        .circle
        .radius
        .partial_cmp(&best.circle.radius)
        .unwrap_or(Ordering::Equal)
}

pub(super) fn refine_circle_from_observations<T, FPoint, FScore, FSector>(
    observations: &[T],
    initial: Circle,
    seed_circle: &Circle,
    config: &HomographyEllipseRefineConfig,
    point_of: FPoint,
    score_of: FScore,
    sector_of: FSector,
) -> Option<CircleFitOutcome>
where
    FPoint: Fn(&T) -> PixelCoord + Copy,
    FScore: Fn(&T) -> Scalar + Copy,
    FSector: Fn(&T) -> usize + Copy,
{
    let ray_count = config.ray_count.max(16);
    let min_count = min_inlier_count(ray_count);
    let mut circle = initial;
    let mut evaluation = evaluate_circle_fit(
        observations,
        &circle,
        ray_count,
        point_of,
        score_of,
        sector_of,
    )?;
    if evaluation.inlier_count < min_count || evaluation.coverage < config.min_inlier_coverage {
        return None;
    }

    let mut converged = false;

    for _ in 0..SOLVER_MAX_STEPS {
        let mut h = Matrix3x3::zeros();
        let mut g = Vector3f::zeros();
        let mut used = 0usize;

        for obs in observations {
            let Some((residual, jacobian)) = circle_residual_and_jacobian(&circle, point_of(obs))
            else {
                continue;
            };
            if (residual as Scalar).abs() > evaluation.threshold {
                continue;
            }
            let weight = score_of(obs).max(1e-3) as f64;
            h += jacobian * jacobian.transpose() * weight;
            g += jacobian * (residual * weight);
            used += 1;
        }

        if used < min_count {
            break;
        }

        for i in 0..3 {
            h[(i, i)] += 1e-4;
        }

        let Some(delta) = h.lu().solve(&(-g)) else {
            break;
        };
        if !delta.iter().all(|value| value.is_finite()) || delta.norm() < 1e-8 {
            converged = true;
            break;
        }

        let mut accepted = None;
        for scale in [1.0f64, 0.5, 0.25] {
            let candidate = Circle::new(
                PixelCoord::new(
                    circle.center.x + (delta[0] * scale) as Scalar,
                    circle.center.y + (delta[1] * scale) as Scalar,
                ),
                (circle.radius.ln() as f64 + delta[2] * scale).exp() as Scalar,
            );
            if !guard_circle(&candidate, seed_circle, config) {
                continue;
            }
            let Some(candidate_eval) = evaluate_circle_fit(
                observations,
                &candidate,
                ray_count,
                point_of,
                score_of,
                sector_of,
            ) else {
                continue;
            };
            let improves_objective = candidate_eval.objective + 1e-6 < evaluation.objective;
            let improves_edge_score = (candidate_eval.objective - evaluation.objective).abs()
                <= 1e-6
                && candidate_eval.trimmed_mean_edge_score > evaluation.trimmed_mean_edge_score;
            if candidate_eval.inlier_count < min_count
                || candidate_eval.coverage < config.min_inlier_coverage
                || (!improves_objective && !improves_edge_score)
            {
                continue;
            }
            accepted = Some((candidate, candidate_eval));
            break;
        }

        let Some((candidate, candidate_eval)) = accepted else {
            break;
        };

        let center_shift = ((candidate.center.x - circle.center.x).powi(2)
            + (candidate.center.y - circle.center.y).powi(2))
        .sqrt();
        let radius_shift = (candidate.radius - circle.radius).abs();
        circle = candidate;
        evaluation = candidate_eval;
        if center_shift < config.convergence_tol && radius_shift < config.convergence_tol {
            converged = true;
            break;
        }
    }

    Some(CircleFitOutcome {
        circle,
        evaluation,
        converged,
    })
}
