use std::cmp::Ordering;
use std::f32::consts::{FRAC_PI_2, PI};

use nalgebra::{SMatrix, SVector};

use crate::core::coords::PixelCoord;
use crate::core::geometry::Ellipse;
use crate::core::scalar::Scalar;

use super::ellipse::EllipseRefineConfig;

type Matrix5 = SMatrix<f64, 5, 5>;
type Vector5 = SVector<f64, 5>;

const MIN_MINOR_RADIUS_FRACTION: Scalar = 0.55;
const MAX_MAJOR_RADIUS_FRACTION: Scalar = 1.6;
const TRIM_KEEP_FRACTION: Scalar = 0.75;
const SOLVER_MAX_STEPS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub(super) struct EllipseFitEvaluation {
    pub objective: Scalar,
    pub threshold: Scalar,
    pub trimmed_mean_edge_score: Scalar,
    pub coverage: Scalar,
    pub inlier_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EllipseFitOutcome {
    pub ellipse: Ellipse,
    pub evaluation: EllipseFitEvaluation,
    pub inner_iterations: usize,
    pub converged: bool,
}

#[inline]
fn wrap_half_turn(mut angle: Scalar) -> Scalar {
    while angle <= -FRAC_PI_2 {
        angle += PI;
    }
    while angle > FRAC_PI_2 {
        angle -= PI;
    }
    angle
}

pub(super) fn normalize_ellipse(
    center: PixelCoord,
    semi_major: Scalar,
    semi_minor: Scalar,
    angle: Scalar,
) -> Option<Ellipse> {
    if !center.x.is_finite()
        || !center.y.is_finite()
        || !semi_major.is_finite()
        || !semi_minor.is_finite()
        || !angle.is_finite()
        || semi_major <= 1e-6
        || semi_minor <= 1e-6
    {
        return None;
    }

    if semi_major >= semi_minor {
        Some(Ellipse::new(
            center,
            semi_major,
            semi_minor,
            wrap_half_turn(angle),
        ))
    } else {
        Some(Ellipse::new(
            center,
            semi_minor,
            semi_major,
            wrap_half_turn(angle + FRAC_PI_2),
        ))
    }
}

pub(super) fn guard_ellipse(
    ellipse: &Ellipse,
    seed_center: PixelCoord,
    seed_radius: Scalar,
    config: &EllipseRefineConfig,
    width: usize,
    height: usize,
) -> bool {
    if !ellipse.center.x.is_finite()
        || !ellipse.center.y.is_finite()
        || !ellipse.semi_major.is_finite()
        || !ellipse.semi_minor.is_finite()
        || !ellipse.angle.is_finite()
        || ellipse.semi_minor < MIN_MINOR_RADIUS_FRACTION * seed_radius
        || ellipse.semi_major > MAX_MAJOR_RADIUS_FRACTION * seed_radius
        || ellipse.semi_major / ellipse.semi_minor > config.max_axis_ratio
    {
        return false;
    }

    let dx = ellipse.center.x - seed_center.x;
    let dy = ellipse.center.y - seed_center.y;
    if (dx * dx + dy * dy).sqrt() > config.max_center_shift_fraction * seed_radius {
        return false;
    }

    ellipse.center.x >= 0.0
        && ellipse.center.y >= 0.0
        && ellipse.center.x < width as Scalar
        && ellipse.center.y < height as Scalar
}

pub(super) fn min_inlier_count(ray_count: usize) -> usize {
    (ray_count / 6).max(16)
}

pub(super) fn weighted_covariance_guess<T, FPoint, FScore>(
    observations: &[T],
    fallback_center: PixelCoord,
    seed_radius: Scalar,
    point_of: FPoint,
    score_of: FScore,
) -> Ellipse
where
    FPoint: Fn(&T) -> PixelCoord,
    FScore: Fn(&T) -> Scalar,
{
    if observations.len() < 5 {
        return Ellipse::new(fallback_center, seed_radius, seed_radius, 0.0);
    }

    let mut ranked = observations.iter().collect::<Vec<_>>();
    ranked.sort_by(|lhs, rhs| {
        score_of(rhs)
            .partial_cmp(&score_of(lhs))
            .unwrap_or(Ordering::Equal)
    });
    ranked.truncate((ranked.len() * 3 / 4).max(12));

    let mut wsum = 0.0f64;
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for obs in &ranked {
        let point = point_of(obs);
        let weight = score_of(obs).max(1e-3) as f64;
        wsum += weight;
        mean_x += weight * point.x as f64;
        mean_y += weight * point.y as f64;
    }

    if wsum <= 0.0 {
        return Ellipse::new(fallback_center, seed_radius, seed_radius, 0.0);
    }

    mean_x /= wsum;
    mean_y /= wsum;

    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for obs in &ranked {
        let point = point_of(obs);
        let weight = score_of(obs).max(1e-3) as f64;
        let dx = point.x as f64 - mean_x;
        let dy = point.y as f64 - mean_y;
        sxx += weight * dx * dx;
        syy += weight * dy * dy;
        sxy += weight * dx * dy;
    }

    sxx /= wsum;
    syy /= wsum;
    sxy /= wsum;
    let trace = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda1 = ((trace + disc) * 0.5).max(1e-6);
    let lambda2 = ((trace - disc) * 0.5).max(1e-6);
    let mut angle = if sxy.abs() > 1e-10 {
        (lambda1 - sxx).atan2(sxy) as Scalar
    } else if sxx >= syy {
        0.0
    } else {
        FRAC_PI_2
    };

    let mut semi_major = (2.0 * lambda1).sqrt() as Scalar;
    let mut semi_minor = (2.0 * lambda2).sqrt() as Scalar;
    semi_major = semi_major.clamp(
        MIN_MINOR_RADIUS_FRACTION * seed_radius,
        MAX_MAJOR_RADIUS_FRACTION * seed_radius,
    );
    semi_minor = semi_minor.clamp(
        MIN_MINOR_RADIUS_FRACTION * seed_radius,
        MAX_MAJOR_RADIUS_FRACTION * seed_radius,
    );

    if semi_minor > semi_major {
        std::mem::swap(&mut semi_major, &mut semi_minor);
        angle += FRAC_PI_2;
    }

    normalize_ellipse(
        PixelCoord::new(mean_x as Scalar, mean_y as Scalar),
        semi_major,
        semi_minor,
        angle,
    )
    .unwrap_or_else(|| Ellipse::new(fallback_center, seed_radius, seed_radius, 0.0))
}

pub(super) fn ellipse_residual_and_jacobian(
    ellipse: &Ellipse,
    point: PixelCoord,
) -> Option<(f64, Vector5)> {
    let a = ellipse.semi_major as f64;
    let b = ellipse.semi_minor as f64;
    if a <= 1e-8 || b <= 1e-8 {
        return None;
    }

    let cos_a = ellipse.angle.cos() as f64;
    let sin_a = ellipse.angle.sin() as f64;
    let dx = point.x as f64 - ellipse.center.x as f64;
    let dy = point.y as f64 - ellipse.center.y as f64;
    let qx = cos_a * dx + sin_a * dy;
    let qy = -sin_a * dx + cos_a * dy;
    let inv_a2 = 1.0 / (a * a);
    let inv_b2 = 1.0 / (b * b);
    let s = (qx * qx * inv_a2 + qy * qy * inv_b2).max(1e-12);
    let k = s.sqrt();
    let residual = k - 1.0;

    let dr_dcx = (-qx * cos_a * inv_a2 + qy * sin_a * inv_b2) / k;
    let dr_dcy = (-qx * sin_a * inv_a2 - qy * cos_a * inv_b2) / k;
    let dr_dloga = -(qx * qx * inv_a2) / k;
    let dr_dlogb = -(qy * qy * inv_b2) / k;
    let dr_dtheta = (qx * qy * (inv_a2 - inv_b2)) / k;

    Some((
        residual,
        Vector5::new(dr_dcx, dr_dcy, dr_dloga, dr_dlogb, dr_dtheta),
    ))
}

pub(super) fn evaluate_fit<T, FPoint, FScore, FSector>(
    observations: &[T],
    ellipse: &Ellipse,
    ray_count: usize,
    point_of: FPoint,
    score_of: FScore,
    sector_of: FSector,
) -> Option<EllipseFitEvaluation>
where
    FPoint: Fn(&T) -> PixelCoord,
    FScore: Fn(&T) -> Scalar,
    FSector: Fn(&T) -> usize,
{
    if observations.len() < 5 {
        return None;
    }

    let mut residuals = Vec::with_capacity(observations.len());
    for obs in observations {
        let (residual, _) = ellipse_residual_and_jacobian(ellipse, point_of(obs))?;
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
        sectors[sector_of(obs)] = true;
        weighted_sum += weight * (*residual as f64) * (*residual as f64);
        weight_total += weight;
        edge_sum += score_of(obs) as f64;
        inlier_count += 1;
    }

    if inlier_count == 0 || weight_total <= 0.0 {
        return None;
    }

    let coverage = sectors.iter().filter(|&&used| used).count() as Scalar / ray_count as Scalar;
    Some(EllipseFitEvaluation {
        objective: (weighted_sum / weight_total).sqrt() as Scalar,
        threshold,
        trimmed_mean_edge_score: (edge_sum / inlier_count as f64) as Scalar,
        coverage,
        inlier_count,
    })
}

pub(super) fn compare_fit_quality(
    candidate: &EllipseFitOutcome,
    best: &EllipseFitOutcome,
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
    let candidate_radius_error = (candidate.ellipse.mean_radius() - seed_radius).abs();
    let best_radius_error = (best.ellipse.mean_radius() - seed_radius).abs();
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
        .ellipse
        .mean_radius()
        .partial_cmp(&best.ellipse.mean_radius())
        .unwrap_or(Ordering::Equal)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn refine_from_observations<T, FPoint, FScore, FSector>(
    observations: &[T],
    initial: Ellipse,
    seed_center: PixelCoord,
    seed_radius: Scalar,
    config: &EllipseRefineConfig,
    width: usize,
    height: usize,
    point_of: FPoint,
    score_of: FScore,
    sector_of: FSector,
) -> Option<EllipseFitOutcome>
where
    FPoint: Fn(&T) -> PixelCoord + Copy,
    FScore: Fn(&T) -> Scalar + Copy,
    FSector: Fn(&T) -> usize + Copy,
{
    let ray_count = config.ray_count.max(16);
    let min_count = min_inlier_count(ray_count);
    let mut ellipse = initial;
    let mut evaluation = evaluate_fit(
        observations,
        &ellipse,
        ray_count,
        point_of,
        score_of,
        sector_of,
    )?;
    if evaluation.inlier_count < min_count || evaluation.coverage < config.min_inlier_coverage {
        return None;
    }

    let mut converged = false;
    let mut inner_iterations = 0usize;

    for _ in 0..SOLVER_MAX_STEPS {
        inner_iterations += 1;
        let mut h = Matrix5::zeros();
        let mut g = Vector5::zeros();
        let mut used = 0usize;

        for obs in observations {
            let Some((residual, jacobian)) = ellipse_residual_and_jacobian(&ellipse, point_of(obs))
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

        for i in 0..5 {
            h[(i, i)] += 1e-4;
        }

        let Some(delta) = h.lu().solve(&(-g)) else {
            break;
        };
        if !delta.iter().all(|value| value.is_finite()) || delta.norm() < 1e-8 {
            converged = true;
            break;
        }

        let state = Vector5::new(
            ellipse.center.x as f64,
            ellipse.center.y as f64,
            (ellipse.semi_major.max(1e-6)).ln() as f64,
            (ellipse.semi_minor.max(1e-6)).ln() as f64,
            ellipse.angle as f64,
        );

        let mut accepted = None;
        for scale in [1.0f64, 0.5, 0.25] {
            let candidate_state = state + delta * scale;
            let Some(candidate) = normalize_ellipse(
                PixelCoord::new(candidate_state[0] as Scalar, candidate_state[1] as Scalar),
                candidate_state[2].exp() as Scalar,
                candidate_state[3].exp() as Scalar,
                candidate_state[4] as Scalar,
            ) else {
                continue;
            };
            if !guard_ellipse(&candidate, seed_center, seed_radius, config, width, height) {
                continue;
            }
            let Some(candidate_eval) = evaluate_fit(
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

        let center_shift = ((candidate.center.x - ellipse.center.x).powi(2)
            + (candidate.center.y - ellipse.center.y).powi(2))
        .sqrt();
        let axis_shift = (candidate.semi_major - ellipse.semi_major)
            .abs()
            .max((candidate.semi_minor - ellipse.semi_minor).abs());

        ellipse = candidate;
        evaluation = candidate_eval;
        if center_shift < config.convergence_tol && axis_shift < config.convergence_tol {
            converged = true;
            break;
        }
    }

    Some(EllipseFitOutcome {
        ellipse,
        evaluation,
        inner_iterations,
        converged,
    })
}
