//! Homography-aware refinement by fitting a circle in rectified space.

use std::f32::consts::PI;

use crate::core::coords::PixelCoord;
use crate::core::error::Result;
use crate::core::geometry::{Circle, Ellipse};
use crate::core::gradient::GradientField;
use crate::core::homography::{rectified_circle_to_image_ellipse, Homography};
use crate::core::scalar::Scalar;

use super::circle_fit::{
    approximate_rectified_circle_from_image_ellipse, circle_residual_and_jacobian,
    compare_circle_fit_quality, fit_circle_initial, min_inlier_count,
    refine_circle_from_observations, CircleFitOutcome,
};
use super::edge_profiles::{
    best_hypotheses, clamp_center_shift, edge_candidates_along_ray, infer_expected_sign,
    select_best_consistent_candidates, DEFAULT_MAX_EDGE_CANDIDATES, DEFAULT_PEAK_MIN_SEPARATION_PX,
};
use super::ellipse_fit::mean_radius;
use super::radial_center::{radial_center_refine_from_gradient, RadialCenterConfig};
use super::result::RefinementStatus;

const RADIAL_SAMPLE_STEP: Scalar = 1.0;
const NORMAL_SAMPLE_STEP: Scalar = 0.75;

/// Configuration for homography-aware ellipse refinement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HomographyEllipseRefineConfig {
    /// Maximum refinement iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for center and radius updates in rectified pixels.
    pub convergence_tol: Scalar,
    /// Radial center config for bounded image-space seed stabilization.
    pub radial_center: RadialCenterConfig,
    /// Number of angular sectors used for image-space edge acquisition.
    pub ray_count: usize,
    /// Inner radius factor for the initial radial search.
    pub radial_search_inner: Scalar,
    /// Outer radius factor for the initial radial search.
    pub radial_search_outer: Scalar,
    /// Half-width of the image-space ellipse normal search window.
    pub normal_search_half_width: Scalar,
    /// Minimum accepted rectified angular coverage.
    pub min_inlier_coverage: Scalar,
    /// Maximum center shift from the initial rectified circle, as a radius fraction.
    pub max_center_shift_fraction: Scalar,
    /// Maximum radius change from the initial rectified circle, as a radius fraction.
    pub max_radius_change_fraction: Scalar,
}

impl Default for HomographyEllipseRefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_tol: 0.1,
            radial_center: RadialCenterConfig::default(),
            ray_count: 96,
            radial_search_inner: 0.6,
            radial_search_outer: 1.45,
            normal_search_half_width: 6.0,
            min_inlier_coverage: 0.45,
            max_center_shift_fraction: 0.4,
            max_radius_change_fraction: 0.6,
        }
    }
}

/// Result of homography-aware refinement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HomographyRefinementResult {
    /// Final image-space ellipse.
    pub image_ellipse: Ellipse,
    /// Final rectified-frame circle.
    pub rectified_circle: Circle,
    /// Convergence status.
    pub status: RefinementStatus,
    /// Number of outer iterations.
    pub iterations: usize,
    /// RMS residual in image-ellipse coordinates.
    pub image_residual: Scalar,
    /// RMS residual in rectified-circle coordinates.
    pub rectified_residual: Scalar,
    /// Rectified inlier sector coverage.
    pub inlier_coverage: Scalar,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ImageObservation {
    image_point: PixelCoord,
    rectified_point: PixelCoord,
    score: Scalar,
    offset: Scalar,
    signed_projection: Scalar,
    sector: usize,
}

#[derive(Clone, Debug)]
struct CircleHypothesisFit {
    fit: CircleFitOutcome,
    observations: Vec<ImageObservation>,
}

#[inline]
fn observation_rectified_point(observation: &ImageObservation) -> PixelCoord {
    observation.rectified_point
}

#[inline]
fn observation_score(observation: &ImageObservation) -> Scalar {
    observation.score
}

#[inline]
fn observation_offset(observation: &ImageObservation) -> Scalar {
    observation.offset
}

#[inline]
fn observation_signed_projection(observation: &ImageObservation) -> Scalar {
    observation.signed_projection
}

#[inline]
fn observation_sector(observation: &ImageObservation) -> usize {
    observation.sector
}

fn observation_rectified_residual(
    circle: &Circle,
    observation: &ImageObservation,
) -> Option<Scalar> {
    let (residual, _) = circle_residual_and_jacobian(circle, observation.rectified_point)?;
    Some(residual.abs() as Scalar)
}

fn collect_radial_candidates(
    gradient: &GradientField,
    homography: &Homography,
    center: PixelCoord,
    radius: Scalar,
    config: &HomographyEllipseRefineConfig,
    expected_sign: Scalar,
) -> Vec<Vec<ImageObservation>> {
    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let ray_count = config.ray_count.max(16);
    let start = config.radial_search_inner * radius;
    let stop = config.radial_search_outer * radius;
    let sigma = 0.35 * radius.max(1.0);
    let mut sectors = vec![Vec::new(); ray_count];

    for (sector, slot) in sectors.iter_mut().enumerate() {
        let theta = 2.0 * PI * sector as Scalar / ray_count as Scalar;
        let dir_x = theta.cos();
        let dir_y = theta.sin();
        *slot = edge_candidates_along_ray(
            gx_view,
            gy_view,
            center,
            dir_x,
            dir_y,
            start,
            stop,
            RADIAL_SAMPLE_STEP,
            sector,
            radius,
            sigma,
            expected_sign,
            DEFAULT_MAX_EDGE_CANDIDATES,
            DEFAULT_PEAK_MIN_SEPARATION_PX,
            |image_point, score, offset, signed_projection, sector| {
                let rectified_point = homography.map_image_to_rectified(image_point)?;
                Some(ImageObservation {
                    image_point,
                    rectified_point,
                    score,
                    offset,
                    signed_projection,
                    sector,
                })
            },
        );
    }

    sectors
}

fn collect_normal_candidates(
    gradient: &GradientField,
    homography: &Homography,
    image_ellipse: &Ellipse,
    config: &HomographyEllipseRefineConfig,
    expected_sign: Scalar,
) -> Vec<Vec<ImageObservation>> {
    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let ray_count = config.ray_count.max(16);
    let cos_a = image_ellipse.angle.cos();
    let sin_a = image_ellipse.angle.sin();
    let sigma = (config.normal_search_half_width * 0.5).max(1.0);
    let mut sectors = vec![Vec::new(); ray_count];

    for (sector, slot) in sectors.iter_mut().enumerate() {
        let theta = 2.0 * PI * sector as Scalar / ray_count as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let ex = image_ellipse.semi_major * cos_t;
        let ey = image_ellipse.semi_minor * sin_t;
        let base = PixelCoord::new(
            image_ellipse.center.x + ex * cos_a - ey * sin_a,
            image_ellipse.center.y + ex * sin_a + ey * cos_a,
        );

        let nx_local = image_ellipse.semi_minor * cos_t;
        let ny_local = image_ellipse.semi_major * sin_t;
        let n_mag = (nx_local * nx_local + ny_local * ny_local).sqrt();
        if n_mag <= 1e-8 {
            continue;
        }
        let dir_x = (nx_local * cos_a - ny_local * sin_a) / n_mag;
        let dir_y = (nx_local * sin_a + ny_local * cos_a) / n_mag;
        *slot = edge_candidates_along_ray(
            gx_view,
            gy_view,
            base,
            dir_x,
            dir_y,
            -config.normal_search_half_width,
            config.normal_search_half_width,
            NORMAL_SAMPLE_STEP,
            sector,
            0.0,
            sigma,
            expected_sign,
            DEFAULT_MAX_EDGE_CANDIDATES,
            DEFAULT_PEAK_MIN_SEPARATION_PX,
            |image_point, score, offset, signed_projection, sector| {
                let rectified_point = homography.map_image_to_rectified(image_point)?;
                Some(ImageObservation {
                    image_point,
                    rectified_point,
                    score,
                    offset,
                    signed_projection,
                    sector,
                })
            },
        );
    }

    sectors
}

fn select_observations_by_circle(
    sector_candidates: &[Vec<ImageObservation>],
    circle: &Circle,
) -> Vec<ImageObservation> {
    select_best_consistent_candidates(
        sector_candidates,
        observation_score,
        observation_offset,
        |observation| observation_rectified_residual(circle, observation),
    )
}

fn better_circle_hypothesis(
    candidate: &CircleHypothesisFit,
    best: &CircleHypothesisFit,
    seed_radius: Scalar,
) -> bool {
    compare_circle_fit_quality(&candidate.fit, &best.fit, seed_radius).is_gt()
}

fn refine_circle_hypothesis(
    sector_candidates: &[Vec<ImageObservation>],
    observations: Vec<ImageObservation>,
    seed_circle: &Circle,
    config: &HomographyEllipseRefineConfig,
) -> Option<CircleHypothesisFit> {
    let min_count = min_inlier_count(config.ray_count.max(16));
    if observations.len() < min_count {
        return None;
    }

    let initial_circle = fit_circle_initial(
        &observations,
        observation_rectified_point,
        observation_score,
    )?;
    let mut fit = refine_circle_from_observations(
        &observations,
        initial_circle,
        seed_circle,
        config,
        observation_rectified_point,
        observation_score,
        observation_sector,
    )?;
    let mut selected = observations;

    let reassigned = select_observations_by_circle(sector_candidates, &fit.circle);
    if reassigned.len() >= min_count {
        if let Some(refined) = refine_circle_from_observations(
            &reassigned,
            fit.circle,
            seed_circle,
            config,
            observation_rectified_point,
            observation_score,
            observation_sector,
        ) {
            selected = reassigned;
            fit = refined;
        }
    }

    Some(CircleHypothesisFit {
        fit,
        observations: selected,
    })
}

fn initial_circle_fit_from_candidates(
    sector_candidates: &[Vec<ImageObservation>],
    seed_circle: &Circle,
    config: &HomographyEllipseRefineConfig,
) -> Option<CircleHypothesisFit> {
    let hypotheses = best_hypotheses(sector_candidates, observation_score, observation_offset);
    let mut best = None;
    for observations in hypotheses {
        let Some(candidate) =
            refine_circle_hypothesis(sector_candidates, observations, seed_circle, config)
        else {
            continue;
        };
        let replace = match best.as_ref() {
            None => true,
            Some(current) => better_circle_hypothesis(&candidate, current, seed_circle.radius),
        };
        if replace {
            best = Some(candidate);
        }
    }
    best
}

fn ellipse_residual(ellipse: &Ellipse, point: PixelCoord) -> Option<Scalar> {
    if ellipse.semi_major <= 1e-8 || ellipse.semi_minor <= 1e-8 {
        return None;
    }
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let dx = point.x - ellipse.center.x;
    let dy = point.y - ellipse.center.y;
    let qx = cos_a * dx + sin_a * dy;
    let qy = -sin_a * dx + cos_a * dy;
    let s = (qx * qx / (ellipse.semi_major * ellipse.semi_major)
        + qy * qy / (ellipse.semi_minor * ellipse.semi_minor))
        .max(1e-12);
    Some(s.sqrt() - 1.0)
}

fn image_residual_from_observations(
    image_ellipse: &Ellipse,
    observations: &[ImageObservation],
) -> Scalar {
    let mut residual_sum = 0.0;
    let mut count = 0usize;
    for observation in observations {
        if let Some(residual) = ellipse_residual(image_ellipse, observation.image_point) {
            residual_sum += residual * residual;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (residual_sum / count as Scalar).sqrt()
    }
}

/// Refine an image-space ellipse by fitting a rectified-space circle.
pub fn refine_ellipse_homography(
    gradient: &GradientField,
    initial_image_ellipse: &Ellipse,
    homography: &Homography,
    config: &HomographyEllipseRefineConfig,
) -> Result<HomographyRefinementResult> {
    let seed_radius = mean_radius(initial_image_ellipse);
    let seed_center = initial_image_ellipse.center;
    let min_count = min_inlier_count(config.ray_count.max(16));

    let radial_center =
        radial_center_refine_from_gradient(gradient, seed_center, &config.radial_center);
    let stabilized_center = match radial_center.status {
        RefinementStatus::Converged => clamp_center_shift(
            seed_center,
            radial_center.hypothesis,
            0.25 * seed_radius.max(1.0),
        ),
        _ => seed_center,
    };
    let fallback_image = *initial_image_ellipse;
    let fallback_rectified = approximate_rectified_circle_from_image_ellipse(
        homography,
        initial_image_ellipse,
        config.ray_count.max(48),
    )
    .unwrap_or_else(|| {
        Circle::new(
            homography
                .map_image_to_rectified(stabilized_center)
                .unwrap_or(PixelCoord::new(stabilized_center.x, stabilized_center.y)),
            seed_radius.max(1.0),
        )
    });
    let radial_candidates = collect_radial_candidates(
        gradient,
        homography,
        stabilized_center,
        seed_radius.max(1.0),
        config,
        0.0,
    );
    let radial_count: usize = radial_candidates.iter().map(Vec::len).sum();

    if radial_count < min_count {
        return Ok(HomographyRefinementResult {
            image_ellipse: fallback_image,
            rectified_circle: fallback_rectified,
            status: RefinementStatus::Degenerate,
            iterations: 0,
            image_residual: 0.0,
            rectified_residual: 0.0,
            inlier_coverage: 0.0,
        });
    }

    let seed_circle = fallback_rectified;
    let Some(mut fit) =
        initial_circle_fit_from_candidates(&radial_candidates, &seed_circle, config)
    else {
        return Ok(HomographyRefinementResult {
            image_ellipse: fallback_image,
            rectified_circle: fallback_rectified,
            status: RefinementStatus::Degenerate,
            iterations: 0,
            image_residual: 0.0,
            rectified_residual: 0.0,
            inlier_coverage: 0.0,
        });
    };

    let expected_sign = infer_expected_sign(
        &fit.observations,
        observation_score,
        observation_signed_projection,
    );
    let mut image_ellipse = rectified_circle_to_image_ellipse(homography, &fit.fit.circle)?;
    let mut last_observations = fit.observations.clone();
    let mut iterations = 1usize;
    let mut status = if fit.fit.converged {
        RefinementStatus::Converged
    } else {
        RefinementStatus::MaxIterations
    };

    for _ in 0..config.max_iterations.saturating_sub(1) {
        let mut normal_candidates =
            collect_normal_candidates(gradient, homography, &image_ellipse, config, expected_sign);
        let mut observations = select_observations_by_circle(&normal_candidates, &fit.fit.circle);
        if observations.len() < min_count && expected_sign.abs() > 0.5 {
            normal_candidates =
                collect_normal_candidates(gradient, homography, &image_ellipse, config, 0.0);
            observations = select_observations_by_circle(&normal_candidates, &fit.fit.circle);
        }
        if observations.len() < min_count {
            break;
        }
        let Some(candidate_fit) = refine_circle_from_observations(
            &observations,
            fit.fit.circle,
            &seed_circle,
            config,
            observation_rectified_point,
            observation_score,
            observation_sector,
        ) else {
            break;
        };
        let reassigned = select_observations_by_circle(&normal_candidates, &candidate_fit.circle);
        let candidate = if reassigned.len() >= min_count {
            if let Some(refined) = refine_circle_from_observations(
                &reassigned,
                candidate_fit.circle,
                &seed_circle,
                config,
                observation_rectified_point,
                observation_score,
                observation_sector,
            ) {
                CircleHypothesisFit {
                    fit: refined,
                    observations: reassigned,
                }
            } else {
                CircleHypothesisFit {
                    fit: candidate_fit,
                    observations,
                }
            }
        } else {
            CircleHypothesisFit {
                fit: candidate_fit,
                observations,
            }
        };
        if !compare_circle_fit_quality(&candidate.fit, &fit.fit, seed_circle.radius).is_gt() {
            status = RefinementStatus::Converged;
            break;
        }

        let center_shift = ((candidate.fit.circle.center.x - fit.fit.circle.center.x).powi(2)
            + (candidate.fit.circle.center.y - fit.fit.circle.center.y).powi(2))
        .sqrt();
        let radius_shift = (candidate.fit.circle.radius - fit.fit.circle.radius).abs();
        fit = candidate;
        image_ellipse = rectified_circle_to_image_ellipse(homography, &fit.fit.circle)?;
        last_observations = fit.observations.clone();
        iterations += 1;
        if fit.fit.converged
            || (center_shift < config.convergence_tol && radius_shift < config.convergence_tol)
        {
            status = RefinementStatus::Converged;
            break;
        }
    }

    if fit.fit.evaluation.coverage < config.min_inlier_coverage
        || fit.fit.evaluation.inlier_count < min_count
    {
        status = RefinementStatus::Degenerate;
    }

    Ok(HomographyRefinementResult {
        image_ellipse,
        rectified_circle: fit.fit.circle,
        status,
        iterations,
        image_residual: image_residual_from_observations(&image_ellipse, &last_observations),
        rectified_residual: fit.fit.evaluation.objective,
        inlier_coverage: fit.fit.evaluation.coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::ImageView;

    fn make_projective_disk(size: usize, homography: &Homography, circle: Circle) -> Vec<u8> {
        let mut data = vec![18u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let point = PixelCoord::new(x as Scalar, y as Scalar);
                let Some(rectified) = homography.map_image_to_rectified(point) else {
                    continue;
                };
                let dx = rectified.x - circle.center.x;
                let dy = rectified.y - circle.center.y;
                if (dx * dx + dy * dy).sqrt() <= circle.radius {
                    data[y * size + x] = 220;
                }
            }
        }
        data
    }

    fn make_projective_concentric_disk(
        size: usize,
        homography: &Homography,
        outer_circle: Circle,
        inner_circle: Circle,
        partial_inner: bool,
    ) -> Vec<u8> {
        let mut data = vec![24u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let point = PixelCoord::new(x as Scalar, y as Scalar);
                let Some(rectified) = homography.map_image_to_rectified(point) else {
                    continue;
                };
                let dx_outer = rectified.x - outer_circle.center.x;
                let dy_outer = rectified.y - outer_circle.center.y;
                let dx_inner = rectified.x - inner_circle.center.x;
                let dy_inner = rectified.y - inner_circle.center.y;
                let mut value =
                    if (dx_outer * dx_outer + dy_outer * dy_outer).sqrt() <= outer_circle.radius {
                        222.0
                    } else {
                        22.0
                    };
                if (dx_inner * dx_inner + dy_inner * dy_inner).sqrt() <= inner_circle.radius {
                    let theta = dy_inner.atan2(dx_inner);
                    let keep_inner = !partial_inner || (-2.25..=0.85).contains(&theta);
                    if keep_inner {
                        value = 2.0;
                    }
                }
                value += 8.0 * ((0.11 * x as Scalar).sin() + (0.09 * y as Scalar).cos());
                data[y * size + x] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
        data
    }

    #[test]
    fn refine_projective_ellipse_via_rectified_circle() {
        let homography = Homography::new([
            [1.12, 0.06, 18.0],
            [0.04, 1.0, 10.0],
            [0.0014, -0.0008, 1.0],
        ])
        .unwrap();
        let rectified_circle = Circle::new(PixelCoord::new(70.0, 64.0), 18.0);
        let true_image_ellipse =
            rectified_circle_to_image_ellipse(&homography, &rectified_circle).unwrap();
        let image = make_projective_disk(140, &homography, rectified_circle);
        let image = ImageView::from_slice(&image, 140, 140).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let initial = Ellipse::new(
            PixelCoord::new(
                true_image_ellipse.center.x + 3.0,
                true_image_ellipse.center.y - 2.0,
            ),
            true_image_ellipse.semi_major * 0.92,
            true_image_ellipse.semi_minor * 1.08,
            true_image_ellipse.angle + 0.08,
        );
        let candidates = collect_radial_candidates(
            &gradient,
            &homography,
            initial.center,
            mean_radius(&initial),
            &HomographyEllipseRefineConfig::default(),
            0.0,
        );
        let observations =
            best_hypotheses(&candidates, observation_score, observation_offset).remove(0);
        let initial_circle_guess = fit_circle_initial(
            &observations,
            observation_rectified_point,
            observation_score,
        );
        let result = refine_ellipse_homography(
            &gradient,
            &initial,
            &homography,
            &HomographyEllipseRefineConfig::default(),
        )
        .unwrap();
        assert!(initial_circle_guess.is_some());
        assert_ne!(result.status, RefinementStatus::Degenerate);
        let center_err = ((result.rectified_circle.center.x - rectified_circle.center.x).powi(2)
            + (result.rectified_circle.center.y - rectified_circle.center.y).powi(2))
        .sqrt();
        assert!(
            center_err < 3.0,
            "rectified center error too large: {center_err}"
        );
        assert!(
            (result.rectified_circle.radius - rectified_circle.radius).abs() < 2.0,
            "rectified radius mismatch: {}",
            result.rectified_circle.radius
        );
    }

    #[test]
    fn default_refiner_prefers_outer_rectified_boundary_with_partial_inner_distractor() {
        let homography = Homography::new([
            [1.12, 0.06, 18.0],
            [0.04, 1.0, 10.0],
            [0.0014, -0.0008, 1.0],
        ])
        .unwrap();
        let outer_circle = Circle::new(PixelCoord::new(70.0, 64.0), 18.0);
        let inner_circle = Circle::new(PixelCoord::new(70.0, 64.0), 10.5);
        let true_image_ellipse =
            rectified_circle_to_image_ellipse(&homography, &outer_circle).unwrap();
        let image =
            make_projective_concentric_disk(140, &homography, outer_circle, inner_circle, true);
        let image = ImageView::from_slice(&image, 140, 140).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let initial = Ellipse::new(
            PixelCoord::new(
                true_image_ellipse.center.x + 3.0,
                true_image_ellipse.center.y - 2.0,
            ),
            true_image_ellipse.semi_major * 0.92,
            true_image_ellipse.semi_minor * 1.08,
            true_image_ellipse.angle + 0.08,
        );
        let config = HomographyEllipseRefineConfig {
            radial_search_inner: 0.3,
            ..HomographyEllipseRefineConfig::default()
        };

        let result = refine_ellipse_homography(&gradient, &initial, &homography, &config).unwrap();
        assert_ne!(result.status, RefinementStatus::Degenerate);
        assert!((result.rectified_circle.radius - outer_circle.radius).abs() < 2.0);
        assert!(result.rectified_circle.radius > inner_circle.radius + 4.0);
    }

    #[test]
    fn degenerate_when_gradient_is_empty() {
        let homography = Homography::identity();
        let image = vec![0u8; 100 * 100];
        let image = ImageView::from_slice(&image, 100, 100).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let initial = Ellipse::new(PixelCoord::new(50.0, 50.0), 14.0, 14.0, 0.0);
        let result = refine_ellipse_homography(
            &gradient,
            &initial,
            &homography,
            &HomographyEllipseRefineConfig::default(),
        )
        .unwrap();
        assert_eq!(result.status, RefinementStatus::Degenerate);
    }
}
