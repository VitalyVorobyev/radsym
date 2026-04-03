//! Iterative ellipse refinement from edge-supported observations.
//!
//! The refiner treats the incoming hypothesis as a seed only. It first gathers
//! one radial edge observation per angular sector, fits an initial ellipse to
//! those boundary points, then alternates between local normal-direction edge
//! searches and guarded robust Gauss-Newton updates on ellipse parameters.
//!
//! Literature:
//! - Parthasarathy, R. "Rapid, accurate particle tracking by calculation of
//!   radial symmetry centers." Nature Methods 9, 724–726 (2012).
//! - Fitzgibbon, A., Pilu, M., Fisher, R. "Direct least squares fitting of
//!   ellipses." ICPR 1996. The implementation below uses a nonlinear geometric
//!   refinement initialized from weighted boundary statistics instead of a full
//!   conic solve because the boundary points are already center-seeded.

use std::f32::consts::PI;

use crate::core::coords::PixelCoord;
use crate::core::error::Result;
use crate::core::geometry::Ellipse;
use crate::core::gradient::GradientField;
use crate::core::image_view::ImageView;
use crate::core::scalar::Scalar;
use crate::support::annulus::AnnulusSamplingConfig;

use super::edge_profiles::{
    best_hypotheses, clamp_center_shift, edge_candidates_along_ray, infer_expected_sign,
    select_best_consistent_candidates, DEFAULT_MAX_EDGE_CANDIDATES, DEFAULT_PEAK_MIN_SEPARATION_PX,
};
use super::ellipse_fit::{
    compare_fit_quality, ellipse_residual_and_jacobian, guard_ellipse, min_inlier_count,
    refine_from_observations, weighted_covariance_guess, EllipseFitOutcome,
};
use super::radial_center::{radial_center_refine_from_gradient, RadialCenterConfig};
use super::result::{RefinementResult, RefinementStatus};

const RADIAL_SAMPLE_STEP: Scalar = 1.0;
const NORMAL_SAMPLE_STEP: Scalar = 0.75;

#[derive(Clone, Copy, Debug, PartialEq)]
struct EdgeObservation {
    point: PixelCoord,
    score: Scalar,
    offset: Scalar,
    signed_projection: Scalar,
    sector: usize,
}

#[derive(Clone, Debug)]
struct HypothesisFit {
    fit: EllipseFitOutcome,
    observations: Vec<EdgeObservation>,
}

/// Configuration for iterative ellipse refinement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EllipseRefineConfig {
    /// Maximum refinement iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for center and axis updates (pixels).
    pub convergence_tol: Scalar,
    /// Fractional annulus margin retained for compatibility and downstream diagnostics.
    pub annulus_margin: Scalar,
    /// Radial center config for bounded seed stabilization.
    pub radial_center: RadialCenterConfig,
    /// Legacy annulus sampling config retained for compatibility.
    pub sampling: AnnulusSamplingConfig,
    /// Minimum alignment for legacy annulus diagnostics.
    pub min_alignment: Scalar,
    /// Number of angular sectors used for edge acquisition.
    pub ray_count: usize,
    /// Inner radius factor for the initial radial search.
    pub radial_search_inner: Scalar,
    /// Outer radius factor for the initial radial search.
    pub radial_search_outer: Scalar,
    /// Half-width of the normal-direction search window around the current ellipse.
    pub normal_search_half_width: Scalar,
    /// Minimum sector coverage of accepted inliers.
    pub min_inlier_coverage: Scalar,
    /// Maximum allowed center shift from the original seed, as a fraction of the seed radius.
    pub max_center_shift_fraction: Scalar,
    /// Maximum allowed ellipse axis ratio.
    pub max_axis_ratio: Scalar,
}

impl EllipseRefineConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> crate::core::error::Result<()> {
        use crate::core::error::RadSymError;
        if self.max_iterations == 0 {
            return Err(RadSymError::InvalidConfig {
                reason: "max_iterations must be > 0",
            });
        }
        if self.convergence_tol <= 0.0 {
            return Err(RadSymError::InvalidConfig {
                reason: "convergence_tol must be > 0.0",
            });
        }
        if self.ray_count < 8 {
            return Err(RadSymError::InvalidConfig {
                reason: "ray_count must be >= 8",
            });
        }
        Ok(())
    }
}

impl Default for EllipseRefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_tol: 0.1,
            annulus_margin: 0.3,
            radial_center: RadialCenterConfig::default(),
            sampling: AnnulusSamplingConfig::default(),
            min_alignment: 0.3,
            ray_count: 96,
            radial_search_inner: 0.6,
            radial_search_outer: 1.45,
            normal_search_half_width: 6.0,
            min_inlier_coverage: 0.6,
            max_center_shift_fraction: 0.4,
            max_axis_ratio: 1.8,
        }
    }
}

#[inline]
fn observation_point(observation: &EdgeObservation) -> PixelCoord {
    observation.point
}

#[inline]
fn observation_score(observation: &EdgeObservation) -> Scalar {
    observation.score
}

#[inline]
fn observation_offset(observation: &EdgeObservation) -> Scalar {
    observation.offset
}

#[inline]
fn observation_signed_projection(observation: &EdgeObservation) -> Scalar {
    observation.signed_projection
}

#[inline]
fn observation_sector(observation: &EdgeObservation) -> usize {
    observation.sector
}

fn observation_residual(ellipse: &Ellipse, observation: &EdgeObservation) -> Option<Scalar> {
    let (residual, _) = ellipse_residual_and_jacobian(ellipse, observation.point)?;
    Some(residual.abs() as Scalar)
}

fn collect_radial_candidates(
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    center: PixelCoord,
    seed_radius: Scalar,
    config: &EllipseRefineConfig,
) -> Vec<Vec<EdgeObservation>> {
    let ray_count = config.ray_count.max(16);
    let start = config.radial_search_inner * seed_radius;
    let stop = config.radial_search_outer * seed_radius;
    let sigma = 0.35 * seed_radius.max(1.0);
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
            seed_radius,
            sigma,
            0.0,
            DEFAULT_MAX_EDGE_CANDIDATES,
            DEFAULT_PEAK_MIN_SEPARATION_PX,
            |point, score, offset, signed_projection, sector| {
                Some(EdgeObservation {
                    point,
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
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    ellipse: &Ellipse,
    config: &EllipseRefineConfig,
    expected_sign: Scalar,
) -> Vec<Vec<EdgeObservation>> {
    let ray_count = config.ray_count.max(16);
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let sigma = (config.normal_search_half_width * 0.5).max(1.0);
    let mut sectors = vec![Vec::new(); ray_count];

    for (sector, slot) in sectors.iter_mut().enumerate() {
        let theta = 2.0 * PI * sector as Scalar / ray_count as Scalar;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let ex = ellipse.semi_major * cos_t;
        let ey = ellipse.semi_minor * sin_t;
        let base = PixelCoord::new(
            ellipse.center.x + ex * cos_a - ey * sin_a,
            ellipse.center.y + ex * sin_a + ey * cos_a,
        );

        let nx_local = ellipse.semi_minor * cos_t;
        let ny_local = ellipse.semi_major * sin_t;
        let n_mag = (nx_local * nx_local + ny_local * ny_local).sqrt();
        if n_mag < 1e-8 {
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
            |point, score, offset, signed_projection, sector| {
                Some(EdgeObservation {
                    point,
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

fn select_observations_by_ellipse(
    sector_candidates: &[Vec<EdgeObservation>],
    ellipse: &Ellipse,
) -> Vec<EdgeObservation> {
    select_best_consistent_candidates(
        sector_candidates,
        observation_score,
        observation_offset,
        |observation| observation_residual(ellipse, observation),
    )
}

fn better_hypothesis(candidate: &HypothesisFit, best: &HypothesisFit, seed_radius: Scalar) -> bool {
    compare_fit_quality(&candidate.fit, &best.fit, seed_radius).is_gt()
}

#[allow(clippy::too_many_arguments)]
fn refine_hypothesis(
    sector_candidates: &[Vec<EdgeObservation>],
    observations: Vec<EdgeObservation>,
    fallback_center: PixelCoord,
    seed_center: PixelCoord,
    seed_radius: Scalar,
    config: &EllipseRefineConfig,
    width: usize,
    height: usize,
) -> Option<HypothesisFit> {
    let min_count = min_inlier_count(config.ray_count.max(16));
    if observations.len() < min_count {
        return None;
    }

    let guess = weighted_covariance_guess(
        &observations,
        fallback_center,
        seed_radius,
        observation_point,
        observation_score,
    );
    let mut fit = refine_from_observations(
        &observations,
        guess,
        seed_center,
        seed_radius,
        config,
        width,
        height,
        observation_point,
        observation_score,
        observation_sector,
    )?;
    let mut selected = observations;

    let reassigned = select_observations_by_ellipse(sector_candidates, &fit.ellipse);
    if reassigned.len() >= min_count {
        if let Some(refined) = refine_from_observations(
            &reassigned,
            fit.ellipse,
            seed_center,
            seed_radius,
            config,
            width,
            height,
            observation_point,
            observation_score,
            observation_sector,
        ) {
            selected = reassigned;
            fit = refined;
        }
    }

    Some(HypothesisFit {
        fit,
        observations: selected,
    })
}

fn initial_fit_from_candidates(
    sector_candidates: &[Vec<EdgeObservation>],
    fallback_center: PixelCoord,
    seed_center: PixelCoord,
    seed_radius: Scalar,
    config: &EllipseRefineConfig,
    width: usize,
    height: usize,
) -> Option<HypothesisFit> {
    let hypotheses = best_hypotheses(sector_candidates, observation_score, observation_offset);
    let mut best = None;
    for observations in hypotheses {
        let Some(candidate) = refine_hypothesis(
            sector_candidates,
            observations,
            fallback_center,
            seed_center,
            seed_radius,
            config,
            width,
            height,
        ) else {
            continue;
        };
        let replace = match best.as_ref() {
            None => true,
            Some(current) => better_hypothesis(&candidate, current, seed_radius),
        };
        if replace {
            best = Some(candidate);
        }
    }
    best
}

/// Iteratively refine an ellipse hypothesis from local gradient evidence.
pub fn refine_ellipse(
    gradient: &GradientField,
    initial: &Ellipse,
    config: &EllipseRefineConfig,
) -> Result<RefinementResult<Ellipse>> {
    config.validate()?;
    let seed_radius = initial.mean_radius().max(1.0);
    let seed_center = initial.center;
    let width = gradient.width();
    let height = gradient.height();
    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let min_count = min_inlier_count(config.ray_count.max(16));

    let radial_center =
        radial_center_refine_from_gradient(gradient, seed_center, &config.radial_center)?;
    let stabilized_center = match radial_center.status {
        RefinementStatus::Converged => {
            clamp_center_shift(seed_center, radial_center.hypothesis, 0.25 * seed_radius)
        }
        _ => seed_center,
    };

    let initial_circle = Ellipse::new(stabilized_center, seed_radius, seed_radius, 0.0);
    let radial_candidates =
        collect_radial_candidates(gx_view, gy_view, stabilized_center, seed_radius, config);
    let radial_count: usize = radial_candidates.iter().map(Vec::len).sum();
    if radial_count < min_count {
        return Ok(RefinementResult {
            hypothesis: initial_circle,
            status: RefinementStatus::Degenerate,
            residual: 0.0,
            iterations: 0,
        });
    }

    let Some(mut fit) = initial_fit_from_candidates(
        &radial_candidates,
        stabilized_center,
        seed_center,
        seed_radius,
        config,
        width,
        height,
    ) else {
        return Ok(RefinementResult {
            hypothesis: initial_circle,
            status: RefinementStatus::Degenerate,
            residual: 0.0,
            iterations: 0,
        });
    };
    let expected_sign = infer_expected_sign(
        &fit.observations,
        observation_score,
        observation_signed_projection,
    );

    if !guard_ellipse(
        &fit.fit.ellipse,
        seed_center,
        seed_radius,
        config,
        width,
        height,
    ) {
        return Ok(RefinementResult {
            hypothesis: initial_circle,
            status: RefinementStatus::Degenerate,
            residual: fit.fit.evaluation.objective,
            iterations: fit.fit.inner_iterations,
        });
    }

    let mut iterations = 1usize;
    let mut status = if fit.fit.converged {
        RefinementStatus::Converged
    } else {
        RefinementStatus::MaxIterations
    };

    for _ in 0..config.max_iterations.saturating_sub(1) {
        let mut normal_candidates =
            collect_normal_candidates(gx_view, gy_view, &fit.fit.ellipse, config, expected_sign);
        let mut observations = select_observations_by_ellipse(&normal_candidates, &fit.fit.ellipse);
        if observations.len() < min_count && expected_sign.abs() > 0.5 {
            normal_candidates =
                collect_normal_candidates(gx_view, gy_view, &fit.fit.ellipse, config, 0.0);
            observations = select_observations_by_ellipse(&normal_candidates, &fit.fit.ellipse);
        }
        if observations.len() < min_count {
            break;
        }

        let Some(candidate_fit) = refine_from_observations(
            &observations,
            fit.fit.ellipse,
            seed_center,
            seed_radius,
            config,
            width,
            height,
            observation_point,
            observation_score,
            observation_sector,
        ) else {
            break;
        };
        let reassigned = select_observations_by_ellipse(&normal_candidates, &candidate_fit.ellipse);
        let candidate = if reassigned.len() >= min_count {
            if let Some(refined) = refine_from_observations(
                &reassigned,
                candidate_fit.ellipse,
                seed_center,
                seed_radius,
                config,
                width,
                height,
                observation_point,
                observation_score,
                observation_sector,
            ) {
                HypothesisFit {
                    fit: refined,
                    observations: reassigned,
                }
            } else {
                HypothesisFit {
                    fit: candidate_fit,
                    observations,
                }
            }
        } else {
            HypothesisFit {
                fit: candidate_fit,
                observations,
            }
        };

        if !compare_fit_quality(&candidate.fit, &fit.fit, seed_radius).is_gt() {
            status = RefinementStatus::Converged;
            break;
        }

        let center_shift = ((candidate.fit.ellipse.center.x - fit.fit.ellipse.center.x).powi(2)
            + (candidate.fit.ellipse.center.y - fit.fit.ellipse.center.y).powi(2))
        .sqrt();
        let axis_shift = (candidate.fit.ellipse.semi_major - fit.fit.ellipse.semi_major)
            .abs()
            .max((candidate.fit.ellipse.semi_minor - fit.fit.ellipse.semi_minor).abs());

        fit = candidate;
        iterations += 1;
        if fit.fit.converged
            || (center_shift < config.convergence_tol && axis_shift < config.convergence_tol)
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

    Ok(RefinementResult {
        hypothesis: fit.fit.ellipse,
        status,
        residual: fit.fit.evaluation.objective,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::{ImageView, OwnedImage};

    fn make_ellipse_image(
        size: usize,
        cx: f32,
        cy: f32,
        semi_major: f32,
        semi_minor: f32,
        angle: f32,
    ) -> Vec<u8> {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let mut data = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;
                let ellipse_r = (lx / semi_major).powi(2) + (ly / semi_minor).powi(2);
                let inside = if ellipse_r <= 1.0 { 210.0 } else { 20.0 };
                let texture = 10.0 * ((0.11 * x as f32).sin() + (0.07 * y as f32).cos());
                data[y * size + x] = (inside + texture).clamp(0.0, 255.0);
            }
        }

        let mut image = OwnedImage::from_vec(data, size, size).unwrap();
        crate::core::blur::gaussian_blur_inplace(&mut image, 1.2);
        image
            .data()
            .iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect()
    }

    fn make_challenging_ellipse(size: usize) -> Vec<u8> {
        let cx = 64.0;
        let cy = 62.0;
        let semi_major = 24.0;
        let semi_minor = 17.0;
        let angle = 0.45f32;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let mut data = vec![25.0f32; size * size];

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;
                let ellipse_r = (lx / semi_major).powi(2) + (ly / semi_minor).powi(2);
                if ellipse_r <= 1.0 {
                    data[y * size + x] = 210.0;
                }
                data[y * size + x] += 12.0 * ((0.17 * x as f32).sin() * (0.09 * y as f32).cos());
            }
        }

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx > 6.0 && dy > 2.0 && (dx * dx + dy * dy).sqrt() < 35.0 {
                    data[y * size + x] = 180.0;
                }
            }
        }

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - 92.0;
                let dy = y as f32 - 34.0;
                if (dx * dx + dy * dy).sqrt() < 5.0 {
                    data[y * size + x] = 255.0;
                }
            }
        }

        let mut image = OwnedImage::from_vec(data, size, size).unwrap();
        crate::core::blur::gaussian_blur_inplace(&mut image, 1.0);
        image
            .data()
            .iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect()
    }

    fn concentric_fixture_params() -> (PixelCoord, f32, f32, f32, f32, f32) {
        (PixelCoord::new(64.0, 63.0), 26.0, 18.0, 15.0, 10.4, 0.42)
    }

    fn make_concentric_boundary_ellipse(size: usize, partial_inner: bool) -> Vec<u8> {
        let (center, outer_a, outer_b, inner_a, inner_b, angle) = concentric_fixture_params();
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let mut data = vec![26.0f32; size * size];

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center.x;
                let dy = y as f32 - center.y;
                let lx = dx * cos_a + dy * sin_a;
                let ly = -dx * sin_a + dy * cos_a;
                let outer_r = (lx / outer_a).powi(2) + (ly / outer_b).powi(2);
                let inner_r = (lx / inner_a).powi(2) + (ly / inner_b).powi(2);
                let mut value = if outer_r <= 1.0 { 222.0 } else { 24.0 };
                if inner_r <= 1.0 {
                    let theta = ly.atan2(lx);
                    let keep_inner = !partial_inner || (-2.25..=0.85).contains(&theta);
                    if keep_inner {
                        value = 2.0;
                    }
                }
                value += 10.0 * ((0.14 * x as f32).sin() + (0.08 * y as f32).cos());
                data[y * size + x] = value.clamp(0.0, 255.0);
            }
        }

        let mut image = OwnedImage::from_vec(data, size, size).unwrap();
        crate::core::blur::gaussian_blur_inplace(&mut image, 1.0);
        image
            .data()
            .iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect()
    }

    #[test]
    fn refine_tilted_ellipse_from_circle_seed() {
        let size = 128;
        let true_center = PixelCoord::new(64.0, 62.0);
        let true_a = 24.0;
        let true_b = 17.0;
        let true_angle = 0.35;
        let data = make_ellipse_image(
            size,
            true_center.x,
            true_center.y,
            true_a,
            true_b,
            true_angle,
        );
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let initial = Ellipse::new(PixelCoord::new(67.0, 60.0), 21.0, 21.0, 0.0);
        let result = refine_ellipse(&grad, &initial, &EllipseRefineConfig::default()).unwrap();
        assert_ne!(result.status, RefinementStatus::Degenerate);
        let ellipse = result.hypothesis;
        let center_err = ((ellipse.center.x - true_center.x).powi(2)
            + (ellipse.center.y - true_center.y).powi(2))
        .sqrt();
        assert!(center_err < 2.0, "center error too large: {center_err}");
        assert!(
            (ellipse.semi_major - true_a).abs() < 2.5,
            "semi-major mismatch: {}",
            ellipse.semi_major
        );
        assert!(
            (ellipse.semi_minor - true_b).abs() < 2.5,
            "semi-minor mismatch: {}",
            ellipse.semi_minor
        );
    }

    #[test]
    fn refine_resists_small_distractor_collapse() {
        let size = 128;
        let data = make_challenging_ellipse(size);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let grad = sobel_gradient(&image).unwrap();
        let initial = Ellipse::new(PixelCoord::new(66.0, 61.0), 22.0, 22.0, 0.0);
        let result = refine_ellipse(&grad, &initial, &EllipseRefineConfig::default()).unwrap();
        assert_ne!(result.status, RefinementStatus::Degenerate);
        let ellipse = result.hypothesis;
        assert!(
            ellipse.semi_minor > 13.0,
            "collapsed semi-minor: {}",
            ellipse.semi_minor
        );
        let center_err =
            ((ellipse.center.x - 64.0).powi(2) + (ellipse.center.y - 62.0).powi(2)).sqrt();
        assert!(center_err < 4.0, "center error too large: {center_err}");
    }

    #[test]
    fn default_refiner_prefers_outer_boundary_when_inner_distractor_is_partial() {
        let size = 128;
        let data = make_concentric_boundary_ellipse(size, true);
        let image = ImageView::from_slice(&data, size, size).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let (center, outer_a, outer_b, inner_a, _inner_b, angle) = concentric_fixture_params();
        let initial = Ellipse::new(
            PixelCoord::new(center.x + 2.5, center.y - 2.0),
            23.0,
            23.0,
            0.0,
        );
        let config = EllipseRefineConfig {
            radial_search_inner: 0.3,
            ..EllipseRefineConfig::default()
        };

        let result = refine_ellipse(&gradient, &initial, &config).unwrap();
        assert_ne!(result.status, RefinementStatus::Degenerate);
        assert!((result.hypothesis.semi_major - outer_a).abs() < 3.0);
        assert!((result.hypothesis.semi_minor - outer_b).abs() < 3.0);
        assert!(result.hypothesis.semi_major > inner_a + 6.0);
        assert!(angle_error_deg(result.hypothesis.angle, angle) < 8.0);
    }

    #[test]
    fn degenerate_on_empty() {
        let data = vec![0u8; 100 * 100];
        let image = ImageView::from_slice(&data, 100, 100).unwrap();
        let grad = sobel_gradient(&image).unwrap();

        let initial = Ellipse::new(PixelCoord::new(50.0, 50.0), 15.0, 15.0, 0.0);
        let result = refine_ellipse(&grad, &initial, &EllipseRefineConfig::default()).unwrap();
        assert_eq!(result.status, RefinementStatus::Degenerate);
    }

    fn angle_error_deg(predicted: f32, truth: f32) -> f32 {
        let mut delta = (predicted - truth).to_degrees();
        delta = (delta + 90.0).rem_euclid(180.0) - 90.0;
        delta.abs()
    }
}
