//! Homography-aware proposal generation and reranking.

use std::cmp::Ordering;
use std::f32::consts::PI;

use nalgebra::Vector2;

use crate::core::blur::gaussian_blur_inplace;
use crate::core::coords::PixelCoord;
use crate::core::error::Result;
use crate::core::geometry::{Circle, Ellipse};
use crate::core::gradient::GradientField;
use crate::core::homography::{rectified_circle_to_image_ellipse, Homography, RectifiedGrid};
use crate::core::image_view::{ImageView, OwnedImage};
use crate::core::nms::{non_maximum_suppression, NmsConfig};
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;
use crate::propose::frst::FrstConfig;

use super::seed::{Proposal, ProposalSource, SeedPoint};

const RADIAL_SAMPLE_STEP: Scalar = 1.0;

/// Homography-aware FRST response on a rectified grid.
#[derive(Debug, Clone)]
pub struct RectifiedResponseMap {
    response: OwnedImage<Scalar>,
    scale_hints: OwnedImage<Scalar>,
    grid: RectifiedGrid,
    source: ProposalSource,
}

impl RectifiedResponseMap {
    /// Create a rectified response map from a response image and scale hints.
    pub fn new(
        response: OwnedImage<Scalar>,
        scale_hints: OwnedImage<Scalar>,
        grid: RectifiedGrid,
        source: ProposalSource,
    ) -> Self {
        Self {
            response,
            scale_hints,
            grid,
            source,
        }
    }

    /// Borrowed response image view.
    pub fn view(&self) -> ImageView<'_, Scalar> {
        self.response.view()
    }

    /// Underlying response image.
    pub fn response(&self) -> &OwnedImage<Scalar> {
        &self.response
    }

    /// Per-pixel rectified radius hints.
    pub fn scale_hints(&self) -> &OwnedImage<Scalar> {
        &self.scale_hints
    }

    /// Rectified grid metadata.
    pub fn grid(&self) -> RectifiedGrid {
        self.grid
    }

    /// Source algorithm.
    pub fn source(&self) -> ProposalSource {
        self.source
    }
}

/// Proposal extracted from a rectified response map.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HomographyProposal {
    /// Peak location in the rectified frame.
    pub rectified_seed: SeedPoint,
    /// Approximate rectified circle hypothesis inferred from the response peak.
    pub rectified_circle_hint: Option<Circle>,
    /// Approximate image-space ellipse obtained from the rectified circle hint.
    pub image_ellipse_hint: Option<Ellipse>,
    /// Which polarity mode produced this proposal.
    pub polarity: Polarity,
    /// Which algorithm generated this proposal.
    pub source: ProposalSource,
}

/// Output of homography-aware proposal reranking.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RerankedProposal {
    /// Original image-space proposal.
    pub proposal: Proposal,
    /// Projectively consistent image-space ellipse hint.
    pub image_ellipse_hint: Option<Ellipse>,
    /// Projectively fitted rectified circle hint.
    pub rectified_circle_hint: Option<Circle>,
    /// Trimmed mean edge strength from the image-space acquisition pass.
    pub rectified_edge_score: Scalar,
    /// Rectified angular coverage in `[0, 1]`.
    pub rectified_coverage: Scalar,
    /// Gaussian size prior around the swept image radius.
    pub size_prior: Scalar,
    /// Mild Gaussian image-center prior in `[0, 1]`.
    pub center_prior: Scalar,
    /// Final reranking score.
    pub total_score: Scalar,
}

/// Configuration for homography-aware proposal reranking.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HomographyRerankConfig {
    /// Minimum image-space radius to evaluate when no scale hint is available.
    pub min_radius: Scalar,
    /// Maximum image-space radius to evaluate. Values <= `min_radius` enable automatic sizing.
    pub max_radius: Scalar,
    /// Step size for the coarse radius sweep in pixels.
    pub radius_step: Scalar,
    /// Number of image-space rays used during the edge acquisition pass.
    pub ray_count: usize,
    /// Inner radius factor for the initial radial search.
    pub radial_search_inner: Scalar,
    /// Outer radius factor for the initial radial search.
    pub radial_search_outer: Scalar,
    /// Width of the Gaussian size prior around the swept radius.
    pub size_prior_sigma: Scalar,
    /// Sigma for the image-center prior as a fraction of `min(width, height)`.
    pub center_prior_sigma_fraction: Scalar,
}

impl Default for HomographyRerankConfig {
    fn default() -> Self {
        Self {
            min_radius: 6.0,
            max_radius: 0.0,
            radius_step: 2.0,
            ray_count: 64,
            radial_search_inner: 0.6,
            radial_search_outer: 1.45,
            size_prior_sigma: 0.22,
            center_prior_sigma_fraction: 0.45,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EdgeObservation {
    point: PixelCoord,
    score: Scalar,
    sector: usize,
}

fn sample_gradient(
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    x: Scalar,
    y: Scalar,
) -> Option<(Scalar, Scalar)> {
    Some((gx_view.sample(x, y)?, gy_view.sample(x, y)?))
}

#[allow(clippy::too_many_arguments)]
fn strongest_edge_along_ray(
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    origin: PixelCoord,
    dir_x: Scalar,
    dir_y: Scalar,
    start: Scalar,
    stop: Scalar,
    bias_center: Scalar,
    bias_sigma: Scalar,
    sector: usize,
) -> Option<EdgeObservation> {
    let samples = (((stop - start) / RADIAL_SAMPLE_STEP).ceil() as usize).max(2) + 1;
    let mut radii = Vec::with_capacity(samples);
    let mut scores = Vec::with_capacity(samples);

    for i in 0..samples {
        let radius = start + i as Scalar * RADIAL_SAMPLE_STEP;
        if radius > stop + 1e-6 {
            break;
        }
        let x = origin.x + dir_x * radius;
        let y = origin.y + dir_y * radius;
        let score = if let Some((gx, gy)) = sample_gradient(gx_view, gy_view, x, y) {
            let projected = (gx * dir_x + gy * dir_y).abs();
            if bias_sigma > 1e-6 {
                let z = (radius - bias_center) / bias_sigma;
                projected * (-0.5 * z * z).exp()
            } else {
                projected
            }
        } else {
            0.0
        };
        radii.push(radius);
        scores.push(score);
    }

    let mut best = None;
    let mut best_score = 0.0;
    for i in 0..scores.len() {
        let prev = if i > 0 { scores[i - 1] } else { scores[i] };
        let next = if i + 1 < scores.len() {
            scores[i + 1]
        } else {
            scores[i]
        };
        let smooth = 0.25 * prev + 0.5 * scores[i] + 0.25 * next;
        let is_peak = (i == 0 || smooth >= prev) && (i + 1 == scores.len() || smooth >= next);
        if is_peak && smooth > best_score {
            best_score = smooth;
            best = Some(i);
        }
    }

    let index = best?;
    if best_score <= 1e-4 {
        return None;
    }

    let radius = radii[index];
    Some(EdgeObservation {
        point: PixelCoord::new(origin.x + dir_x * radius, origin.y + dir_y * radius),
        score: best_score,
        sector,
    })
}

fn collect_radial_observations(
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    center: PixelCoord,
    radius: Scalar,
    config: &HomographyRerankConfig,
) -> Vec<EdgeObservation> {
    let ray_count = config.ray_count.max(16);
    let start = config.radial_search_inner * radius;
    let stop = config.radial_search_outer * radius;
    let sigma = 0.35 * radius.max(1.0);
    let mut observations = Vec::with_capacity(ray_count);

    for sector in 0..ray_count {
        let theta = 2.0 * PI * sector as Scalar / ray_count as Scalar;
        let dir_x = theta.cos();
        let dir_y = theta.sin();
        if let Some(obs) = strongest_edge_along_ray(
            gx_view, gy_view, center, dir_x, dir_y, start, stop, radius, sigma, sector,
        ) {
            observations.push(obs);
        }
    }

    observations
}

fn circle_support_score(
    gradient: &GradientField,
    center: PixelCoord,
    radius: Scalar,
    ray_count: usize,
) -> Scalar {
    if radius <= 1.0 || ray_count == 0 {
        return 0.0;
    }

    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let mut aligned = 0usize;
    let mut sampled = 0usize;
    let mut alignment_sum = 0.0;

    for sector in 0..ray_count {
        let theta = 2.0 * PI * sector as Scalar / ray_count as Scalar;
        let dir_x = theta.cos();
        let dir_y = theta.sin();
        let x = center.x + radius * dir_x;
        let y = center.y + radius * dir_y;
        let Some((gx, gy)) = sample_gradient(gx_view, gy_view, x, y) else {
            continue;
        };
        let mag = (gx * gx + gy * gy).sqrt();
        if mag <= 1e-6 {
            continue;
        }
        let alignment = ((gx * dir_x + gy * dir_y) / mag).abs();
        sampled += 1;
        alignment_sum += alignment;
        if alignment > 0.45 {
            aligned += 1;
        }
    }

    if aligned == 0 || sampled == 0 {
        return 0.0;
    }
    let coverage = aligned as Scalar / ray_count as Scalar;
    let mean_alignment = alignment_sum / sampled as Scalar;
    (0.7 * mean_alignment + 0.3 * coverage).clamp(0.0, 1.0)
}

fn fit_circle_from_points(points: &[(PixelCoord, Scalar)]) -> Option<Circle> {
    let (coords, weights): (Vec<_>, Vec<_>) = points.iter().copied().unzip();
    crate::core::circle_fit::fit_circle_weighted(&coords, &weights)
}

fn fit_rectified_circle(
    homography: &Homography,
    observations: &[EdgeObservation],
    ray_count: usize,
) -> Option<(Circle, Scalar, Scalar)> {
    let mut mapped = Vec::with_capacity(observations.len());
    let mut sectors = vec![false; ray_count];
    let mut edge_sum = 0.0;

    for obs in observations {
        let rectified = homography.map_image_to_rectified(obs.point)?;
        mapped.push((rectified, obs.score));
        sectors[obs.sector] = true;
        edge_sum += obs.score;
    }

    let circle = fit_circle_from_points(&mapped)?;
    let coverage = sectors.iter().filter(|&&used| used).count() as Scalar / ray_count as Scalar;
    let edge_score = edge_sum / observations.len().max(1) as Scalar;
    Some((circle, edge_score, coverage))
}

fn auto_radius_bounds(
    gradient: &GradientField,
    config: &HomographyRerankConfig,
) -> (Scalar, Scalar) {
    let min_radius = config.min_radius.max(3.0);
    if config.max_radius > min_radius {
        (min_radius, config.max_radius)
    } else {
        let max_auto = 0.35 * gradient.width().min(gradient.height()) as Scalar;
        (min_radius, max_auto.max(min_radius + 4.0))
    }
}

fn sweep_radius(
    gradient: &GradientField,
    center: PixelCoord,
    radius_hint: Option<Scalar>,
    config: &HomographyRerankConfig,
) -> Circle {
    let (global_min, global_max) = auto_radius_bounds(gradient, config);
    let (start, stop) = if let Some(hint) = radius_hint {
        let start = (0.7 * hint).max(global_min);
        let stop = (1.35 * hint).max(start + config.radius_step.max(0.5));
        (
            start,
            stop.min(global_max.max(start + config.radius_step.max(0.5))),
        )
    } else {
        (global_min, global_max)
    };

    let mut best = Circle::new(center, start);
    let mut best_score = -1.0;
    let mut radius = start;
    while radius <= stop + 1e-6 {
        let score = circle_support_score(gradient, center, radius, config.ray_count.max(16));
        if score > best_score {
            best_score = score;
            best = Circle::new(center, radius);
        }
        radius += config.radius_step.max(0.5);
    }

    best
}

fn center_prior(
    position: PixelCoord,
    width: usize,
    height: usize,
    sigma_fraction: Scalar,
) -> Scalar {
    let image_center = PixelCoord::new(0.5 * width as Scalar, 0.5 * height as Scalar);
    let dx = position.x - image_center.x;
    let dy = position.y - image_center.y;
    let sigma = sigma_fraction.max(1e-3) * width.min(height) as Scalar;
    let z2 = (dx * dx + dy * dy) / (sigma * sigma);
    (-0.5 * z2).exp()
}

fn response_single_homography(
    gradient: &GradientField,
    homography: &Homography,
    grid: RectifiedGrid,
    radius: u32,
    config: &FrstConfig,
) -> Result<OwnedImage<Scalar>> {
    let mut orientation = OwnedImage::<Scalar>::zeros(grid.width, grid.height)?;
    let mut magnitude = OwnedImage::<Scalar>::zeros(grid.width, grid.height)?;
    let o_data = orientation.data_mut();
    let m_data = magnitude.data_mut();
    let gx_data = gradient.gx.data();
    let gy_data = gradient.gy.data();
    let w = gradient.width();
    let h = gradient.height();
    let radius = radius as Scalar;
    let thresh_sq = config.gradient_threshold * config.gradient_threshold;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let gx = gx_data[idx];
            let gy = gy_data[idx];
            let mag_sq = gx * gx + gy * gy;
            if mag_sq < thresh_sq {
                continue;
            }

            let image_point = PixelCoord::new(x as Scalar, y as Scalar);
            let Some(rectified) = homography.map_image_to_rectified(image_point) else {
                continue;
            };
            let Some(gradient_r) =
                homography.transform_gradient_image_to_rectified(image_point, Vector2::new(gx, gy))
            else {
                continue;
            };
            let mag_r = gradient_r.norm();
            if mag_r <= 1e-6 {
                continue;
            }
            let dir = gradient_r / mag_r;
            let affected = [
                (
                    config.polarity.votes_positive(),
                    1.0,
                    rectified.x + dir[0] * radius,
                    rectified.y + dir[1] * radius,
                ),
                (
                    config.polarity.votes_negative(),
                    -1.0,
                    rectified.x - dir[0] * radius,
                    rectified.y - dir[1] * radius,
                ),
            ];

            for (enabled, sign, ax, ay) in affected {
                if !enabled {
                    continue;
                }
                let rx = ax.round() as isize;
                let ry = ay.round() as isize;
                if rx < 0 || ry < 0 || rx as usize >= grid.width || ry as usize >= grid.height {
                    continue;
                }
                let ridx = ry as usize * grid.width + rx as usize;
                o_data[ridx] += sign;
                m_data[ridx] += mag_r;
            }
        }
    }

    let o_max = o_data
        .iter()
        .map(|value| value.abs())
        .fold(0.0, Scalar::max)
        .max(1.0);
    let m_max = m_data.iter().copied().fold(0.0, Scalar::max).max(1.0);
    let mut response = OwnedImage::<Scalar>::zeros(grid.width, grid.height)?;
    let response_data = response.data_mut();

    for i in 0..response_data.len() {
        let o_tilde = o_data[i] / o_max;
        let m_tilde = m_data[i] / m_max;
        response_data[i] = o_tilde.abs().powf(config.alpha) * m_tilde;
    }

    let sigma = config.smoothing_factor * radius;
    if sigma > 0.5 {
        gaussian_blur_inplace(&mut response, sigma);
    }

    Ok(response)
}

/// Compute homography-aware FRST on the caller-defined rectified grid.
pub fn frst_response_homography(
    gradient: &GradientField,
    homography: &Homography,
    grid: RectifiedGrid,
    config: &FrstConfig,
) -> Result<RectifiedResponseMap> {
    let mut response = OwnedImage::<Scalar>::zeros(grid.width, grid.height)?;
    let mut scale_hints = OwnedImage::<Scalar>::zeros(grid.width, grid.height)?;
    let mut best_per_pixel = vec![Scalar::NEG_INFINITY; grid.width * grid.height];

    for &radius in &config.radii {
        let response_single =
            response_single_homography(gradient, homography, grid, radius, config)?;
        let data = response.data_mut();
        let scale_data = scale_hints.data_mut();
        for (idx, value) in response_single.data().iter().enumerate() {
            data[idx] += *value;
            if *value > best_per_pixel[idx] {
                best_per_pixel[idx] = *value;
                scale_data[idx] = radius as Scalar;
            }
        }
    }

    Ok(RectifiedResponseMap::new(
        response,
        scale_hints,
        grid,
        ProposalSource::Frst,
    ))
}

/// Extract rectified proposals from a homography-aware response map.
pub fn extract_rectified_proposals(
    response: &RectifiedResponseMap,
    homography: &Homography,
    nms_config: &NmsConfig,
    polarity: Polarity,
) -> Vec<HomographyProposal> {
    let peaks = non_maximum_suppression(&response.view(), nms_config);
    let scale_data = response.scale_hints().data();
    let width = response.response().width();

    peaks
        .into_iter()
        .map(|peak| {
            let x = peak.position.x.round() as usize;
            let y = peak.position.y.round() as usize;
            let scale_hint = if x < width && y < response.response().height() {
                let radius = scale_data[y * width + x];
                if radius > 0.0 {
                    Some(radius)
                } else {
                    None
                }
            } else {
                None
            };

            let rectified_circle_hint =
                scale_hint.map(|radius| Circle::new(peak.position, radius.max(1.0)));
            let image_ellipse_hint = rectified_circle_hint
                .as_ref()
                .and_then(|circle| rectified_circle_to_image_ellipse(homography, circle).ok());

            HomographyProposal {
                rectified_seed: SeedPoint {
                    position: peak.position,
                    score: peak.score,
                },
                rectified_circle_hint,
                image_ellipse_hint,
                polarity,
                source: response.source(),
            }
        })
        .collect()
}

/// Rerank image-space proposals using a known image->rectified homography.
pub fn rerank_proposals_homography(
    gradient: &GradientField,
    proposals: &[Proposal],
    homography: &Homography,
    config: &HomographyRerankConfig,
) -> Vec<RerankedProposal> {
    let gx_view = gradient.gx();
    let gy_view = gradient.gy();
    let ray_count = config.ray_count.max(16);
    let mut reranked = Vec::with_capacity(proposals.len());

    for proposal in proposals {
        let radius_hint = proposal.scale_hint;
        let swept_circle = sweep_radius(gradient, proposal.seed.position, radius_hint, config);
        let observations = collect_radial_observations(
            gx_view,
            gy_view,
            swept_circle.center,
            swept_circle.radius,
            config,
        );

        let (rectified_circle_hint, edge_score, coverage) =
            if let Some((circle, edge_score, coverage)) =
                fit_rectified_circle(homography, &observations, ray_count)
            {
                (Some(circle), edge_score, coverage)
            } else {
                (None, 0.0, 0.0)
            };

        let image_ellipse_hint = rectified_circle_hint
            .as_ref()
            .and_then(|circle| rectified_circle_to_image_ellipse(homography, circle).ok());

        let size_prior = if let Some(ellipse) = image_ellipse_hint.as_ref() {
            let ratio = ellipse.mean_radius() / swept_circle.radius.max(1e-3);
            let z = (ratio - 1.0) / config.size_prior_sigma.max(1e-3);
            (-0.5 * z * z).exp()
        } else {
            0.0
        };

        let center_prior = center_prior(
            proposal.seed.position,
            gradient.width(),
            gradient.height(),
            config.center_prior_sigma_fraction,
        );
        let total_score = (edge_score * coverage * size_prior * center_prior).clamp(0.0, 1.0);

        reranked.push(RerankedProposal {
            proposal: proposal.clone(),
            image_ellipse_hint,
            rectified_circle_hint,
            rectified_edge_score: edge_score,
            rectified_coverage: coverage,
            size_prior,
            center_prior,
            total_score,
        });
    }

    reranked.sort_by(|a, b| {
        b.total_score
            .partial_cmp(&a.total_score)
            .unwrap_or(Ordering::Equal)
    });
    reranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gradient::sobel_gradient;
    use crate::core::image_view::{ImageView, OwnedImage};

    fn gradient_field_from_samples(
        width: usize,
        height: usize,
        samples: &[(usize, usize, Scalar, Scalar)],
    ) -> GradientField {
        let mut gx = OwnedImage::<Scalar>::zeros(width, height).unwrap();
        let mut gy = OwnedImage::<Scalar>::zeros(width, height).unwrap();
        for &(x, y, dx, dy) in samples {
            *gx.get_mut(x, y).unwrap() = dx;
            *gy.get_mut(x, y).unwrap() = dy;
        }
        GradientField { gx, gy }
    }

    fn make_projective_disk(size: usize, homography: &Homography, circle: Circle) -> Vec<u8> {
        let mut data = vec![25u8; size * size];
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

    #[test]
    fn homography_frst_finds_rectified_peak() {
        let homography = Homography::new([
            [1.15, 0.08, 20.0],
            [0.04, 1.05, 12.0],
            [0.0012, -0.0009, 1.0],
        ])
        .unwrap();
        let circle = Circle::new(PixelCoord::new(70.0, 60.0), 16.0);
        let image = make_projective_disk(128, &homography, circle);
        let image = ImageView::from_slice(&image, 128, 128).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let grid = RectifiedGrid::new(128, 128).unwrap();
        let response = frst_response_homography(
            &gradient,
            &homography,
            grid,
            &FrstConfig {
                radii: vec![15, 16, 17],
                polarity: Polarity::Bright,
                gradient_threshold: 1.0,
                ..FrstConfig::default()
            },
        )
        .unwrap();
        let proposals = extract_rectified_proposals(
            &response,
            &homography,
            &NmsConfig {
                radius: 5,
                threshold: 0.0,
                max_detections: 4,
            },
            Polarity::Bright,
        );
        assert!(!proposals.is_empty());
        let best = &proposals[0];
        let dx = best.rectified_seed.position.x - circle.center.x;
        let dy = best.rectified_seed.position.y - circle.center.y;
        assert!((dx * dx + dy * dy).sqrt() < 6.0);
        assert!(best.rectified_circle_hint.is_some());
        assert!(best.image_ellipse_hint.is_some());
    }

    #[test]
    fn reranking_prefers_projectively_consistent_center() {
        let homography = Homography::new([
            [1.12, 0.05, 18.0],
            [0.03, 0.98, 8.0],
            [0.0011, -0.0007, 1.0],
        ])
        .unwrap();
        let circle = Circle::new(PixelCoord::new(64.0, 64.0), 18.0);
        let image = make_projective_disk(128, &homography, circle);
        let image = ImageView::from_slice(&image, 128, 128).unwrap();
        let gradient = sobel_gradient(&image).unwrap();
        let proposals = vec![
            Proposal {
                seed: SeedPoint {
                    position: PixelCoord::new(80.0, 80.0),
                    score: 0.8,
                },
                scale_hint: Some(18.0),
                polarity: Polarity::Bright,
                source: ProposalSource::External,
            },
            Proposal {
                seed: SeedPoint {
                    position: PixelCoord::new(63.0, 63.0),
                    score: 0.7,
                },
                scale_hint: Some(18.0),
                polarity: Polarity::Bright,
                source: ProposalSource::External,
            },
        ];
        let reranked = rerank_proposals_homography(
            &gradient,
            &proposals,
            &homography,
            &HomographyRerankConfig::default(),
        );
        assert_eq!(reranked.len(), 2);
        let best = &reranked[0];
        let dx = best.proposal.seed.position.x - 63.0;
        let dy = best.proposal.seed.position.y - 63.0;
        assert!((dx * dx + dy * dy).sqrt() < 2.0);
        assert!(best.total_score > reranked[1].total_score);
    }

    #[test]
    fn circle_support_score_averages_over_sampled_rays() {
        let weak = (1.0_f32 - 0.4_f32 * 0.4_f32).sqrt();
        let gradient = gradient_field_from_samples(
            20,
            20,
            &[
                (14, 10, 1.0, 0.0),
                (14, 11, 1.0, 0.0),
                (15, 10, 1.0, 0.0),
                (15, 11, 1.0, 0.0),
                (9, 14, weak, 0.4),
                (10, 14, weak, 0.4),
                (9, 15, weak, 0.4),
                (10, 15, weak, 0.4),
            ],
        );

        let score = circle_support_score(&gradient, PixelCoord::new(10.0, 10.0), 4.0, 4);
        let expected = 0.7 * ((1.0 + 0.4) / 2.0) + 0.3 * 0.25;

        assert!(
            (score - expected).abs() < 1e-4,
            "score={score} expected={expected}"
        );
        assert!(
            score < 1.0,
            "sampling average should not saturate the score"
        );
    }
}
