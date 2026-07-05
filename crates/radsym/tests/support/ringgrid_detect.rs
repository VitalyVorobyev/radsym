#![allow(dead_code)]

//! Ring-grid detection: locate each ring's outer (and inner) ellipse via an RSD
//! center vote followed by local ellipse refinement.
//!
//! Shared single source of truth for `ringgrid_regression.rs` (synthetic raster)
//! and `examples/perf_export.rs` (the real `testdata/ringgrid.png` overlay), so
//! the tested pipeline and the published overlay never drift.

use std::hint::black_box;
use std::time::{Duration, Instant};

use radsym::core::gradient::{GradientField, sobel_gradient};
use radsym::core::nms::NmsConfig;
use radsym::propose::rsd::RsdConfig;
use radsym::{
    Ellipse, EllipseRefineAdvanced, EllipseRefineConfig, OwnedImage, PixelCoord, Polarity,
    Proposal, extract_proposals, refine_ellipse, rsd_response, suppress_proposals_by_distance,
};

/// One detected ring: the outer annulus ellipse and, when recoverable, the
/// inner ellipse (both in image coordinates).
#[derive(Clone, Copy, Debug)]
pub struct RingDetection {
    pub outer: Ellipse,
    pub inner: Option<Ellipse>,
    pub seed: PixelCoord,
    pub outer_residual: f32,
}

/// Wall-clock cost of each stage of [`detect_rings`], for the performance page.
#[derive(Clone, Copy, Debug)]
pub struct RingStageDurations {
    pub gradient: Duration,
    pub voting: Duration,
    pub extract: Duration,
    pub score: Duration,
    pub refine: Duration,
}

/// Geometric mean-radius band for the RSD voting pass.
pub fn build_radius_band(
    base_radius: f32,
    start_scale: f32,
    stop_scale: f32,
    steps: usize,
) -> Vec<u32> {
    let start = (base_radius * start_scale).round().max(4.0) as u32;
    let stop = (base_radius * stop_scale).round().max(start as f32 + 1.0) as u32;
    if steps <= 1 || start == stop {
        return vec![start];
    }
    let mut radii = (0..steps)
        .map(|index| {
            let t = index as f32 / (steps - 1) as f32;
            (start as f32 + t * (stop - start) as f32).round() as u32
        })
        .collect::<Vec<_>>();
    radii.sort_unstable();
    radii.dedup();
    radii
}

fn outer_rsd_config(outer_hint: f32) -> RsdConfig {
    let mut rsd_config = RsdConfig::default();
    rsd_config.radii = build_radius_band(outer_hint, 0.8, 1.16, 5);
    rsd_config.gradient_threshold = 2.0;
    rsd_config.polarity = Polarity::Dark;
    rsd_config.smoothing_factor = 0.5;
    rsd_config
}

pub fn outer_ellipse_config() -> EllipseRefineConfig {
    let mut advanced = EllipseRefineAdvanced::default();
    advanced.ray_count = 96;
    advanced.radial_search_inner = 0.60;
    advanced.radial_search_outer = 1.45;
    advanced.normal_search_half_width = 6.0;
    advanced.min_inlier_coverage = 0.60;
    let mut config = EllipseRefineConfig::default();
    config.max_iterations = 5;
    config.convergence_tol = 0.05;
    config.max_center_shift_fraction = 0.40;
    config.max_axis_ratio = 1.80;
    config.advanced = advanced;
    config
}

pub fn inner_ellipse_config() -> EllipseRefineConfig {
    let mut advanced = EllipseRefineAdvanced::default();
    advanced.ray_count = 96;
    advanced.radial_search_inner = 0.75;
    advanced.radial_search_outer = 1.20;
    advanced.normal_search_half_width = 4.0;
    advanced.min_inlier_coverage = 0.55;
    let mut config = EllipseRefineConfig::default();
    config.max_iterations = 5;
    config.convergence_tol = 0.05;
    config.max_center_shift_fraction = 0.25;
    config.max_axis_ratio = 1.80;
    config.advanced = advanced;
    config
}

/// RSD center-vote proposals for the ring outer annuli, one per ring.
pub fn detect_outer_candidates(
    gradient: &GradientField,
    outer_hint: f32,
    max_candidates: usize,
) -> Vec<Proposal> {
    let response = rsd_response(gradient, &outer_rsd_config(outer_hint)).unwrap();
    let mut nms_config = NmsConfig::default();
    nms_config.radius = (0.55 * outer_hint).round().max(6.0) as usize;
    nms_config.threshold = 0.01;
    nms_config.max_detections = 256;
    let proposals = extract_proposals(&response, &nms_config, Polarity::Dark);
    suppress_proposals_by_distance(&proposals, 1.25 * outer_hint, max_candidates)
}

fn plausible_outer(outer: &Ellipse, outer_hint: f32) -> bool {
    let mean_radius = 0.5 * (outer.semi_major + outer.semi_minor);
    (0.6 * outer_hint..=1.6 * outer_hint).contains(&mean_radius)
}

fn plausible_inner(inner: &Ellipse, outer: &Ellipse, outer_hint: f32) -> bool {
    let mean_radius = 0.5 * (inner.semi_major + inner.semi_minor);
    let outer_mean = 0.5 * (outer.semi_major + outer.semi_minor);
    mean_radius > 0.2 * outer_hint && mean_radius < 0.9 * outer_mean
}

/// Detect ring outer (+inner) ellipses on a precomputed gradient.
///
/// `outer_hint` is the expected outer mean radius; `inner_ratio` the expected
/// inner/outer radius ratio. Candidates whose refined outer ellipse is
/// implausibly sized are dropped so the overlay traces real rings only.
pub fn detect_rings(
    gradient: &GradientField,
    outer_hint: f32,
    inner_ratio: f32,
    max_rings: usize,
) -> Vec<RingDetection> {
    let outer_config = outer_ellipse_config();
    let inner_config = inner_ellipse_config();
    detect_outer_candidates(gradient, outer_hint, max_rings)
        .iter()
        .filter_map(|proposal| {
            let refined = refine_ellipse(
                gradient,
                &Ellipse::new(proposal.seed.position, outer_hint, outer_hint, 0.0),
                &outer_config,
            )
            .ok()?;
            let outer = refined.hypothesis;
            if !plausible_outer(&outer, outer_hint) {
                return None;
            }
            let inner_seed = Ellipse::new(
                outer.center,
                outer.semi_major * inner_ratio,
                outer.semi_minor * inner_ratio,
                outer.angle,
            );
            let inner = refine_ellipse(gradient, &inner_seed, &inner_config)
                .ok()
                .map(|r| r.hypothesis)
                .filter(|inner| plausible_inner(inner, &outer, outer_hint));
            Some(RingDetection {
                outer,
                inner,
                seed: proposal.seed.position,
                outer_residual: refined.residual,
            })
        })
        .collect()
}

/// Convenience wrapper: compute the gradient and detect rings on an image.
pub fn detect_rings_in_image(
    image: &OwnedImage<u8>,
    outer_hint: f32,
    inner_ratio: f32,
    max_rings: usize,
) -> Vec<RingDetection> {
    let gradient = sobel_gradient(&image.view()).unwrap();
    detect_rings(&gradient, outer_hint, inner_ratio, max_rings)
}

/// Time the [`detect_rings`] stages for the performance page. `refine` is the
/// outer ellipse fit; `score` is the inner ellipse fit.
pub fn time_rings_once(
    image: &OwnedImage<u8>,
    outer_hint: f32,
    inner_ratio: f32,
    max_rings: usize,
) -> RingStageDurations {
    let t = Instant::now();
    let gradient = sobel_gradient(&image.view()).unwrap();
    let gradient_dur = t.elapsed();

    let t = Instant::now();
    let response = rsd_response(black_box(&gradient), &outer_rsd_config(outer_hint)).unwrap();
    let voting_dur = t.elapsed();

    let t = Instant::now();
    let mut nms_config = NmsConfig::default();
    nms_config.radius = (0.55 * outer_hint).round().max(6.0) as usize;
    nms_config.threshold = 0.01;
    nms_config.max_detections = 256;
    let proposals = extract_proposals(black_box(&response), &nms_config, Polarity::Dark);
    let proposals = suppress_proposals_by_distance(&proposals, 1.25 * outer_hint, max_rings);
    let extract_dur = t.elapsed();

    let outer_config = outer_ellipse_config();
    let inner_config = inner_ellipse_config();
    let mut refine_dur = Duration::ZERO;
    let mut score_dur = Duration::ZERO;
    for proposal in &proposals {
        let t = Instant::now();
        let refined = refine_ellipse(
            black_box(&gradient),
            &Ellipse::new(proposal.seed.position, outer_hint, outer_hint, 0.0),
            &outer_config,
        );
        refine_dur += t.elapsed();
        if let Ok(refined) = refined {
            let outer = refined.hypothesis;
            let inner_seed = Ellipse::new(
                outer.center,
                outer.semi_major * inner_ratio,
                outer.semi_minor * inner_ratio,
                outer.angle,
            );
            let t = Instant::now();
            let _ = black_box(refine_ellipse(
                black_box(&gradient),
                &inner_seed,
                &inner_config,
            ));
            score_dur += t.elapsed();
        }
    }

    RingStageDurations {
        gradient: gradient_dur,
        voting: voting_dur,
        extract: extract_dur,
        score: score_dur,
        refine: refine_dur,
    }
}
