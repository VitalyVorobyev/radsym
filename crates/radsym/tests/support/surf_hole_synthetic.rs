#![allow(dead_code)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use radsym::core::gradient::sobel_gradient;
use radsym::core::nms::NmsConfig;
use radsym::core::polarity::Polarity;
use radsym::core::pyramid::pyramid_level_owned;
use radsym::propose::extract::extract_proposals;
use radsym::propose::seed::Proposal;
use radsym::support::score::{
    SupportScoreBreakdown, score_circle_support_detailed, score_ellipse_support_detailed,
};
use radsym::{
    Circle, Ellipse, EllipseRefineAdvanced, EllipseRefineConfig, FrstConfig, OwnedImage,
    PixelCoord, RadSymError, Result, ScoringConfig, refine_ellipse,
};

pub const DEFAULT_PYRAMID_LEVEL: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct RenderedEllipse {
    pub center: PixelCoord,
    pub semi_major: f32,
    pub semi_minor: f32,
    pub angle: f32,
    pub value: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SyntheticSurfCase {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub target: RenderedEllipse,
    pub distractors: &'static [RenderedEllipse],
    pub blur_passes: usize,
}

#[derive(Clone, Debug)]
pub struct SyntheticSurfCandidate {
    pub proposal: Proposal,
    pub working_ellipse: Ellipse,
    pub image_ellipse: Ellipse,
    pub combined_score: f32,
    pub radius_sweep_radius: f32,
    pub final_support: SupportScoreBreakdown,
}

#[derive(Clone, Debug)]
pub struct SyntheticSurfDetection {
    pub level: u8,
    pub radii: Vec<u32>,
    pub proposals: Vec<Proposal>,
    pub best: SyntheticSurfCandidate,
    pub candidates: Vec<SyntheticSurfCandidate>,
}

const CASE1_DISTRACTORS: [RenderedEllipse; 2] = [
    RenderedEllipse {
        center: PixelCoord::new(340.0, 280.0),
        semi_major: 70.0,
        semi_minor: 58.0,
        angle: -0.2,
        value: 150.0,
    },
    RenderedEllipse {
        center: PixelCoord::new(1660.0, 330.0),
        semi_major: 58.0,
        semi_minor: 45.0,
        angle: 0.35,
        value: 138.0,
    },
];

const CASE2_DISTRACTORS: [RenderedEllipse; 2] = [
    RenderedEllipse {
        center: PixelCoord::new(910.0, 200.0),
        semi_major: 72.0,
        semi_minor: 54.0,
        angle: 0.4,
        value: 148.0,
    },
    RenderedEllipse {
        center: PixelCoord::new(220.0, 710.0),
        semi_major: 48.0,
        semi_minor: 40.0,
        angle: -0.3,
        value: 132.0,
    },
];

const CASE3_DISTRACTORS: [RenderedEllipse; 3] = [
    RenderedEllipse {
        center: PixelCoord::new(1510.0, 1110.0),
        semi_major: 92.0,
        semi_minor: 74.0,
        angle: 0.1,
        value: 155.0,
    },
    RenderedEllipse {
        center: PixelCoord::new(470.0, 1115.0),
        semi_major: 62.0,
        semi_minor: 52.0,
        angle: -0.45,
        value: 135.0,
    },
    RenderedEllipse {
        center: PixelCoord::new(1630.0, 410.0),
        semi_major: 54.0,
        semi_minor: 42.0,
        angle: 0.2,
        value: 128.0,
    },
];

pub const CASES: [SyntheticSurfCase; 3] = [
    SyntheticSurfCase {
        name: "large_tilted",
        width: 2048,
        height: 1536,
        target: RenderedEllipse {
            center: PixelCoord::new(1065.0, 791.0),
            semi_major: 178.0,
            semi_minor: 149.0,
            angle: 0.18,
            value: 224.0,
        },
        distractors: &CASE1_DISTRACTORS,
        blur_passes: 2,
    },
    SyntheticSurfCase {
        name: "medium_tilted",
        width: 1184,
        height: 928,
        target: RenderedEllipse {
            center: PixelCoord::new(631.0, 448.0),
            semi_major: 110.0,
            semi_minor: 90.0,
            angle: -0.31,
            value: 220.0,
        },
        distractors: &CASE2_DISTRACTORS,
        blur_passes: 2,
    },
    SyntheticSurfCase {
        name: "large_wide",
        width: 2048,
        height: 1536,
        target: RenderedEllipse {
            center: PixelCoord::new(1005.0, 748.0),
            semi_major: 226.0,
            semi_minor: 185.0,
            angle: -0.35,
            value: 228.0,
        },
        distractors: &CASE3_DISTRACTORS,
        blur_passes: 2,
    },
];

pub fn render_case(case: &SyntheticSurfCase) -> OwnedImage<u8> {
    let mut data = vec![0.0f32; case.width * case.height];
    add_background(&mut data, case.width, case.height);
    fill_ellipse(&mut data, case.width, case.height, case.target);
    for distractor in case.distractors {
        fill_ellipse(&mut data, case.width, case.height, *distractor);
    }
    blur3x3_inplace(&mut data, case.width, case.height, case.blur_passes);

    let bytes = data
        .into_iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    OwnedImage::from_vec(bytes, case.width, case.height).unwrap()
}

pub fn detect_case_image(image: &OwnedImage<u8>, level: u8) -> Result<SyntheticSurfDetection> {
    let pyramid = pyramid_level_owned(&image.view(), level)?;
    let working_image = pyramid.image();
    let working_size = working_image.width().min(working_image.height()) as f32;
    let radius_hint = (working_size * 0.16).max(14.0);
    let radii = build_radius_band(radius_hint, 5);

    let gradient = sobel_gradient(&working_image)?;
    let frst_config = surf_frst_config(radii.clone());
    let response = radsym::frst_response(&gradient, &frst_config)?;
    let nms_config = surf_nms_config(radius_hint);
    let proposals = extract_proposals(&response, &nms_config, Polarity::Bright);

    if proposals.is_empty() {
        return Err(RadSymError::DegenerateHypothesis {
            reason: "no proposals remained after extraction",
        });
    }

    let candidates = rank_candidates(
        &gradient,
        working_image.width(),
        working_image.height(),
        &pyramid,
        &proposals,
        radius_hint,
    );
    let best = candidates
        .first()
        .cloned()
        .ok_or(RadSymError::DegenerateHypothesis {
            reason: "no ranked candidates remained after refinement",
        })?;

    Ok(SyntheticSurfDetection {
        level,
        radii,
        proposals,
        best,
        candidates,
    })
}

// ---------------------------------------------------------------------------
// Shared config builders — single source of truth for the surf-hole pipeline,
// used by both `detect_case_image`/`rank_candidates` (the CI-tested detection)
// and `time_case_once` (the perf page's stage timing) so the two never drift.
// ---------------------------------------------------------------------------

fn surf_frst_config(radii: Vec<u32>) -> FrstConfig {
    let mut frst_config = FrstConfig::default();
    frst_config.radii = radii;
    frst_config.alpha = 2.0;
    frst_config.gradient_threshold = 1.5;
    frst_config.polarity = Polarity::Bright;
    frst_config.smoothing_factor = 0.5;
    frst_config
}

fn surf_nms_config(radius_hint: f32) -> NmsConfig {
    let mut nms_config = NmsConfig::default();
    nms_config.radius = (radius_hint * 0.8).round().max(10.0) as usize;
    nms_config.threshold = 0.01;
    nms_config.max_detections = 12;
    nms_config
}

fn surf_rank_scoring_config() -> ScoringConfig {
    let mut scoring_config = ScoringConfig::default();
    scoring_config.annulus_margin = 0.12;
    scoring_config.min_samples = 32;
    scoring_config
}

fn surf_ellipse_refine_config() -> EllipseRefineConfig {
    let mut refine_advanced = EllipseRefineAdvanced::default();
    refine_advanced.ray_count = 96;
    refine_advanced.radial_search_inner = 0.60;
    refine_advanced.radial_search_outer = 1.45;
    refine_advanced.normal_search_half_width = 6.0;
    refine_advanced.min_inlier_coverage = 0.60;
    let mut refine_config = EllipseRefineConfig::default();
    refine_config.max_iterations = 5;
    refine_config.convergence_tol = 0.05;
    refine_config.max_center_shift_fraction = 0.40;
    refine_config.max_axis_ratio = 1.80;
    refine_config.advanced = refine_advanced;
    refine_config
}

/// Wall-clock cost of each stage of one [`detect_case_image`] pass.
///
/// Mirrors the five stage bars the performance page draws for other cases:
/// gradient (pyramid downsample + Sobel), voting (FRST), extract (NMS), score
/// (per-proposal radius sweep), and refine (per-candidate ellipse fit).
#[derive(Clone, Copy, Debug)]
pub struct SurfStageDurations {
    pub gradient: Duration,
    pub voting: Duration,
    pub extract: Duration,
    pub score: Duration,
    pub refine: Duration,
}

/// Run one full [`detect_case_image`]-equivalent pass, timing each stage.
///
/// Reproduces the exact stage sequence of [`detect_case_image`] using the same
/// shared config builders and helpers, so the reported bars stay faithful to the
/// detection whose ellipse the page overlays. The one-time pyramid downsample is
/// folded into the `gradient` stage (detection runs on the small working image,
/// so the stage bars do not sum to a full-resolution `detect_circles` cost).
pub fn time_case_once(image: &OwnedImage<u8>, level: u8) -> Result<SurfStageDurations> {
    // Stage 1: pyramid downsample + Sobel gradient on the working image.
    let t = Instant::now();
    let pyramid = pyramid_level_owned(&image.view(), level)?;
    let working_image = pyramid.image();
    let gradient = sobel_gradient(&working_image)?;
    let gradient_dur = t.elapsed();

    let working_size = working_image.width().min(working_image.height()) as f32;
    let radius_hint = (working_size * 0.16).max(14.0);
    let radii = build_radius_band(radius_hint, 5);

    // Stage 2: FRST voting.
    let frst_config = surf_frst_config(radii);
    let t = Instant::now();
    let response = radsym::frst_response(black_box(&gradient), black_box(&frst_config))?;
    let voting_dur = t.elapsed();

    // Stage 3: NMS proposal extraction.
    let nms_config = surf_nms_config(radius_hint);
    let t = Instant::now();
    let proposals = extract_proposals(black_box(&response), &nms_config, Polarity::Bright);
    let extract_dur = t.elapsed();

    if proposals.is_empty() {
        return Err(RadSymError::DegenerateHypothesis {
            reason: "no proposals remained after extraction",
        });
    }

    // Stage 4: per-proposal radius sweep → ellipse seeds (mirrors rank_candidates).
    let t = Instant::now();
    let mut seeds = Vec::with_capacity(proposals.len().min(12));
    for proposal in proposals.iter().take(12) {
        let center = proposal.seed.position;
        let (radius_sweep_radius, _) = sweep_radius_at_center(&gradient, center, radius_hint);
        let seed_radius = radius_sweep_radius.max(0.8 * radius_hint);
        seeds.push(Ellipse::new(center, seed_radius, seed_radius, 0.0));
    }
    let score_dur = t.elapsed();

    // Stage 5: per-candidate ellipse refinement.
    let refine_config = surf_ellipse_refine_config();
    let t = Instant::now();
    for seed in &seeds {
        let _ = black_box(refine_ellipse(
            black_box(&gradient),
            black_box(seed),
            &refine_config,
        ));
    }
    let refine_dur = t.elapsed();

    Ok(SurfStageDurations {
        gradient: gradient_dur,
        voting: voting_dur,
        extract: extract_dur,
        score: score_dur,
        refine: refine_dur,
    })
}

// ---------------------------------------------------------------------------
// Multi-ellipse detection: recover ALL elliptical features (the target hole
// plus the smaller distractors), not just the single best one. Used by the
// performance page to show that varied-scale ellipses are all localised.
// ---------------------------------------------------------------------------

/// FRST radii spanning the small distractors up to the large target hole (on
/// the level-3 working image); wider than [`surf_frst_config`]'s target-only band.
const MULTI_SEED_RADII: [f32; 4] = [8.0, 14.0, 22.0, 30.0];
/// Two full-res detections closer than this are treated as the same feature.
const MULTI_DEDUPE_DIST: f32 = 60.0;
/// Minimum ellipse support (ringness+coverage) to accept a refined candidate.
const MULTI_MIN_SUPPORT: f32 = 0.4;

/// One detected elliptical feature, in both working- and full-image frames.
#[derive(Clone, Copy, Debug)]
pub struct SurfEllipse {
    /// Ellipse in full-resolution image coordinates (for overlays/GT checks).
    pub image_ellipse: Ellipse,
    /// Ellipse in the downsampled working-image frame.
    pub working_ellipse: Ellipse,
    /// Support score (0..1) at acceptance.
    pub support: f32,
}

/// All elliptical features found in a surf-hole image.
#[derive(Clone, Debug)]
pub struct SurfMultiDetection {
    pub level: u8,
    pub radii: Vec<u32>,
    pub proposal_count: usize,
    /// Accepted ellipses, strongest support first.
    pub ellipses: Vec<SurfEllipse>,
}

fn surf_multi_frst_config() -> FrstConfig {
    let mut frst_config = FrstConfig::default();
    frst_config.radii = vec![6, 9, 12, 16, 20, 26, 32];
    frst_config.alpha = 2.0;
    frst_config.gradient_threshold = 1.5;
    frst_config.polarity = Polarity::Bright;
    frst_config.smoothing_factor = 0.5;
    frst_config
}

fn surf_multi_nms_config() -> NmsConfig {
    let mut nms_config = NmsConfig::default();
    nms_config.radius = 8;
    nms_config.threshold = 0.01;
    nms_config.max_detections = 16;
    nms_config
}

fn surf_multi_scoring_config() -> ScoringConfig {
    let mut scoring_config = ScoringConfig::default();
    scoring_config.annulus_margin = 0.15;
    scoring_config.min_samples = 16;
    scoring_config
}

fn surf_multi_refine_config() -> EllipseRefineConfig {
    let mut refine_advanced = EllipseRefineAdvanced::default();
    refine_advanced.ray_count = 96;
    refine_advanced.radial_search_inner = 0.55;
    refine_advanced.radial_search_outer = 1.5;
    refine_advanced.normal_search_half_width = 6.0;
    refine_advanced.min_inlier_coverage = 0.5;
    let mut refine_config = EllipseRefineConfig::default();
    refine_config.max_iterations = 6;
    refine_config.convergence_tol = 0.05;
    refine_config.max_center_shift_fraction = 0.45;
    refine_config.max_axis_ratio = 2.2;
    refine_config.advanced = refine_advanced;
    refine_config
}

/// Detect every elliptical feature (target hole + distractors) in a surf image.
///
/// Runs FRST over a wide radius band on the level-`level` working image, refines
/// each proposal into an ellipse from several seed radii, keeps well-supported
/// fits, and deduplicates by full-resolution center. Returns ellipses sorted by
/// descending support.
pub fn detect_all_ellipses(image: &OwnedImage<u8>, level: u8) -> Result<SurfMultiDetection> {
    let pyramid = pyramid_level_owned(&image.view(), level)?;
    let working_image = pyramid.image();
    let gradient = sobel_gradient(&working_image)?;

    let frst_config = surf_multi_frst_config();
    let radii = frst_config.radii.clone();
    let response = radsym::frst_response(&gradient, &frst_config)?;
    let proposals = extract_proposals(&response, &surf_multi_nms_config(), Polarity::Bright);
    let proposal_count = proposals.len();

    let scoring_config = surf_multi_scoring_config();
    let refine_config = surf_multi_refine_config();
    let mut candidates: Vec<SurfEllipse> = Vec::new();
    for proposal in &proposals {
        for &seed_radius in &MULTI_SEED_RADII {
            let seed = Ellipse::new(proposal.seed.position, seed_radius, seed_radius, 0.0);
            let Ok(refined) = refine_ellipse(&gradient, &seed, &refine_config) else {
                continue;
            };
            let working_ellipse = refined.hypothesis;
            let support =
                score_ellipse_support_detailed(&gradient, &working_ellipse, &scoring_config);
            if support.is_degenerate || support.total <= MULTI_MIN_SUPPORT {
                continue;
            }
            candidates.push(SurfEllipse {
                image_ellipse: pyramid.map_ellipse_to_image(working_ellipse),
                working_ellipse,
                support: support.total,
            });
        }
    }

    candidates.sort_by(|a, b| b.support.total_cmp(&a.support));
    let mut ellipses: Vec<SurfEllipse> = Vec::new();
    for candidate in candidates {
        let is_new = ellipses.iter().all(|kept| {
            (kept.image_ellipse.center.x - candidate.image_ellipse.center.x)
                .hypot(kept.image_ellipse.center.y - candidate.image_ellipse.center.y)
                > MULTI_DEDUPE_DIST
        });
        if is_new {
            ellipses.push(candidate);
        }
    }

    Ok(SurfMultiDetection {
        level,
        radii,
        proposal_count,
        ellipses,
    })
}

/// Time the [`detect_all_ellipses`] stages for the performance page (same five
/// bars as the other cases; the one-time pyramid downsample folds into gradient).
pub fn time_multi_once(image: &OwnedImage<u8>, level: u8) -> Result<SurfStageDurations> {
    let t = Instant::now();
    let pyramid = pyramid_level_owned(&image.view(), level)?;
    let working_image = pyramid.image();
    let gradient = sobel_gradient(&working_image)?;
    let gradient_dur = t.elapsed();

    let frst_config = surf_multi_frst_config();
    let t = Instant::now();
    let response = radsym::frst_response(black_box(&gradient), black_box(&frst_config))?;
    let voting_dur = t.elapsed();

    let t = Instant::now();
    let proposals = extract_proposals(
        black_box(&response),
        &surf_multi_nms_config(),
        Polarity::Bright,
    );
    let extract_dur = t.elapsed();

    let scoring_config = surf_multi_scoring_config();
    let refine_config = surf_multi_refine_config();
    let mut refine_dur = Duration::ZERO;
    let mut score_dur = Duration::ZERO;
    for proposal in &proposals {
        for &seed_radius in &MULTI_SEED_RADII {
            let seed = Ellipse::new(proposal.seed.position, seed_radius, seed_radius, 0.0);
            let t = Instant::now();
            let refined = refine_ellipse(black_box(&gradient), black_box(&seed), &refine_config);
            refine_dur += t.elapsed();
            if let Ok(refined) = refined {
                let t = Instant::now();
                let _ = black_box(score_ellipse_support_detailed(
                    black_box(&gradient),
                    black_box(&refined.hypothesis),
                    &scoring_config,
                ));
                score_dur += t.elapsed();
            }
        }
    }

    Ok(SurfStageDurations {
        gradient: gradient_dur,
        voting: voting_dur,
        extract: extract_dur,
        score: score_dur,
        refine: refine_dur,
    })
}

pub fn normalize_ellipse(ellipse: Ellipse) -> Ellipse {
    if ellipse.semi_minor > ellipse.semi_major {
        Ellipse::new(
            ellipse.center,
            ellipse.semi_minor,
            ellipse.semi_major,
            wrap_half_turn(ellipse.angle + std::f32::consts::FRAC_PI_2),
        )
    } else {
        Ellipse::new(
            ellipse.center,
            ellipse.semi_major,
            ellipse.semi_minor,
            wrap_half_turn(ellipse.angle),
        )
    }
}

pub fn ellipse_iou(pred: Ellipse, target: Ellipse) -> f32 {
    let min_x = (pred.center.x - pred.semi_major)
        .min(target.center.x - target.semi_major)
        .floor() as i32
        - 4;
    let max_x = (pred.center.x + pred.semi_major)
        .max(target.center.x + target.semi_major)
        .ceil() as i32
        + 4;
    let min_y = (pred.center.y - pred.semi_major)
        .min(target.center.y - target.semi_major)
        .floor() as i32
        - 4;
    let max_y = (pred.center.y + pred.semi_major)
        .max(target.center.y + target.semi_major)
        .ceil() as i32
        + 4;

    let mut intersection = 0usize;
    let mut union = 0usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let pred_contains = ellipse_contains(pred, x as f32, y as f32);
            let target_contains = ellipse_contains(target, x as f32, y as f32);
            if pred_contains && target_contains {
                intersection += 1;
            }
            if pred_contains || target_contains {
                union += 1;
            }
        }
    }

    intersection as f32 / union.max(1) as f32
}

fn rank_candidates(
    gradient: &radsym::core::gradient::GradientField,
    width: usize,
    height: usize,
    pyramid: &radsym::core::pyramid::OwnedPyramidLevel,
    proposals: &[Proposal],
    radius_hint: f32,
) -> Vec<SyntheticSurfCandidate> {
    let center_x = width as f32 * 0.5;
    let center_y = height as f32 * 0.5;
    let max_center_distance = center_x.hypot(center_y);
    let scoring_config = surf_rank_scoring_config();
    let refine_config = surf_ellipse_refine_config();

    let mut ranked = Vec::new();
    for proposal in proposals.iter().take(12) {
        let center = proposal.seed.position;
        let (radius_sweep_radius, _) = sweep_radius_at_center(gradient, center, radius_hint);
        let seed_radius = radius_sweep_radius.max(0.8 * radius_hint);
        let ellipse_seed = Ellipse::new(center, seed_radius, seed_radius, 0.0);
        let refined = refine_ellipse(gradient, &ellipse_seed, &refine_config).unwrap();
        let working_ellipse = refined.hypothesis;
        let final_support =
            score_ellipse_support_detailed(gradient, &working_ellipse, &scoring_config);

        let dx = working_ellipse.center.x - center_x;
        let dy = working_ellipse.center.y - center_y;
        let center_distance = dx.hypot(dy);
        let center_bonus = (1.0 - center_distance / max_center_distance).max(0.0);
        let mean_radius = 0.5 * (working_ellipse.semi_major + working_ellipse.semi_minor);
        let radius_error = (mean_radius - radius_sweep_radius).abs();
        let size_sigma = (0.22 * radius_sweep_radius).max(1.0);
        let size_consistency = (-0.5 * (radius_error / size_sigma).powi(2)).exp();
        let fit_quality = 1.0 / (1.0 + 12.0 * refined.residual.max(0.0));
        let combined = fit_quality
            * final_support.angular_coverage
            * (0.6 + 0.4 * final_support.ringness)
            * size_consistency
            * (0.85 + 0.15 * center_bonus);

        ranked.push(SyntheticSurfCandidate {
            proposal: proposal.clone(),
            working_ellipse,
            image_ellipse: pyramid.map_ellipse_to_image(working_ellipse),
            combined_score: combined,
            radius_sweep_radius,
            final_support,
        });
    }

    ranked.sort_by(|lhs, rhs| rhs.combined_score.total_cmp(&lhs.combined_score));
    ranked
}

fn sweep_radius_at_center(
    gradient: &radsym::core::gradient::GradientField,
    center: PixelCoord,
    radius_hint: f32,
) -> (f32, SupportScoreBreakdown) {
    let mut scoring_config = ScoringConfig::default();
    scoring_config.annulus_margin = 0.10;
    scoring_config.min_samples = 24;
    scoring_config.weight_ringness = 0.75;
    scoring_config.weight_coverage = 0.25;
    let radius_min = (radius_hint * 0.35).max(6.0);
    let radius_max = (radius_hint * 1.05).max(radius_min + 6.0);

    let mut radius = radius_min;
    let mut best: Option<(f32, SupportScoreBreakdown)> = None;
    while radius <= radius_max + 0.5 {
        let circle = Circle::new(center, radius);
        let score = score_circle_support_detailed(gradient, &circle, &scoring_config);
        if !score.is_degenerate {
            match best {
                None => best = Some((radius, score)),
                Some((best_radius, best_score))
                    if score.total > best_score.total + 1e-4
                        || ((score.total - best_score.total).abs() <= 1e-4
                            && radius < best_radius) =>
                {
                    best = Some((radius, score))
                }
                _ => {}
            }
        }
        radius += 1.0;
    }

    best.unwrap_or_else(|| {
        let fallback = Circle::new(center, radius_hint);
        (
            radius_hint,
            score_circle_support_detailed(gradient, &fallback, &scoring_config),
        )
    })
}

fn build_radius_band(base_radius: f32, steps: usize) -> Vec<u32> {
    let start = (base_radius * 0.65).round().max(6.0) as i32;
    let stop = (base_radius * 1.35).round().max((start + 1) as f32) as i32;
    if steps <= 1 || start == stop {
        return vec![start as u32];
    }

    let mut radii = Vec::with_capacity(steps);
    for index in 0..steps {
        let value = (start as f32 + (stop - start) as f32 * index as f32 / (steps - 1) as f32)
            .round() as u32;
        if value > 0 && radii.last().copied() != Some(value) {
            radii.push(value);
        }
    }
    radii
}

fn add_background(data: &mut [f32], width: usize, height: usize) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let smooth_gradient = 22.0 + 0.012 * x as f32 + 0.008 * y as f32;
            let texture = 3.5 * ((x as f32 * 0.021).sin() * (y as f32 * 0.017).cos() + 1.0);
            data[idx] = smooth_gradient + texture;
        }
    }
}

fn fill_ellipse(data: &mut [f32], width: usize, height: usize, ellipse: RenderedEllipse) {
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let extent = ellipse.semi_major.max(ellipse.semi_minor) + 3.0;
    let x0 = (ellipse.center.x - extent).floor().max(0.0) as usize;
    let y0 = (ellipse.center.y - extent).floor().max(0.0) as usize;
    let x1 = (ellipse.center.x + extent).ceil().min((width - 1) as f32) as usize;
    let y1 = (ellipse.center.y + extent).ceil().min((height - 1) as f32) as usize;

    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 - ellipse.center.x;
            let dy = y as f32 - ellipse.center.y;
            let lx = dx * cos_a + dy * sin_a;
            let ly = -dx * sin_a + dy * cos_a;
            let level = (lx / ellipse.semi_major).powi(2) + (ly / ellipse.semi_minor).powi(2);
            if level <= 1.0 {
                data[y * width + x] = ellipse.value;
            }
        }
    }
}

fn blur3x3_inplace(data: &mut [f32], width: usize, height: usize, passes: usize) {
    let mut tmp = vec![0.0f32; data.len()];
    for _ in 0..passes {
        for y in 0..height {
            for x in 0..width {
                let left = x.saturating_sub(1);
                let right = (x + 1).min(width - 1);
                tmp[y * width + x] =
                    (data[y * width + left] + data[y * width + x] + data[y * width + right]) / 3.0;
            }
        }

        for y in 0..height {
            let up = y.saturating_sub(1);
            let down = (y + 1).min(height - 1);
            for x in 0..width {
                data[y * width + x] =
                    (tmp[up * width + x] + tmp[y * width + x] + tmp[down * width + x]) / 3.0;
            }
        }
    }
}

fn wrap_half_turn(mut angle: f32) -> f32 {
    while angle <= -std::f32::consts::FRAC_PI_2 {
        angle += std::f32::consts::PI;
    }
    while angle > std::f32::consts::FRAC_PI_2 {
        angle -= std::f32::consts::PI;
    }
    angle
}

fn ellipse_contains(ellipse: Ellipse, x: f32, y: f32) -> bool {
    let dx = x - ellipse.center.x;
    let dy = y - ellipse.center.y;
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let lx = dx * cos_a + dy * sin_a;
    let ly = -dx * sin_a + dy * cos_a;
    (lx / ellipse.semi_major).powi(2) + (ly / ellipse.semi_minor).powi(2) <= 1.0
}
