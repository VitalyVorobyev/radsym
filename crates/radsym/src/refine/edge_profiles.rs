use std::cmp::Ordering;

use crate::core::coords::PixelCoord;
use crate::core::image_view::ImageView;
use crate::core::scalar::Scalar;

pub(super) const DEFAULT_MAX_EDGE_CANDIDATES: usize = 3;
pub(super) const DEFAULT_PEAK_MIN_SEPARATION_PX: Scalar = 2.0;

#[derive(Clone, Copy)]
enum EnvelopeSelector {
    Strongest,
    Inner,
    SecondInner,
    Outer,
    SecondOuter,
    Median,
}

#[inline]
fn sample_gradient(
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    x: Scalar,
    y: Scalar,
) -> Option<(Scalar, Scalar)> {
    Some((gx_view.sample(x, y)?, gy_view.sample(x, y)?))
}

pub(super) fn clamp_center_shift(
    seed: PixelCoord,
    candidate: PixelCoord,
    max_shift: Scalar,
) -> PixelCoord {
    let dx = candidate.x - seed.x;
    let dy = candidate.y - seed.y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist <= max_shift || dist <= 1e-8 {
        candidate
    } else {
        let scale = max_shift / dist;
        PixelCoord::new(seed.x + dx * scale, seed.y + dy * scale)
    }
}

pub(super) fn smooth_profile(scores: &[Scalar]) -> Vec<Scalar> {
    let mut smoothed = Vec::with_capacity(scores.len());
    for i in 0..scores.len() {
        let prev = if i > 0 { scores[i - 1] } else { scores[i] };
        let next = if i + 1 < scores.len() {
            scores[i + 1]
        } else {
            scores[i]
        };
        smoothed.push(0.25 * prev + 0.5 * scores[i] + 0.25 * next);
    }
    smoothed
}

pub(super) fn select_peak_indices(
    offsets: &[Scalar],
    smooth_scores: &[Scalar],
    max_candidates: usize,
    min_separation: Scalar,
) -> Vec<usize> {
    if offsets.is_empty() || smooth_scores.is_empty() || max_candidates == 0 {
        return Vec::new();
    }

    let mut peaks = Vec::new();
    for i in 0..smooth_scores.len() {
        let prev = if i > 0 {
            smooth_scores[i - 1]
        } else {
            smooth_scores[i]
        };
        let next = if i + 1 < smooth_scores.len() {
            smooth_scores[i + 1]
        } else {
            smooth_scores[i]
        };
        let is_peak = (i == 0 || smooth_scores[i] >= prev)
            && (i + 1 == smooth_scores.len() || smooth_scores[i] >= next);
        if is_peak && smooth_scores[i] > 1e-4 {
            peaks.push(i);
        }
    }

    if peaks.is_empty() {
        return Vec::new();
    }

    let mut selected = Vec::new();
    let try_add = |index: usize, selected: &mut Vec<usize>| {
        if selected.len() >= max_candidates || selected.contains(&index) {
            return;
        }
        if selected
            .iter()
            .any(|&other| (offsets[other] - offsets[index]).abs() < min_separation)
        {
            return;
        }
        selected.push(index);
    };

    let strongest = peaks
        .iter()
        .copied()
        .max_by(|&lhs, &rhs| {
            smooth_scores[lhs]
                .partial_cmp(&smooth_scores[rhs])
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    offsets[rhs]
                        .partial_cmp(&offsets[lhs])
                        .unwrap_or(Ordering::Equal)
                })
        })
        .unwrap();
    let inner = *peaks
        .iter()
        .min_by(|&&lhs, &&rhs| {
            offsets[lhs]
                .partial_cmp(&offsets[rhs])
                .unwrap_or(Ordering::Equal)
        })
        .unwrap();
    let outer = *peaks
        .iter()
        .max_by(|&&lhs, &&rhs| {
            offsets[lhs]
                .partial_cmp(&offsets[rhs])
                .unwrap_or(Ordering::Equal)
        })
        .unwrap();

    for index in [strongest, inner, outer] {
        try_add(index, &mut selected);
    }

    let mut remaining = peaks;
    remaining.sort_by(|&lhs, &rhs| {
        smooth_scores[rhs]
            .partial_cmp(&smooth_scores[lhs])
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                offsets[lhs]
                    .partial_cmp(&offsets[rhs])
                    .unwrap_or(Ordering::Equal)
            })
    });
    for index in remaining {
        try_add(index, &mut selected);
        if selected.len() >= max_candidates {
            break;
        }
    }

    selected.sort_by(|&lhs, &rhs| {
        offsets[lhs]
            .partial_cmp(&offsets[rhs])
            .unwrap_or(Ordering::Equal)
    });
    selected
}

#[allow(clippy::too_many_arguments)]
pub(super) fn edge_candidates_along_ray<T, F>(
    gx_view: ImageView<'_, Scalar>,
    gy_view: ImageView<'_, Scalar>,
    origin: PixelCoord,
    dir_x: Scalar,
    dir_y: Scalar,
    start: Scalar,
    stop: Scalar,
    step: Scalar,
    sector: usize,
    bias_center: Scalar,
    bias_sigma: Scalar,
    expected_sign: Scalar,
    max_candidates: usize,
    peak_min_separation_px: Scalar,
    mut build_observation: F,
) -> Vec<T>
where
    F: FnMut(PixelCoord, Scalar, Scalar, Scalar, usize) -> Option<T>,
{
    let samples = (((stop - start) / step).ceil() as usize).max(2) + 1;
    let mut offsets = Vec::with_capacity(samples);
    let mut scores = Vec::with_capacity(samples);
    let mut signed_projections = Vec::with_capacity(samples);

    for i in 0..samples {
        let offset = start + i as Scalar * step;
        if offset > stop + 1e-6 {
            break;
        }
        let x = origin.x + dir_x * offset;
        let y = origin.y + dir_y * offset;
        let (score, signed_projection) =
            if let Some((gx, gy)) = sample_gradient(gx_view, gy_view, x, y) {
                let signed_projection = gx * dir_x + gy * dir_y;
                let projected = if expected_sign.abs() > 0.5 {
                    (expected_sign * signed_projection).max(0.0)
                } else {
                    signed_projection.abs()
                };
                let score = if bias_sigma > 1e-6 {
                    let z = (offset - bias_center) / bias_sigma;
                    projected * (-0.5 * z * z).exp()
                } else {
                    projected
                };
                (score, signed_projection)
            } else {
                (0.0, 0.0)
            };
        offsets.push(offset);
        scores.push(score);
        signed_projections.push(signed_projection);
    }

    let smooth_scores = smooth_profile(&scores);
    let selected = select_peak_indices(
        &offsets,
        &smooth_scores,
        max_candidates,
        peak_min_separation_px.max(step),
    );
    let mut observations = Vec::with_capacity(selected.len());
    for index in selected {
        let offset = offsets[index];
        let point = PixelCoord::new(origin.x + dir_x * offset, origin.y + dir_y * offset);
        if let Some(observation) = build_observation(
            point,
            smooth_scores[index],
            offset,
            signed_projections[index],
            sector,
        ) {
            observations.push(observation);
        }
    }
    observations
}

pub(super) fn infer_expected_sign<T, FScore, FProjection>(
    observations: &[T],
    score_of: FScore,
    signed_projection_of: FProjection,
) -> Scalar
where
    FScore: Fn(&T) -> Scalar,
    FProjection: Fn(&T) -> Scalar,
{
    let signed_sum: Scalar = observations
        .iter()
        .map(|obs| score_of(obs) * signed_projection_of(obs).signum())
        .sum();
    let magnitude_sum: Scalar = observations.iter().map(|obs| score_of(obs).abs()).sum();
    if magnitude_sum <= 1e-6 || signed_sum.abs() < 0.1 * magnitude_sum {
        0.0
    } else {
        signed_sum.signum()
    }
}

fn envelope_candidate<T, FScore, FOffset>(
    candidates: &[T],
    selector: EnvelopeSelector,
    score_of: FScore,
    offset_of: FOffset,
) -> Option<T>
where
    T: Copy,
    FScore: Fn(&T) -> Scalar,
    FOffset: Fn(&T) -> Scalar,
{
    match selector {
        EnvelopeSelector::Strongest => candidates.iter().copied().max_by(|lhs, rhs| {
            score_of(lhs)
                .partial_cmp(&score_of(rhs))
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    offset_of(lhs)
                        .partial_cmp(&offset_of(rhs))
                        .unwrap_or(Ordering::Equal)
                })
        }),
        EnvelopeSelector::Inner => candidates.first().copied(),
        EnvelopeSelector::SecondInner => candidates
            .get(1)
            .copied()
            .or_else(|| candidates.first().copied()),
        EnvelopeSelector::Outer => candidates.last().copied(),
        EnvelopeSelector::SecondOuter => candidates
            .get(candidates.len().saturating_sub(2))
            .copied()
            .or_else(|| candidates.last().copied()),
        EnvelopeSelector::Median => candidates.get(candidates.len() / 2).copied(),
    }
}

fn observations_from_envelope<T, FScore, FOffset>(
    sector_candidates: &[Vec<T>],
    selector: EnvelopeSelector,
    score_of: FScore,
    offset_of: FOffset,
) -> Vec<T>
where
    T: Copy,
    FScore: Fn(&T) -> Scalar + Copy,
    FOffset: Fn(&T) -> Scalar + Copy,
{
    sector_candidates
        .iter()
        .filter_map(|candidates| envelope_candidate(candidates, selector, score_of, offset_of))
        .collect()
}

fn push_unique_hypothesis<T>(hypotheses: &mut Vec<Vec<T>>, observations: Vec<T>)
where
    T: PartialEq,
{
    if observations.is_empty() || hypotheses.iter().any(|existing| existing == &observations) {
        return;
    }
    hypotheses.push(observations);
}

pub(super) fn best_hypotheses<T, FScore, FOffset>(
    sector_candidates: &[Vec<T>],
    score_of: FScore,
    offset_of: FOffset,
) -> Vec<Vec<T>>
where
    T: Copy + PartialEq,
    FScore: Fn(&T) -> Scalar + Copy,
    FOffset: Fn(&T) -> Scalar + Copy,
{
    let mut hypotheses = Vec::new();
    for selector in [
        EnvelopeSelector::Strongest,
        EnvelopeSelector::Inner,
        EnvelopeSelector::SecondInner,
        EnvelopeSelector::Outer,
        EnvelopeSelector::SecondOuter,
        EnvelopeSelector::Median,
    ] {
        push_unique_hypothesis(
            &mut hypotheses,
            observations_from_envelope(sector_candidates, selector, score_of, offset_of),
        );
    }
    hypotheses
}

fn select_best_candidate<T, FScore, FOffset, FResidual>(
    candidates: &[T],
    score_of: FScore,
    offset_of: FOffset,
    residual_of: FResidual,
) -> Option<T>
where
    T: Copy,
    FScore: Fn(&T) -> Scalar,
    FOffset: Fn(&T) -> Scalar,
    FResidual: Fn(&T) -> Option<Scalar>,
{
    let mut best = None;
    let mut best_residual = Scalar::INFINITY;
    let mut best_score = -Scalar::INFINITY;
    let mut best_offset = 0.0;

    for candidate in candidates {
        let residual = residual_of(candidate)?.abs();
        let score = score_of(candidate);
        let offset = offset_of(candidate);
        let is_better = score > best_score + 1e-6
            || ((score - best_score).abs() <= 1e-6
                && (residual + 1e-6 < best_residual
                    || ((residual - best_residual).abs() <= 1e-6 && offset > best_offset + 1e-6)));
        if is_better {
            best = Some(*candidate);
            best_residual = residual;
            best_score = score;
            best_offset = offset;
        }
    }

    best
}

pub(super) fn select_best_consistent_candidates<T, FScore, FOffset, FResidual>(
    sector_candidates: &[Vec<T>],
    score_of: FScore,
    offset_of: FOffset,
    residual_of: FResidual,
) -> Vec<T>
where
    T: Copy,
    FScore: Fn(&T) -> Scalar + Copy,
    FOffset: Fn(&T) -> Scalar + Copy,
    FResidual: Fn(&T) -> Option<Scalar> + Copy,
{
    sector_candidates
        .iter()
        .filter_map(|candidates| {
            select_best_candidate(candidates, score_of, offset_of, residual_of)
        })
        .collect()
}
