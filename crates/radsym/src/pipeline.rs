//! High-level detection pipeline.
//!
//! Provides a single-call entry point for the common propose-score-refine
//! workflow. Power users can still compose the individual stages manually.

use crate::core::error::Result;
use crate::core::geometry::Circle;
use crate::core::gradient::{GradientOperator, compute_gradient};
use crate::core::image_view::ImageView;
use crate::core::nms::NmsConfig;
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;
use crate::diagnostics::detection::{
    CircleDetectionDiagnostics, RejectedProposal, RejectionReason,
};
use crate::propose::extract::extract_proposals;
use crate::propose::frst::{FrstConfig, frst_response};
use crate::refine::circle::{CircleRefineConfig, refine_circle};
use crate::refine::result::RefinementStatus;
use crate::support::score::{
    ScoringConfig, SupportScore, SupportScoreBreakdown, score_circle_support,
};

/// Aggregated configuration for [`detect_circles`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DetectCirclesConfig {
    /// Candidate FRST voting radii, in pixels.
    ///
    /// This is the source of truth for the radii the pipeline votes over:
    /// [`detect_circles`] copies it into the working FRST config, overriding
    /// [`DetectCirclesAdvanced::frst`]'s own `radii`.
    pub radii: Vec<u32>,
    /// Which polarity to detect.
    pub polarity: Polarity,
    /// Approximate expected radius used as the initial circle hypothesis.
    pub radius_hint: Scalar,
    /// Minimum support score to keep a detection (in `[0, 1]`).
    pub min_score: Scalar,
    /// Gradient operator to use (default: Sobel).
    pub gradient_operator: GradientOperator,
    /// Advanced per-stage configuration.
    pub advanced: DetectCirclesAdvanced,
}

/// Advanced per-stage configuration for [`DetectCirclesConfig`].
///
/// These are the individual stage configs assembled by the one-call
/// [`detect_circles`] pipeline. Most callers should leave them at their
/// defaults and drive detection through the stable [`DetectCirclesConfig`]
/// fields; the stage configs are split out here for power users who need to
/// tune FRST voting, NMS, scoring, or refinement directly.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DetectCirclesAdvanced {
    /// FRST voting configuration.
    ///
    /// Note: the pipeline overrides this config's `radii` and `polarity` with
    /// the top-level [`DetectCirclesConfig::radii`] and
    /// [`DetectCirclesConfig::polarity`] before voting, so setting them here
    /// has no effect on [`detect_circles`].
    pub frst: FrstConfig,
    /// Non-maximum suppression for proposal extraction.
    pub nms: NmsConfig,
    /// Support scoring configuration.
    pub scoring: ScoringConfig,
    /// Iterative circle refinement configuration.
    pub refinement: CircleRefineConfig,
}

impl Default for DetectCirclesConfig {
    fn default() -> Self {
        Self {
            radii: FrstConfig::default().radii,
            polarity: Polarity::Both,
            radius_hint: 10.0,
            min_score: 0.0,
            gradient_operator: GradientOperator::default(),
            advanced: DetectCirclesAdvanced::default(),
        }
    }
}

impl DetectCirclesConfig {
    /// Build a configuration for the given candidate FRST radii.
    ///
    /// This is the builder entry point: it starts from
    /// [`DetectCirclesConfig::default()`] and overrides only the top-level
    /// voting [`radii`](DetectCirclesConfig::radii) with the collected
    /// iterator. Chain the setter methods to override further fields, e.g.:
    ///
    /// ```rust
    /// use radsym::pipeline::DetectCirclesConfig;
    /// use radsym::Polarity;
    ///
    /// let config = DetectCirclesConfig::for_radii([9, 10, 11])
    ///     .polarity(Polarity::Bright)
    ///     .radius_hint(10.0)
    ///     .min_score(0.2);
    /// ```
    pub fn for_radii(radii: impl IntoIterator<Item = u32>) -> Self {
        Self {
            radii: radii.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Set the detection polarity (chainable).
    ///
    /// This sets both the top-level `polarity` field and the nested
    /// `advanced.frst.polarity` field. The [`detect_circles`] pipeline drives
    /// FRST voting and proposal extraction from the top-level `polarity` (it
    /// overwrites `advanced.frst.polarity` with it before voting), so updating
    /// both keeps the configuration internally consistent for code that
    /// inspects `advanced.frst.polarity` directly.
    pub fn polarity(mut self, polarity: Polarity) -> Self {
        self.polarity = polarity;
        self.advanced.frst.polarity = polarity;
        self
    }

    /// Set the expected radius used as the initial circle hypothesis (chainable).
    pub fn radius_hint(mut self, radius_hint: Scalar) -> Self {
        self.radius_hint = radius_hint;
        self
    }

    /// Set the minimum support score required to keep a detection (chainable).
    pub fn min_score(mut self, min_score: Scalar) -> Self {
        self.min_score = min_score;
        self
    }

    /// Set the gradient operator used for the pipeline (chainable).
    pub fn gradient_operator(mut self, gradient_operator: GradientOperator) -> Self {
        self.gradient_operator = gradient_operator;
        self
    }
}

/// A detected circle with its support score and refinement status.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[non_exhaustive]
pub struct Detection<T> {
    /// Refined geometric hypothesis.
    pub hypothesis: T,
    /// Support score from gradient evidence.
    pub score: SupportScore,
    /// Refinement convergence status.
    pub status: RefinementStatus,
}

/// The concrete result type produced by [`detect_circles`].
///
/// This is an alias for [`Detection<Circle>`](Detection): the generic
/// `Detection<T>` struct is shared with ellipse-refinement results, so the
/// circle-detection path names the contract through this alias rather than
/// duplicating the struct.
pub type CircleDetection = Detection<Circle>;

/// Detect circles in a grayscale image using the full propose-score-refine pipeline.
///
/// This is a convenience wrapper around the composable stages:
/// 1. Sobel gradient computation
/// 2. FRST voting and NMS proposal extraction
/// 3. Support scoring and filtering
/// 4. Iterative circle refinement
///
/// Returns detections sorted by descending support score.
///
/// # Example
///
/// ```rust
/// use radsym::pipeline::{detect_circles, DetectCirclesConfig};
/// use radsym::{ImageView, Polarity};
///
/// let size = 64;
/// let mut data = vec![0u8; size * size];
/// for y in 0..size {
///     for x in 0..size {
///         let dx = x as f32 - 32.0;
///         let dy = y as f32 - 32.0;
///         if (dx * dx + dy * dy).sqrt() <= 10.0 {
///             data[y * size + x] = 255;
///         }
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
///
/// let config = DetectCirclesConfig::for_radii([9, 10, 11])
///     .polarity(Polarity::Bright)
///     .radius_hint(10.0);
///
/// let detections = detect_circles(&image, &config).unwrap();
/// assert!(!detections.is_empty());
/// ```
pub fn detect_circles(
    image: &ImageView<'_, u8>,
    config: &DetectCirclesConfig,
) -> Result<Vec<CircleDetection>> {
    run_detection(image, config).map(|(detections, _diagnostics)| detections)
}

/// Detect circles and also return diagnostic evidence about the run.
///
/// This is the diagnostics-channel companion to [`detect_circles`]: it returns
/// the same `Vec<CircleDetection>` result plus a [`CircleDetectionDiagnostics`]
/// carrying the response map, the raw proposals, the rejected candidates with
/// their [`RejectionReason`], and a per-detection [`SupportScoreBreakdown`].
/// Use [`detect_circles`] when only the result is needed.
///
/// The diagnostics' `score_breakdowns` vec is index-aligned with the returned
/// detections.
///
/// # Example
///
/// ```rust
/// use radsym::pipeline::{detect_circles_with_diagnostics, DetectCirclesConfig};
/// use radsym::{ImageView, Polarity};
///
/// let size = 64;
/// let mut data = vec![0u8; size * size];
/// for y in 0..size {
///     for x in 0..size {
///         let dx = x as f32 - 32.0;
///         let dy = y as f32 - 32.0;
///         if (dx * dx + dy * dy).sqrt() <= 10.0 {
///             data[y * size + x] = 255;
///         }
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let config = DetectCirclesConfig::for_radii([9, 10, 11]).polarity(Polarity::Bright);
///
/// let (detections, diagnostics) = detect_circles_with_diagnostics(&image, &config).unwrap();
/// assert_eq!(detections.len(), diagnostics.score_breakdowns.len());
/// ```
pub fn detect_circles_with_diagnostics(
    image: &ImageView<'_, u8>,
    config: &DetectCirclesConfig,
) -> Result<(Vec<CircleDetection>, CircleDetectionDiagnostics)> {
    run_detection(image, config)
}

/// Shared implementation behind [`detect_circles`] and
/// [`detect_circles_with_diagnostics`].
///
/// The full propose-score-refine pipeline always builds the diagnostic
/// evidence; [`detect_circles`] simply discards it. The extra bookkeeping is
/// negligible next to the gradient field and response map the pipeline
/// allocates regardless.
fn run_detection(
    image: &ImageView<'_, u8>,
    config: &DetectCirclesConfig,
) -> Result<(Vec<CircleDetection>, CircleDetectionDiagnostics)> {
    config.advanced.refinement.validate()?;

    let gradient = compute_gradient(image, config.gradient_operator)?;

    let mut frst_config = config.advanced.frst.clone();
    frst_config.radii = config.radii.clone();
    frst_config.polarity = config.polarity;
    let response = frst_response(&gradient, &frst_config)?;

    let proposals = extract_proposals(&response, &config.advanced.nms, config.polarity);

    let mut accepted: Vec<(CircleDetection, SupportScoreBreakdown)> = Vec::new();
    let mut rejected: Vec<RejectedProposal> = Vec::new();

    for proposal in &proposals {
        let circle = Circle::new(proposal.seed.position, config.radius_hint);
        let breakdown = score_circle_support(&gradient, &circle, &config.advanced.scoring);

        if breakdown.is_degenerate {
            rejected.push(RejectedProposal {
                proposal: proposal.clone(),
                reason: RejectionReason::Degenerate,
                score: breakdown,
            });
            continue;
        }
        if breakdown.total < config.min_score {
            rejected.push(RejectedProposal {
                proposal: proposal.clone(),
                reason: RejectionReason::LowScore,
                score: breakdown,
            });
            continue;
        }

        match refine_circle(&gradient, &circle, &config.advanced.refinement) {
            Ok(refined) => accepted.push((
                Detection {
                    hypothesis: refined.hypothesis,
                    score: SupportScore {
                        total: breakdown.total,
                    },
                    status: refined.status,
                },
                breakdown,
            )),
            Err(_) => rejected.push(RejectedProposal {
                proposal: proposal.clone(),
                reason: RejectionReason::RefinementFailed,
                score: breakdown,
            }),
        }
    }

    // Sort accepted detections by descending support score, keeping each score
    // breakdown aligned with its detection.
    accepted.sort_by(|a, b| {
        b.0.score
            .total
            .partial_cmp(&a.0.score.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (detections, score_breakdowns): (Vec<CircleDetection>, Vec<SupportScoreBreakdown>) =
        accepted.into_iter().unzip();

    let diagnostics = CircleDetectionDiagnostics {
        response,
        proposals,
        rejected,
        score_breakdowns,
    };

    Ok((detections, diagnostics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_circles_finds_synthetic_disk() {
        let size = 128;
        let cx = 64.0f32;
        let cy = 64.0f32;
        let radius = 18.0f32;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() <= radius {
                    data[y * size + x] = 255;
                }
            }
        }
        let image = ImageView::from_slice(&data, size, size).unwrap();

        let config = DetectCirclesConfig {
            radii: vec![17, 18, 19],
            polarity: Polarity::Bright,
            radius_hint: radius,
            advanced: DetectCirclesAdvanced {
                frst: FrstConfig {
                    gradient_threshold: 1.0,
                    ..FrstConfig::default()
                },
                ..DetectCirclesAdvanced::default()
            },
            ..DetectCirclesConfig::default()
        };

        let detections = detect_circles(&image, &config).unwrap();
        assert!(!detections.is_empty(), "should detect the synthetic disk");

        let best = &detections[0];
        let dx = best.hypothesis.center.x - cx;
        let dy = best.hypothesis.center.y - cy;
        assert!(
            (dx * dx + dy * dy).sqrt() < 3.0,
            "center should be near ({cx}, {cy}), got ({}, {})",
            best.hypothesis.center.x,
            best.hypothesis.center.y,
        );
    }

    #[test]
    fn builder_chain_sets_expected_fields() {
        let config = DetectCirclesConfig::for_radii([9, 10, 11])
            .polarity(Polarity::Bright)
            .radius_hint(12.5)
            .min_score(0.2)
            .gradient_operator(GradientOperator::Scharr);

        // for_radii sets the top-level voting radii.
        assert_eq!(config.radii, vec![9, 10, 11]);
        // polarity sets both the top-level and the nested FRST field.
        assert_eq!(config.polarity, Polarity::Bright);
        assert_eq!(config.advanced.frst.polarity, Polarity::Bright);
        // remaining chainable setters.
        assert_eq!(config.radius_hint, 12.5);
        assert_eq!(config.min_score, 0.2);
        assert_eq!(config.gradient_operator, GradientOperator::Scharr);

        // Untouched fields fall back to the defaults.
        let defaults = DetectCirclesConfig::default();
        assert_eq!(config.advanced.nms.radius, defaults.advanced.nms.radius);
        assert_eq!(
            config.advanced.refinement.max_iterations,
            defaults.advanced.refinement.max_iterations
        );
    }

    #[test]
    fn invalid_refinement_config_returns_error() {
        let size = 64;
        let data = vec![128u8; size * size];
        let image = ImageView::from_slice(&data, size, size).unwrap();

        let config = DetectCirclesConfig {
            advanced: DetectCirclesAdvanced {
                refinement: CircleRefineConfig {
                    max_iterations: 0,
                    ..CircleRefineConfig::default()
                },
                ..DetectCirclesAdvanced::default()
            },
            ..DetectCirclesConfig::default()
        };

        let result = detect_circles(&image, &config);
        assert!(
            matches!(
                result,
                Err(crate::core::error::RadSymError::InvalidConfig { .. })
            ),
            "expected InvalidConfig error, got {result:?}"
        );
    }

    #[test]
    fn detect_circles_with_diagnostics_matches_detect_circles() {
        let size = 128;
        let cx = 64.0f32;
        let cy = 64.0f32;
        let radius = 18.0f32;
        let mut data = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() <= radius {
                    data[y * size + x] = 255;
                }
            }
        }
        let image = ImageView::from_slice(&data, size, size).unwrap();

        let config = DetectCirclesConfig {
            radii: vec![17, 18, 19],
            polarity: Polarity::Bright,
            radius_hint: radius,
            advanced: DetectCirclesAdvanced {
                frst: FrstConfig {
                    gradient_threshold: 1.0,
                    ..FrstConfig::default()
                },
                ..DetectCirclesAdvanced::default()
            },
            ..DetectCirclesConfig::default()
        };

        let plain = detect_circles(&image, &config).unwrap();
        let (detailed, diagnostics) = detect_circles_with_diagnostics(&image, &config).unwrap();

        // detect_circles and the diagnostics variant agree on the detections.
        assert_eq!(plain.len(), detailed.len());
        for (a, b) in plain.iter().zip(&detailed) {
            assert_eq!(a.hypothesis.center.x, b.hypothesis.center.x);
            assert_eq!(a.hypothesis.center.y, b.hypothesis.center.y);
            assert_eq!(a.hypothesis.radius, b.hypothesis.radius);
            assert_eq!(a.score.total, b.score.total);
        }

        // Score breakdowns are index-aligned with the detections.
        assert_eq!(detailed.len(), diagnostics.score_breakdowns.len());
        for (det, breakdown) in detailed.iter().zip(&diagnostics.score_breakdowns) {
            assert_eq!(det.score.total, breakdown.total);
        }

        // The diagnostics expose the proposals that fed the pipeline.
        assert!(!diagnostics.proposals.is_empty());
    }
}
