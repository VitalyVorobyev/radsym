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
use crate::propose::extract::extract_proposals;
use crate::propose::frst::{FrstConfig, frst_response};
use crate::refine::circle::{CircleRefineConfig, refine_circle};
use crate::refine::result::RefinementStatus;
use crate::support::score::{ScoringConfig, SupportScore, score_circle_support};

/// Aggregated configuration for [`detect_circles`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectCirclesConfig {
    /// FRST voting configuration.
    pub frst: FrstConfig,
    /// Non-maximum suppression for proposal extraction.
    pub nms: NmsConfig,
    /// Support scoring configuration.
    pub scoring: ScoringConfig,
    /// Iterative circle refinement configuration.
    pub refinement: CircleRefineConfig,
    /// Which polarity to detect.
    pub polarity: Polarity,
    /// Approximate expected radius used as the initial circle hypothesis.
    pub radius_hint: Scalar,
    /// Minimum support score to keep a detection (in `[0, 1]`).
    pub min_score: Scalar,
    /// Gradient operator to use (default: Sobel).
    pub gradient_operator: GradientOperator,
}

impl Default for DetectCirclesConfig {
    fn default() -> Self {
        Self {
            frst: FrstConfig::default(),
            nms: NmsConfig::default(),
            scoring: ScoringConfig::default(),
            refinement: CircleRefineConfig::default(),
            polarity: Polarity::Both,
            radius_hint: 10.0,
            min_score: 0.0,
            gradient_operator: GradientOperator::default(),
        }
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
pub struct Detection<T> {
    /// Refined geometric hypothesis.
    pub hypothesis: T,
    /// Support score from gradient evidence.
    pub score: SupportScore,
    /// Refinement convergence status.
    pub status: RefinementStatus,
}

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
/// use radsym::{ImageView, FrstConfig, Polarity};
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
/// let config = DetectCirclesConfig {
///     frst: FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() },
///     polarity: Polarity::Bright,
///     radius_hint: 10.0,
///     ..DetectCirclesConfig::default()
/// };
///
/// let detections = detect_circles(&image, &config).unwrap();
/// assert!(!detections.is_empty());
/// ```
pub fn detect_circles(
    image: &ImageView<'_, u8>,
    config: &DetectCirclesConfig,
) -> Result<Vec<Detection<Circle>>> {
    config.refinement.validate()?;

    let gradient = compute_gradient(image, config.gradient_operator)?;

    let mut frst_config = config.frst.clone();
    frst_config.polarity = config.polarity;
    let response = frst_response(&gradient, &frst_config)?;

    let proposals = extract_proposals(&response, &config.nms, config.polarity);

    let mut detections: Vec<Detection<Circle>> = proposals
        .iter()
        .filter_map(|proposal| {
            let circle = Circle::new(proposal.seed.position, config.radius_hint);
            let score = score_circle_support(&gradient, &circle, &config.scoring);
            if score.is_degenerate || score.total < config.min_score {
                return None;
            }
            let refined = refine_circle(&gradient, &circle, &config.refinement).ok()?;
            Some(Detection {
                hypothesis: refined.hypothesis,
                score,
                status: refined.status,
            })
        })
        .collect();

    detections.sort_by(|a, b| {
        b.score
            .total
            .partial_cmp(&a.score.total)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(detections)
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
            frst: FrstConfig {
                radii: vec![17, 18, 19],
                gradient_threshold: 1.0,
                ..FrstConfig::default()
            },
            polarity: Polarity::Bright,
            radius_hint: radius,
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
    fn invalid_refinement_config_returns_error() {
        let size = 64;
        let data = vec![128u8; size * size];
        let image = ImageView::from_slice(&data, size, size).unwrap();

        let config = DetectCirclesConfig {
            refinement: CircleRefineConfig {
                max_iterations: 0,
                ..CircleRefineConfig::default()
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
}
