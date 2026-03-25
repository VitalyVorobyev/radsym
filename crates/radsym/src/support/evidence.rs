//! Support evidence types.
//!
//! [`SupportEvidence`] holds the raw gradient samples and summary statistics
//! extracted from an annular region around a hypothesis.

use crate::core::coords::PixelCoord;
use crate::core::scalar::Scalar;

/// A single gradient sample with its radial alignment score.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientSample {
    /// Sample position in image coordinates.
    pub position: PixelCoord,
    /// Horizontal gradient component.
    pub gx: Scalar,
    /// Vertical gradient component.
    pub gy: Scalar,
    /// Absolute cosine between gradient direction and the radial direction
    /// from the hypothesis center. Range `[0, 1]`: 1 = perfectly radial.
    pub radial_alignment: Scalar,
}

impl GradientSample {
    /// Gradient magnitude.
    #[inline]
    pub fn magnitude(&self) -> Scalar {
        (self.gx * self.gx + self.gy * self.gy).sqrt()
    }
}

/// Extracted local evidence around a hypothesis.
///
/// Produced by annulus sampling functions and consumed by scoring functions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SupportEvidence {
    /// Individual gradient samples with alignment scores.
    pub gradient_samples: Vec<GradientSample>,
    /// Fraction of the annulus that has well-aligned gradient support.
    /// Range `[0, 1]`.
    pub angular_coverage: Scalar,
    /// Number of valid samples (with non-negligible gradient).
    pub sample_count: usize,
    /// Mean absolute radial alignment across all samples. Range `[0, 1]`.
    pub mean_gradient_alignment: Scalar,
}
