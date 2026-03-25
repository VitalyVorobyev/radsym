//! Polarity modes for radial symmetry detection.

/// Controls which gradient polarity is used during voting and scoring.
///
/// - `Bright`: detect bright structures on dark backgrounds (gradient points inward).
/// - `Dark`: detect dark structures on bright backgrounds (gradient points outward).
/// - `Both`: detect either polarity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Polarity {
    /// Bright center on dark surround.
    Bright,
    /// Dark center on bright surround.
    Dark,
    /// Detect either polarity.
    #[default]
    Both,
}

impl Polarity {
    /// Whether this polarity votes for bright (positive-affected) pixels.
    #[inline]
    pub fn votes_positive(&self) -> bool {
        matches!(self, Polarity::Bright | Polarity::Both)
    }

    /// Whether this polarity votes for dark (negative-affected) pixels.
    #[inline]
    pub fn votes_negative(&self) -> bool {
        matches!(self, Polarity::Dark | Polarity::Both)
    }
}
