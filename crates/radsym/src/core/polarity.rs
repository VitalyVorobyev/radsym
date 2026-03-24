//! Polarity modes for radial symmetry detection.

/// Controls which gradient polarity is used during voting and scoring.
///
/// - `Bright`: detect bright structures on dark backgrounds (gradient points inward).
/// - `Dark`: detect dark structures on bright backgrounds (gradient points outward).
/// - `Both`: detect either polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// Bright center on dark surround.
    Bright,
    /// Dark center on bright surround.
    Dark,
    /// Detect either polarity.
    Both,
}
