//! Seed and proposal types for center-proposal generation.

use crate::core::coords::PixelCoord;
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;

/// A candidate center location with a response score.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeedPoint {
    /// Position in pixel coordinates (x = col, y = row).
    pub position: PixelCoord,
    /// Response score (higher = stronger radial symmetry evidence).
    pub score: Scalar,
}

/// Source algorithm that generated a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ProposalSource {
    /// Fast Radial Symmetry Transform (Loy & Zelinsky 2002).
    Frst,
    /// Radial Symmetry Detector fast variant (Barnes et al. 2008).
    Rsd,
    /// Externally provided seed (e.g. from a downstream application).
    External,
}

/// A center-proposal with metadata.
///
/// Proposals are the primary output of the `propose` module. They represent
/// candidate center locations for downstream support analysis and refinement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proposal {
    /// The seed point (position + score).
    pub seed: SeedPoint,
    /// Approximate radius hint, if available from the response map.
    pub scale_hint: Option<Scalar>,
    /// Which polarity mode produced this proposal.
    pub polarity: Polarity,
    /// Which algorithm generated this proposal.
    pub source: ProposalSource,
}
