//! Diagnostics for the circle-detection pipeline.
//!
//! These types carry evidence about *how* a detection result was produced —
//! the response map, the raw proposals, the rejected candidates, and the
//! per-detection score breakdowns. They are produced by
//! [`detect_circles_with_diagnostics`](crate::pipeline::detect_circles_with_diagnostics)
//! and carry a deliberately looser stability promise than the primary
//! [`CircleDetection`](crate::pipeline::CircleDetection) result: they exist for
//! debugging, parameter tuning, and visualization.

use crate::propose::extract::ResponseMap;
use crate::propose::seed::Proposal;
use crate::support::score::SupportScoreBreakdown;

/// Why a center proposal did not become an accepted detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum RejectionReason {
    /// Support evidence was degenerate (too few gradient samples).
    Degenerate,
    /// Support score was below the configured `min_score`.
    LowScore,
    /// Circle refinement returned an error for this proposal.
    RefinementFailed,
}

/// A center proposal the pipeline evaluated but did not accept.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RejectedProposal {
    /// The proposal that was rejected.
    pub proposal: Proposal,
    /// Why the proposal was rejected.
    pub reason: RejectionReason,
    /// Support score breakdown computed for the proposal before it was
    /// rejected.
    ///
    /// A rejection diagnostic carries the full evidence breakdown rather than
    /// the compact [`SupportScore`](crate::support::score::SupportScore) so the
    /// individual components explain why the proposal was dropped.
    pub score: SupportScoreBreakdown,
}

/// Diagnostic evidence from a circle-detection run.
///
/// Produced alongside the primary `Vec<CircleDetection>` result by
/// [`detect_circles_with_diagnostics`](crate::pipeline::detect_circles_with_diagnostics).
/// Unlike the result, this struct carries a looser stability promise: it exists
/// for debugging, parameter tuning, and visualization, and its shape may evolve.
#[derive(Debug)]
#[non_exhaustive]
pub struct CircleDetectionDiagnostics {
    /// The response map that proposals were extracted from.
    pub response: ResponseMap,
    /// Every center proposal extracted from the response map, before scoring.
    pub proposals: Vec<Proposal>,
    /// Proposals that were evaluated but did not become accepted detections.
    pub rejected: Vec<RejectedProposal>,
    /// Score breakdown for each accepted detection.
    ///
    /// Index-aligned with the `Vec<CircleDetection>` returned alongside this
    /// struct: `score_breakdowns[i]` describes the detection at index `i`.
    pub score_breakdowns: Vec<SupportScoreBreakdown>,
}
