//! Refinement result types.
//!
//! [`RefinementResult`] wraps a refined hypothesis with convergence status,
//! residual information, and optional uncertainty estimate.

use crate::core::scalar::Scalar;

/// Status of a refinement procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementStatus {
    /// Converged within the requested tolerance.
    Converged,
    /// Hit the maximum iteration count without converging.
    MaxIterations,
    /// The problem is degenerate (e.g., insufficient gradient evidence).
    Degenerate,
    /// The refined hypothesis moved outside the valid image region.
    OutOfBounds,
}

/// Result of a local refinement procedure.
#[derive(Debug, Clone)]
pub struct RefinementResult<H> {
    /// The refined hypothesis.
    pub hypothesis: H,
    /// Convergence status.
    pub status: RefinementStatus,
    /// RMS residual of the final fit (meaning depends on the algorithm).
    pub residual: Scalar,
    /// Number of iterations performed.
    pub iterations: usize,
}

impl<H> RefinementResult<H> {
    /// Whether refinement converged successfully.
    #[inline]
    pub fn converged(&self) -> bool {
        self.status == RefinementStatus::Converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::coords::PixelCoord;

    #[test]
    fn converged_result() {
        let result = RefinementResult {
            hypothesis: PixelCoord::new(10.5, 20.3),
            status: RefinementStatus::Converged,
            residual: 0.01,
            iterations: 3,
        };
        assert!(result.converged());
    }

    #[test]
    fn degenerate_result() {
        let result = RefinementResult {
            hypothesis: PixelCoord::new(0.0, 0.0),
            status: RefinementStatus::Degenerate,
            residual: 0.0,
            iterations: 0,
        };
        assert!(!result.converged());
    }
}
