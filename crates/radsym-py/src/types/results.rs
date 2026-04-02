#[pyclass(name = "SupportScore")]
pub struct PySupportScore {
    pub inner: radsym::SupportScore,
}

#[pymethods]
impl PySupportScore {
    /// Combined total score in [0, 1].
    #[getter]
    fn total(&self) -> f32 {
        self.inner.total
    }

    /// Gradient alignment strength.
    #[getter]
    fn ringness(&self) -> f32 {
        self.inner.ringness
    }

    /// Angular coverage fraction in [0, 1].
    #[getter]
    fn angular_coverage(&self) -> f32 {
        self.inner.angular_coverage
    }

    /// True if evidence is degenerate (insufficient samples).
    #[getter]
    fn is_degenerate(&self) -> bool {
        self.inner.is_degenerate
    }

    fn __repr__(&self) -> String {
        format!(
            "SupportScore(total={:.3}, ringness={:.3}, coverage={:.3})",
            self.inner.total, self.inner.ringness, self.inner.angular_coverage
        )
    }
}

/// A detected center proposal from FRST or RSD voting.
///
/// Attributes:
///     position (tuple[float, float]): Detected center (x, y) in pixels.
///     score (float): Response strength at this location.
///     scale_hint (float | None): Approximate radius if available.
#[pyclass(name = "Proposal")]
pub struct PyProposal {
    pub inner: radsym::Proposal,
}

#[pymethods]
impl PyProposal {
    /// Detected center coordinates (x, y) in pixels.
    #[getter]
    fn position(&self) -> (f32, f32) {
        (self.inner.seed.position.x, self.inner.seed.position.y)
    }

    /// Response strength at this location.
    #[getter]
    fn score(&self) -> f32 {
        self.inner.seed.score
    }

    /// Approximate radius hint, if available.
    #[getter]
    fn scale_hint(&self) -> Option<f32> {
        self.inner.scale_hint
    }

    fn __repr__(&self) -> String {
        format!(
            "Proposal(position=({:.1}, {:.1}), score={:.3})",
            self.inner.seed.position.x, self.inner.seed.position.y, self.inner.seed.score
        )
    }
}

/// Proposal extracted from a rectified homography-aware response map.
#[pyclass(name = "HomographyProposal")]
pub struct PyHomographyProposal {
    pub inner: radsym::HomographyProposal,
}

#[pymethods]
impl PyHomographyProposal {
    #[getter]
    fn rectified_position(&self) -> (f32, f32) {
        (
            self.inner.rectified_seed.position.x,
            self.inner.rectified_seed.position.y,
        )
    }

    #[getter]
    fn rectified_score(&self) -> f32 {
        self.inner.rectified_seed.score
    }

    #[getter]
    fn rectified_circle_hint(&self) -> Option<PyCircle> {
        self.inner
            .rectified_circle_hint
            .map(|circle| PyCircle { inner: circle })
    }

    #[getter]
    fn image_ellipse_hint(&self) -> Option<PyEllipse> {
        self.inner
            .image_ellipse_hint
            .map(|ellipse| PyEllipse { inner: ellipse })
    }

    fn __repr__(&self) -> String {
        format!(
            "HomographyProposal(rectified_position=({:.1}, {:.1}), score={:.3})",
            self.inner.rectified_seed.position.x,
            self.inner.rectified_seed.position.y,
            self.inner.rectified_seed.score
        )
    }
}

/// Homography-aware reranked proposal.
#[pyclass(name = "RerankedProposal")]
pub struct PyRerankedProposal {
    pub inner: radsym::RerankedProposal,
}

#[pymethods]
impl PyRerankedProposal {
    #[getter]
    fn proposal(&self) -> PyProposal {
        PyProposal {
            inner: self.inner.proposal.clone(),
        }
    }

    #[getter]
    fn image_ellipse_hint(&self) -> Option<PyEllipse> {
        self.inner
            .image_ellipse_hint
            .map(|ellipse| PyEllipse { inner: ellipse })
    }

    #[getter]
    fn rectified_circle_hint(&self) -> Option<PyCircle> {
        self.inner
            .rectified_circle_hint
            .map(|circle| PyCircle { inner: circle })
    }

    #[getter]
    fn rectified_edge_score(&self) -> f32 {
        self.inner.rectified_edge_score
    }

    #[getter]
    fn rectified_coverage(&self) -> f32 {
        self.inner.rectified_coverage
    }

    #[getter]
    fn size_prior(&self) -> f32 {
        self.inner.size_prior
    }

    #[getter]
    fn center_prior(&self) -> f32 {
        self.inner.center_prior
    }

    #[getter]
    fn total_score(&self) -> f32 {
        self.inner.total_score
    }

    fn __repr__(&self) -> String {
        format!(
            "RerankedProposal(position=({:.1}, {:.1}), total_score={:.3})",
            self.inner.proposal.seed.position.x,
            self.inner.proposal.seed.position.y,
            self.inner.total_score
        )
    }
}

/// Result of iterative circle refinement.
///
/// Attributes:
///     hypothesis (Circle): The refined circle.
///     status (str): Convergence status — ``"converged"``, ``"max_iterations"``,
///         ``"degenerate"``, or ``"out_of_bounds"``.
///     residual (float): Final residual (center shift at last iteration).
///     iterations (int): Number of iterations performed.
///     converged (bool): True if status is ``"converged"``.
#[pyclass(name = "CircleRefinementResult")]
pub struct PyCircleRefinementResult {
    pub circle: radsym::Circle,
    pub status: radsym::RefinementStatus,
    pub residual: f32,
    pub iterations: usize,
}

#[pymethods]
impl PyCircleRefinementResult {
    /// The refined circle hypothesis.
    #[getter]
    fn hypothesis(&self) -> PyCircle {
        PyCircle { inner: self.circle }
    }

    /// Convergence status string.
    #[getter]
    fn status(&self) -> &'static str {
        status_to_str(&self.status)
    }

    /// Final residual (center shift at last iteration).
    #[getter]
    fn residual(&self) -> f32 {
        self.residual
    }

    /// Number of iterations performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// True if refinement converged.
    #[getter]
    fn converged(&self) -> bool {
        matches!(self.status, radsym::RefinementStatus::Converged)
    }

    fn __repr__(&self) -> String {
        format!(
            "CircleRefinementResult(status='{}', iterations={}, residual={:.4})",
            status_to_str(&self.status),
            self.iterations,
            self.residual,
        )
    }
}

/// Result of iterative ellipse refinement.
///
/// Attributes:
///     hypothesis (Ellipse): The refined ellipse.
///     status (str): Convergence status.
///     residual (float): Final residual.
///     iterations (int): Number of iterations performed.
///     converged (bool): True if status is ``"converged"``.
#[pyclass(name = "EllipseRefinementResult")]
pub struct PyEllipseRefinementResult {
    pub ellipse: radsym::Ellipse,
    pub status: radsym::RefinementStatus,
    pub residual: f32,
    pub iterations: usize,
}

#[pymethods]
impl PyEllipseRefinementResult {
    /// The refined ellipse hypothesis.
    #[getter]
    fn hypothesis(&self) -> PyEllipse {
        PyEllipse {
            inner: self.ellipse,
        }
    }

    /// Convergence status string.
    #[getter]
    fn status(&self) -> &'static str {
        status_to_str(&self.status)
    }

    /// Final residual.
    #[getter]
    fn residual(&self) -> f32 {
        self.residual
    }

    /// Number of iterations performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// True if refinement converged.
    #[getter]
    fn converged(&self) -> bool {
        matches!(self.status, radsym::RefinementStatus::Converged)
    }

    fn __repr__(&self) -> String {
        format!(
            "EllipseRefinementResult(status='{}', iterations={}, residual={:.4})",
            status_to_str(&self.status),
            self.iterations,
            self.residual,
        )
    }
}

/// Result of homography-aware ellipse refinement.
#[pyclass(name = "HomographyRefinementResult")]
pub struct PyHomographyRefinementResult {
    pub inner: radsym::HomographyRefinementResult,
}

#[pymethods]
impl PyHomographyRefinementResult {
    #[getter]
    fn image_ellipse(&self) -> PyEllipse {
        PyEllipse {
            inner: self.inner.image_ellipse,
        }
    }

    #[getter]
    fn rectified_circle(&self) -> PyCircle {
        PyCircle {
            inner: self.inner.rectified_circle,
        }
    }

    #[getter]
    fn status(&self) -> &'static str {
        status_to_str(&self.inner.status)
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn image_residual(&self) -> f32 {
        self.inner.image_residual
    }

    #[getter]
    fn rectified_residual(&self) -> f32 {
        self.inner.rectified_residual
    }

    #[getter]
    fn inlier_coverage(&self) -> f32 {
        self.inner.inlier_coverage
    }

    #[getter]
    fn converged(&self) -> bool {
        matches!(self.inner.status, radsym::RefinementStatus::Converged)
    }

    fn __repr__(&self) -> String {
        format!(
            "HomographyRefinementResult(status='{}', iterations={}, rectified_residual={:.4})",
            status_to_str(&self.inner.status),
            self.inner.iterations,
            self.inner.rectified_residual,
        )
    }
}

/// Result of Parthasarathy radial center refinement.
///
/// Attributes:
///     hypothesis (tuple[float, float]): Refined center (x, y) in pixels.
///     status (str): Convergence status.
///     residual (float): Final residual.
///     iterations (int): Number of iterations (always 1 for radial center).
///     converged (bool): True if status is ``"converged"``.
#[pyclass(name = "PointRefinementResult")]
pub struct PyPointRefinementResult {
    pub point: (f32, f32),
    pub status: radsym::RefinementStatus,
    pub residual: f32,
    pub iterations: usize,
}

#[pymethods]
impl PyPointRefinementResult {
    /// Refined center coordinates (x, y) in pixels.
    #[getter]
    fn hypothesis(&self) -> (f32, f32) {
        self.point
    }

    /// Convergence status string.
    #[getter]
    fn status(&self) -> &'static str {
        status_to_str(&self.status)
    }

    /// Final residual.
    #[getter]
    fn residual(&self) -> f32 {
        self.residual
    }

    /// Number of iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// True if refinement converged.
    #[getter]
    fn converged(&self) -> bool {
        matches!(self.status, radsym::RefinementStatus::Converged)
    }

    fn __repr__(&self) -> String {
        format!(
            "PointRefinementResult(hypothesis=({:.3}, {:.3}), status='{}')",
            self.point.0,
            self.point.1,
            status_to_str(&self.status),
        )
    }
}
/// A detected circle from the one-call :func:`detect_circles` pipeline.
///
/// Attributes:
///     hypothesis (Circle): Refined circle hypothesis.
///     score (SupportScore): Gradient evidence score.
///     status (str): Refinement convergence status.
///     converged (bool): True if refinement converged.
#[pyclass(name = "Detection")]
pub struct PyDetection {
    pub inner: radsym::Detection<radsym::Circle>,
}

#[pymethods]
impl PyDetection {
    /// Refined circle hypothesis.
    #[getter]
    fn hypothesis(&self) -> PyCircle {
        PyCircle {
            inner: self.inner.hypothesis,
        }
    }

    /// Support score from gradient evidence.
    #[getter]
    fn score(&self) -> PySupportScore {
        PySupportScore {
            inner: self.inner.score,
        }
    }

    /// Convergence status string.
    #[getter]
    fn status(&self) -> &'static str {
        status_to_str(&self.inner.status)
    }

    /// True if refinement converged.
    #[getter]
    fn converged(&self) -> bool {
        matches!(self.inner.status, radsym::RefinementStatus::Converged)
    }

    fn __repr__(&self) -> String {
        format!(
            "Detection(center=({:.1}, {:.1}), radius={:.1}, score={:.3}, status='{}')",
            self.inner.hypothesis.center.x,
            self.inner.hypothesis.center.y,
            self.inner.hypothesis.radius,
            self.inner.score.total,
            status_to_str(&self.inner.status),
        )
    }
}

use pyo3::prelude::*;

use crate::convert::status_to_str;
use crate::types::{PyCircle, PyEllipse};
