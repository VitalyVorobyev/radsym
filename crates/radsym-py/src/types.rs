use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::convert::{polarity_from_str, polarity_to_str, status_to_str};
use crate::error::to_pyerr;

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// A circle defined by center (x, y) and radius.
///
/// Attributes:
///     center (tuple[float, float]): Center coordinates (x, y) in pixels.
///     radius (float): Circle radius in pixels.
///
/// Example::
///
///     c = radsym.Circle((100.0, 200.0), 15.0)
///     print(c.center, c.radius)
#[pyclass(name = "Circle")]
#[derive(Clone)]
pub struct PyCircle {
    pub inner: radsym::Circle,
}

#[pymethods]
impl PyCircle {
    #[new]
    #[pyo3(signature = (center, radius))]
    fn new(center: (f32, f32), radius: f32) -> Self {
        Self {
            inner: radsym::Circle::new(radsym::PixelCoord::new(center.0, center.1), radius),
        }
    }

    /// Center coordinates (x, y) in pixels.
    #[getter]
    fn center(&self) -> (f32, f32) {
        (self.inner.center.x, self.inner.center.y)
    }

    /// Circle radius in pixels.
    #[getter]
    fn radius(&self) -> f32 {
        self.inner.radius
    }

    fn __repr__(&self) -> String {
        format!(
            "Circle(center=({:.1}, {:.1}), radius={:.2})",
            self.inner.center.x, self.inner.center.y, self.inner.radius
        )
    }

    fn __eq__(&self, other: &PyCircle) -> bool {
        self.inner.center == other.inner.center && self.inner.radius == other.inner.radius
    }
}

/// An ellipse defined by center, semi-axes, and orientation angle.
///
/// Attributes:
///     center (tuple[float, float]): Center coordinates (x, y) in pixels.
///     semi_major (float): Semi-major axis length in pixels.
///     semi_minor (float): Semi-minor axis length in pixels.
///     angle (float): Orientation angle in radians.
///
/// Example::
///
///     e = radsym.Ellipse((100.0, 200.0), 20.0, 10.0, angle=0.5)
#[pyclass(name = "Ellipse")]
#[derive(Clone)]
pub struct PyEllipse {
    pub inner: radsym::Ellipse,
}

#[pymethods]
impl PyEllipse {
    #[new]
    #[pyo3(signature = (center, semi_major, semi_minor, angle=0.0))]
    fn new(center: (f32, f32), semi_major: f32, semi_minor: f32, angle: f32) -> Self {
        Self {
            inner: radsym::Ellipse::new(
                radsym::PixelCoord::new(center.0, center.1),
                semi_major,
                semi_minor,
                angle,
            ),
        }
    }

    /// Center coordinates (x, y) in pixels.
    #[getter]
    fn center(&self) -> (f32, f32) {
        (self.inner.center.x, self.inner.center.y)
    }

    /// Semi-major axis length in pixels.
    #[getter]
    fn semi_major(&self) -> f32 {
        self.inner.semi_major
    }

    /// Semi-minor axis length in pixels.
    #[getter]
    fn semi_minor(&self) -> f32 {
        self.inner.semi_minor
    }

    /// Orientation angle in radians.
    #[getter]
    fn angle(&self) -> f32 {
        self.inner.angle
    }

    fn __repr__(&self) -> String {
        format!(
            "Ellipse(center=({:.1}, {:.1}), a={:.2}, b={:.2}, angle={:.3})",
            self.inner.center.x,
            self.inner.center.y,
            self.inner.semi_major,
            self.inner.semi_minor,
            self.inner.angle,
        )
    }
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

/// Configuration for FRST (Fast Radial Symmetry Transform) response computation.
///
/// Args:
///     radii: List of discrete radii to test, in pixels. Default: [3, 5, 7, 9, 11].
///     alpha: Radial strictness exponent. Higher values require more consistent
///         orientation evidence. Default: 2.0.
///     gradient_threshold: Minimum gradient magnitude to participate in voting.
///         Pixels below this threshold are skipped. Default: 0.0.
///     polarity: Which polarity to detect — ``"bright"``, ``"dark"``, or ``"both"``.
///         Default: ``"both"``.
///     smoothing_factor: Gaussian smoothing sigma relative to radius.
///         Default: 0.5.
///
/// Example::
///
///     config = radsym.FrstConfig(radii=[8, 10, 12], polarity="dark")
#[pyclass(name = "FrstConfig")]
#[derive(Clone)]
pub struct PyFrstConfig {
    pub inner: radsym::FrstConfig,
}

#[pymethods]
impl PyFrstConfig {
    #[new]
    #[pyo3(signature = (radii=None, alpha=2.0, gradient_threshold=0.0, polarity="both", smoothing_factor=0.5))]
    fn new(
        radii: Option<Vec<u32>>,
        alpha: f32,
        gradient_threshold: f32,
        polarity: &str,
        smoothing_factor: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: radsym::FrstConfig {
                radii: radii.unwrap_or_else(|| vec![3, 5, 7, 9, 11]),
                alpha,
                gradient_threshold,
                polarity: polarity_from_str(polarity)?,
                smoothing_factor,
            },
        })
    }

    /// List of discrete radii to test, in pixels.
    #[getter]
    fn radii(&self) -> Vec<u32> {
        self.inner.radii.clone()
    }

    /// Radial strictness exponent.
    #[getter]
    fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    /// Minimum gradient magnitude threshold.
    #[getter]
    fn gradient_threshold(&self) -> f32 {
        self.inner.gradient_threshold
    }

    /// Polarity mode: ``"bright"``, ``"dark"``, or ``"both"``.
    #[getter]
    fn polarity(&self) -> &'static str {
        polarity_to_str(self.inner.polarity)
    }

    /// Gaussian smoothing factor (sigma = factor * radius).
    #[getter]
    fn smoothing_factor(&self) -> f32 {
        self.inner.smoothing_factor
    }

    fn __repr__(&self) -> String {
        format!(
            "FrstConfig(radii={:?}, alpha={}, polarity='{}')",
            self.inner.radii,
            self.inner.alpha,
            polarity_to_str(self.inner.polarity),
        )
    }
}

/// Configuration for RSD (Radial Symmetry Detector) response computation.
///
/// Args:
///     radii: List of discrete radii to test, in pixels. Default: [3, 5, 7, 9, 11].
///     gradient_threshold: Minimum gradient magnitude. Default: 0.0.
///     polarity: ``"bright"``, ``"dark"``, or ``"both"``. Default: ``"both"``.
///     smoothing_factor: Gaussian smoothing factor. Default: 0.5.
#[pyclass(name = "RsdConfig")]
#[derive(Clone)]
pub struct PyRsdConfig {
    pub inner: radsym::RsdConfig,
}

#[pymethods]
impl PyRsdConfig {
    #[new]
    #[pyo3(signature = (radii=None, gradient_threshold=0.0, polarity="both", smoothing_factor=0.5))]
    fn new(
        radii: Option<Vec<u32>>,
        gradient_threshold: f32,
        polarity: &str,
        smoothing_factor: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: radsym::RsdConfig {
                radii: radii.unwrap_or_else(|| vec![3, 5, 7, 9, 11]),
                gradient_threshold,
                polarity: polarity_from_str(polarity)?,
                smoothing_factor,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RsdConfig(radii={:?}, polarity='{}')",
            self.inner.radii,
            polarity_to_str(self.inner.polarity),
        )
    }
}

/// Configuration for non-maximum suppression.
///
/// Args:
///     radius: Suppression radius in pixels (half-window size). Default: 5.
///     threshold: Minimum response value for a candidate peak. Default: 0.0.
///     max_detections: Maximum number of detections to return. Default: 1000.
///
/// Example::
///
///     nms = radsym.NmsConfig(radius=10, threshold=0.01, max_detections=50)
#[pyclass(name = "NmsConfig")]
#[derive(Clone)]
pub struct PyNmsConfig {
    pub inner: radsym::core::nms::NmsConfig,
}

#[pymethods]
impl PyNmsConfig {
    #[new]
    #[pyo3(signature = (radius=5, threshold=0.0, max_detections=1000))]
    fn new(radius: usize, threshold: f32, max_detections: usize) -> Self {
        Self {
            inner: radsym::core::nms::NmsConfig {
                radius,
                threshold,
                max_detections,
            },
        }
    }

    /// Suppression radius in pixels.
    #[getter]
    fn radius(&self) -> usize {
        self.inner.radius
    }

    /// Minimum response threshold.
    #[getter]
    fn threshold(&self) -> f32 {
        self.inner.threshold
    }

    /// Maximum number of detections.
    #[getter]
    fn max_detections(&self) -> usize {
        self.inner.max_detections
    }

    fn __repr__(&self) -> String {
        format!(
            "NmsConfig(radius={}, threshold={}, max_detections={})",
            self.inner.radius, self.inner.threshold, self.inner.max_detections,
        )
    }
}

/// Configuration for support scoring.
///
/// Args:
///     annulus_margin: Fractional width of the annular sampling region
///         around the hypothesized radius. Default: 0.3.
///     min_samples: Minimum gradient samples to avoid degeneracy. Default: 8.
///     weight_ringness: Weight of gradient alignment in total score. Default: 0.6.
///     weight_coverage: Weight of angular coverage in total score. Default: 0.4.
#[pyclass(name = "ScoringConfig")]
#[derive(Clone)]
pub struct PyScoringConfig {
    pub inner: radsym::ScoringConfig,
}

#[pymethods]
impl PyScoringConfig {
    #[new]
    #[pyo3(signature = (annulus_margin=0.3, min_samples=8, weight_ringness=0.6, weight_coverage=0.4))]
    fn new(
        annulus_margin: f32,
        min_samples: usize,
        weight_ringness: f32,
        weight_coverage: f32,
    ) -> Self {
        Self {
            inner: radsym::ScoringConfig {
                annulus_margin,
                min_samples,
                weight_ringness,
                weight_coverage,
                ..Default::default()
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoringConfig(annulus_margin={}, weights=({}, {}))",
            self.inner.annulus_margin, self.inner.weight_ringness, self.inner.weight_coverage,
        )
    }
}

/// Configuration for iterative circle refinement.
///
/// Args:
///     max_iterations: Maximum refinement iterations. Default: 10.
///     convergence_tol: Stop when center shift is below this (pixels). Default: 0.1.
///     annulus_margin: Fractional annulus margin around the radius. Default: 0.3.
#[pyclass(name = "CircleRefineConfig")]
#[derive(Clone)]
pub struct PyCircleRefineConfig {
    pub inner: radsym::CircleRefineConfig,
}

#[pymethods]
impl PyCircleRefineConfig {
    #[new]
    #[pyo3(signature = (max_iterations=10, convergence_tol=0.1, annulus_margin=0.3))]
    fn new(max_iterations: usize, convergence_tol: f32, annulus_margin: f32) -> Self {
        Self {
            inner: radsym::CircleRefineConfig {
                max_iterations,
                convergence_tol,
                annulus_margin,
                ..Default::default()
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CircleRefineConfig(max_iterations={}, convergence_tol={})",
            self.inner.max_iterations, self.inner.convergence_tol,
        )
    }
}

/// Configuration for iterative ellipse refinement.
///
/// Args:
///     max_iterations: Maximum refinement iterations. Default: 10.
///     convergence_tol: Stop when center shift is below this (pixels). Default: 0.1.
///     annulus_margin: Fractional annulus margin around the ellipse. Default: 0.3.
#[pyclass(name = "EllipseRefineConfig")]
#[derive(Clone)]
pub struct PyEllipseRefineConfig {
    pub inner: radsym::EllipseRefineConfig,
}

#[pymethods]
impl PyEllipseRefineConfig {
    #[new]
    #[pyo3(signature = (max_iterations=10, convergence_tol=0.1, annulus_margin=0.3))]
    fn new(max_iterations: usize, convergence_tol: f32, annulus_margin: f32) -> Self {
        Self {
            inner: radsym::EllipseRefineConfig {
                max_iterations,
                convergence_tol,
                annulus_margin,
                ..Default::default()
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EllipseRefineConfig(max_iterations={}, convergence_tol={})",
            self.inner.max_iterations, self.inner.convergence_tol,
        )
    }
}

/// Configuration for Parthasarathy radial center refinement.
///
/// Args:
///     patch_radius: Half-size of the local patch around the seed. Default: library default.
///     gradient_threshold: Minimum gradient magnitude in the patch. Default: library default.
///
/// Literature: Parthasarathy, Nature Methods 2012.
#[pyclass(name = "RadialCenterConfig")]
#[derive(Clone)]
pub struct PyRadialCenterConfig {
    pub inner: radsym::RadialCenterConfig,
}

#[pymethods]
impl PyRadialCenterConfig {
    #[new]
    #[pyo3(signature = (patch_radius=None, gradient_threshold=None))]
    fn new(patch_radius: Option<usize>, gradient_threshold: Option<f32>) -> Self {
        let mut inner = radsym::RadialCenterConfig::default();
        if let Some(r) = patch_radius {
            inner.patch_radius = r;
        }
        if let Some(t) = gradient_threshold {
            inner.gradient_threshold = t;
        }
        Self { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "RadialCenterConfig(patch_radius={}, gradient_threshold={})",
            self.inner.patch_radius, self.inner.gradient_threshold,
        )
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Structured support score quantifying local gradient evidence for a hypothesis.
///
/// Attributes:
///     total (float): Combined score in [0, 1].
///     ringness (float): Gradient alignment strength.
///     angular_coverage (float): Fraction of the annulus with gradient evidence, in [0, 1].
///     is_degenerate (bool): True if evidence is insufficient.
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

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Precomputed Sobel gradient field (gx, gy) for an image.
///
/// Created by :func:`sobel_gradient`. Passed to all downstream pipeline
/// functions (FRST, scoring, refinement).
///
/// Attributes:
///     width (int): Image width in pixels.
///     height (int): Image height in pixels.
///
/// Methods:
///     gx_numpy(): Returns the x-gradient component as a 2D float32 numpy array.
///     gy_numpy(): Returns the y-gradient component as a 2D float32 numpy array.
#[pyclass(name = "GradientField")]
pub struct PyGradientField {
    pub inner: radsym::core::gradient::GradientField,
}

#[pymethods]
impl PyGradientField {
    /// Image width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Image height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    /// Return the x-gradient component as a 2D float32 numpy array (H x W).
    fn gx_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let view = self.inner.gx();
        let w = view.width();
        let h = view.height();
        let arr = Array2::from_shape_fn((h, w), |(y, x)| *view.get(x, y).unwrap());
        PyArray2::from_owned_array(py, arr)
    }

    /// Return the y-gradient component as a 2D float32 numpy array (H x W).
    fn gy_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let view = self.inner.gy();
        let w = view.width();
        let h = view.height();
        let arr = Array2::from_shape_fn((h, w), |(y, x)| *view.get(x, y).unwrap());
        PyArray2::from_owned_array(py, arr)
    }

    fn __repr__(&self) -> String {
        format!(
            "GradientField({}x{})",
            self.inner.width(),
            self.inner.height()
        )
    }
}

/// FRST or RSD response accumulator map.
///
/// Created by :func:`frst_response` or :func:`rsd_response`.
/// Pass to :func:`extract_proposals` for center detection.
///
/// Attributes:
///     width (int): Map width in pixels.
///     height (int): Map height in pixels.
///
/// Methods:
///     to_numpy(): Returns the response values as a 2D float32 numpy array (H x W).
#[pyclass(name = "ResponseMap")]
pub struct PyResponseMap {
    pub inner: radsym::ResponseMap,
}

#[pymethods]
impl PyResponseMap {
    /// Return the response values as a 2D float32 numpy array (H x W).
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let response = self.inner.response();
        let w = response.width();
        let h = response.height();
        let data = response.data();
        let arr = Array2::from_shape_fn((h, w), |(y, x)| data[y * w + x]);
        PyArray2::from_owned_array(py, arr)
    }

    /// Map width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.inner.response().width()
    }

    /// Map height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.inner.response().height()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResponseMap({}x{})",
            self.inner.response().width(),
            self.inner.response().height()
        )
    }
}

/// RGBA diagnostic image for visualization.
///
/// Created by :func:`response_heatmap`. Can be drawn on with
/// :func:`overlay_circle` and :func:`overlay_ellipse`, then saved
/// with :func:`save_diagnostic`.
///
/// Attributes:
///     width (int): Image width in pixels.
///     height (int): Image height in pixels.
///
/// Methods:
///     to_numpy(): Returns pixel data as a uint8 numpy array (H x W x 4, RGBA).
#[pyclass(name = "DiagnosticImage")]
pub struct PyDiagnosticImage {
    pub inner: radsym::diagnostics::heatmap::DiagnosticImage,
}

#[pymethods]
impl PyDiagnosticImage {
    /// Image width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Image height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    /// Return pixel data as a uint8 numpy array with shape (H, W, 4) in RGBA order.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
        let w = self.inner.width();
        let h = self.inner.height();
        let data = self.inner.data();
        let arr =
            numpy::ndarray::Array3::from_shape_fn((h, w, 4), |(y, x, c)| data[(y * w + x) * 4 + c]);
        Ok(numpy::PyArray3::from_owned_array(py, arr))
    }

    fn __repr__(&self) -> String {
        format!(
            "DiagnosticImage({}x{})",
            self.inner.width(),
            self.inner.height()
        )
    }
}

// ---------------------------------------------------------------------------
// Numpy -> ImageView helper
// ---------------------------------------------------------------------------

/// Convert a 2D uint8 numpy array (H x W) to an OwnedImage<u8>.
pub fn numpy_to_owned_u8(array: &PyReadonlyArray2<u8>) -> PyResult<radsym::OwnedImage<u8>> {
    let shape = array.shape();
    let h = shape[0];
    let w = shape[1];
    let data: Vec<u8> = array.as_slice()?.to_vec();
    radsym::OwnedImage::from_vec(data, w, h).map_err(to_pyerr)
}
