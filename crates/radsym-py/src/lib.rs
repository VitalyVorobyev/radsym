//! Python bindings for the radsym radial symmetry detection library.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use radsym::core::gradient::sobel_gradient;
use radsym::propose::extract::extract_proposals;

mod convert;
mod error;
mod types;

use convert::{colormap_from_str, polarity_from_str};
use error::to_pyerr;
use types::*;

// ---------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------

/// Compute Sobel gradient from a grayscale image.
///
/// Args:
///     image: 2D numpy array (uint8, H x W) representing a grayscale image.
///
/// Returns:
///     GradientField: opaque gradient object used by all downstream functions.
///
/// Raises:
///     ValueError: if the image is empty or has invalid dimensions.
#[pyfunction]
#[pyo3(name = "sobel_gradient")]
fn sobel_gradient_py(image: PyReadonlyArray2<u8>) -> PyResult<PyGradientField> {
    let owned = numpy_to_owned_u8(&image)?;
    let grad = sobel_gradient(&owned.view()).map_err(to_pyerr)?;
    Ok(PyGradientField { inner: grad })
}

/// Extract a one-shot pyramid level from a grayscale image.
///
/// Args:
///     image: 2D numpy array (uint8, H x W).
///     level: Pyramid level, where ``0`` is the base image and each increment
///         applies one additional 2x box-filter downsample.
///
/// Returns:
///     PyramidLevelImage: working image plus remap helpers back to image space.
#[pyfunction]
#[pyo3(name = "pyramid_level_image")]
fn pyramid_level_image_py(image: PyReadonlyArray2<u8>, level: u8) -> PyResult<PyPyramidLevelImage> {
    let owned = numpy_to_owned_u8(&image)?;
    let level_image = radsym::pyramid_level_owned(&owned.view(), level).map_err(to_pyerr)?;
    Ok(PyPyramidLevelImage { inner: level_image })
}

// ---------------------------------------------------------------------------
// Proposal generation
// ---------------------------------------------------------------------------

/// Compute the FRST (Fast Radial Symmetry Transform) response map.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     config: Optional FrstConfig. Uses defaults if None.
///
/// Returns:
///     ResponseMap: response accumulator that can be passed to ``extract_proposals``
///     or converted to numpy with ``.to_numpy()``.
///
/// Raises:
///     ValueError: on invalid config parameters.
#[pyfunction]
#[pyo3(name = "frst_response", signature = (gradient, config=None))]
fn frst_response_py(
    gradient: &PyGradientField,
    config: Option<&PyFrstConfig>,
) -> PyResult<PyResponseMap> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let response = radsym::frst_response(&gradient.inner, &cfg).map_err(to_pyerr)?;
    Ok(PyResponseMap { inner: response })
}

/// Compute a fused multi-radius magnitude-only response map.
///
/// Faster than ``frst_response`` when testing many radii. Uses a single
/// image pass and a single blur. The ``alpha`` config field is ignored.
#[pyfunction]
#[pyo3(name = "multiradius_response", signature = (gradient, config=None))]
fn multiradius_response_py(
    gradient: &PyGradientField,
    config: Option<&PyFrstConfig>,
) -> PyResult<PyResponseMap> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let response = radsym::multiradius_response(&gradient.inner, &cfg).map_err(to_pyerr)?;
    Ok(PyResponseMap { inner: response })
}

/// Compute homography-aware FRST on a rectified grid.
#[pyfunction]
#[pyo3(name = "frst_response_homography", signature = (gradient, homography, grid, config=None))]
fn frst_response_homography_py(
    gradient: &PyGradientField,
    homography: &PyHomography,
    grid: &PyRectifiedGrid,
    config: Option<&PyFrstConfig>,
) -> PyResult<PyRectifiedResponseMap> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let response =
        radsym::frst_response_homography(&gradient.inner, &homography.inner, grid.inner, &cfg)
            .map_err(to_pyerr)?;
    Ok(PyRectifiedResponseMap { inner: response })
}

/// Compute the RSD (Radial Symmetry Detector) response map.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     config: Optional RsdConfig. Uses defaults if None.
///
/// Returns:
///     ResponseMap: response accumulator.
#[pyfunction]
#[pyo3(name = "rsd_response", signature = (gradient, config=None))]
fn rsd_response_py(
    gradient: &PyGradientField,
    config: Option<&PyRsdConfig>,
) -> PyResult<PyResponseMap> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let response = radsym::rsd_response(&gradient.inner, &cfg).map_err(to_pyerr)?;
    Ok(PyResponseMap { inner: response })
}

/// Compute a fused multi-radius RSD response map in a single pixel pass.
///
/// Faster than ``rsd_response`` when testing many radii. Uses a single
/// image pass and a single Gaussian blur.
#[pyfunction]
#[pyo3(name = "rsd_response_fused", signature = (gradient, config=None))]
fn rsd_response_fused_py(
    gradient: &PyGradientField,
    config: Option<&PyRsdConfig>,
) -> PyResult<PyResponseMap> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let response = radsym::rsd_response_fused(&gradient.inner, &cfg).map_err(to_pyerr)?;
    Ok(PyResponseMap { inner: response })
}

/// Extract center proposals from a response map using non-maximum suppression.
///
/// Args:
///     response: ResponseMap from ``frst_response`` or ``rsd_response``.
///     nms_config: Optional NmsConfig controlling suppression radius, threshold, and budget.
///     polarity: One of ``"bright"``, ``"dark"``, ``"both"`` (default: ``"both"``).
///
/// Returns:
///     list[Proposal]: detected center proposals sorted by descending score.
#[pyfunction]
#[pyo3(name = "extract_proposals", signature = (response, nms_config=None, polarity="both"))]
fn extract_proposals_py(
    response: &PyResponseMap,
    nms_config: Option<&PyNmsConfig>,
    polarity: &str,
) -> PyResult<Vec<PyProposal>> {
    let nms = nms_config.map(|c| c.inner.clone()).unwrap_or_default();
    let pol = polarity_from_str(polarity)?;
    let proposals = extract_proposals(&response.inner, &nms, pol);
    Ok(proposals
        .into_iter()
        .map(|p| PyProposal { inner: p })
        .collect())
}

/// Extract rectified proposals from a homography-aware response map.
#[pyfunction]
#[pyo3(name = "extract_rectified_proposals", signature = (response, homography, nms_config=None, polarity="both"))]
fn extract_rectified_proposals_py(
    response: &PyRectifiedResponseMap,
    homography: &PyHomography,
    nms_config: Option<&PyNmsConfig>,
    polarity: &str,
) -> PyResult<Vec<PyHomographyProposal>> {
    let nms = nms_config.map(|c| c.inner.clone()).unwrap_or_default();
    let pol = polarity_from_str(polarity)?;
    let proposals =
        radsym::extract_rectified_proposals(&response.inner, &homography.inner, &nms, pol);
    Ok(proposals
        .into_iter()
        .map(|proposal| PyHomographyProposal { inner: proposal })
        .collect())
}

/// Rerank image-space proposals under a known homography.
#[pyfunction]
#[pyo3(name = "rerank_proposals_homography", signature = (gradient, proposals, homography, config=None))]
fn rerank_proposals_homography_py(
    gradient: &PyGradientField,
    proposals: Vec<PyRef<'_, PyProposal>>,
    homography: &PyHomography,
    config: Option<&PyHomographyRerankConfig>,
) -> PyResult<Vec<PyRerankedProposal>> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let rust_proposals = proposals
        .into_iter()
        .map(|proposal| proposal.inner.clone())
        .collect::<Vec<_>>();
    let reranked = radsym::rerank_proposals_homography(
        &gradient.inner,
        &rust_proposals,
        &homography.inner,
        &cfg,
    );
    Ok(reranked
        .into_iter()
        .map(|proposal| PyRerankedProposal { inner: proposal })
        .collect())
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Score how strongly local gradient evidence supports a circle hypothesis.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     circle: Circle hypothesis to evaluate.
///     config: Optional ScoringConfig. Uses defaults if None.
///
/// Returns:
///     SupportScore: structured score with total, ringness, and angular coverage.
#[pyfunction]
#[pyo3(name = "score_circle_support", signature = (gradient, circle, config=None))]
fn score_circle_support_py(
    gradient: &PyGradientField,
    circle: &PyCircle,
    config: Option<&PyScoringConfig>,
) -> PySupportScore {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let score = radsym::support::score::score_circle_support(&gradient.inner, &circle.inner, &cfg);
    PySupportScore { inner: score }
}

/// Score how strongly local gradient evidence supports an ellipse hypothesis.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     ellipse: Ellipse hypothesis to evaluate.
///     config: Optional ScoringConfig. Uses defaults if None.
///
/// Returns:
///     SupportScore: structured score with total, ringness, and angular coverage.
#[pyfunction]
#[pyo3(name = "score_ellipse_support", signature = (gradient, ellipse, config=None))]
fn score_ellipse_support_py(
    gradient: &PyGradientField,
    ellipse: &PyEllipse,
    config: Option<&PyScoringConfig>,
) -> PySupportScore {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let score =
        radsym::support::score::score_ellipse_support(&gradient.inner, &ellipse.inner, &cfg);
    PySupportScore { inner: score }
}

/// Score support for a rectified-frame circle under a known homography.
#[pyfunction]
#[pyo3(name = "score_rectified_circle_support", signature = (gradient, circle, homography, config=None))]
fn score_rectified_circle_support_py(
    gradient: &PyGradientField,
    circle: &PyCircle,
    homography: &PyHomography,
    config: Option<&PyScoringConfig>,
) -> PySupportScore {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let score = radsym::score_rectified_circle_support(
        &gradient.inner,
        &circle.inner,
        &homography.inner,
        &cfg,
    );
    PySupportScore { inner: score }
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

/// Iteratively refine a circle hypothesis using gradient evidence.
///
/// Each iteration updates the center via the radial center method and
/// re-estimates the radius from the gradient peak distribution.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     circle: Initial circle hypothesis.
///     config: Optional CircleRefineConfig. Uses defaults if None.
///
/// Returns:
///     CircleRefinementResult: refined circle, status, residual, and iterations.
#[pyfunction]
#[pyo3(name = "refine_circle", signature = (gradient, circle, config=None))]
fn refine_circle_py(
    gradient: &PyGradientField,
    circle: &PyCircle,
    config: Option<&PyCircleRefineConfig>,
) -> PyResult<PyCircleRefinementResult> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = radsym::refine_circle(&gradient.inner, &circle.inner, &cfg).map_err(to_pyerr)?;
    Ok(PyCircleRefinementResult {
        circle: result.hypothesis,
        status: result.status,
        residual: result.residual,
        iterations: result.iterations,
    })
}

/// Iteratively refine an ellipse hypothesis using gradient evidence.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     ellipse: Initial ellipse hypothesis.
///     config: Optional EllipseRefineConfig. Uses defaults if None.
///
/// Returns:
///     EllipseRefinementResult: refined ellipse, status, residual, and iterations.
#[pyfunction]
#[pyo3(name = "refine_ellipse", signature = (gradient, ellipse, config=None))]
fn refine_ellipse_py(
    gradient: &PyGradientField,
    ellipse: &PyEllipse,
    config: Option<&PyEllipseRefineConfig>,
) -> PyResult<PyEllipseRefinementResult> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = radsym::refine_ellipse(&gradient.inner, &ellipse.inner, &cfg).map_err(to_pyerr)?;
    Ok(PyEllipseRefinementResult {
        ellipse: result.hypothesis,
        status: result.status,
        residual: result.residual,
        iterations: result.iterations,
    })
}

/// Refine an image-space ellipse by fitting a rectified-frame circle.
#[pyfunction]
#[pyo3(name = "refine_ellipse_homography", signature = (gradient, ellipse, homography, config=None))]
fn refine_ellipse_homography_py(
    gradient: &PyGradientField,
    ellipse: &PyEllipse,
    homography: &PyHomography,
    config: Option<&PyHomographyEllipseRefineConfig>,
) -> PyResult<PyHomographyRefinementResult> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result =
        radsym::refine_ellipse_homography(&gradient.inner, &ellipse.inner, &homography.inner, &cfg)
            .map_err(to_pyerr)?;
    Ok(PyHomographyRefinementResult { inner: result })
}

/// Transport a rectified-frame circle back into an image-space ellipse.
#[pyfunction]
#[pyo3(name = "rectified_circle_to_image_ellipse")]
fn rectified_circle_to_image_ellipse_py(
    homography: &PyHomography,
    circle: &PyCircle,
) -> PyResult<PyEllipse> {
    let ellipse = radsym::rectified_circle_to_image_ellipse(&homography.inner, &circle.inner)
        .map_err(to_pyerr)?;
    Ok(PyEllipse { inner: ellipse })
}

/// Refine a seed center to subpixel accuracy using the Parthasarathy radial center method.
///
/// Uses gradient evidence within a local patch around the seed to compute
/// a weighted least-squares intersection of gradient lines.
///
/// Args:
///     gradient: GradientField from ``sobel_gradient``.
///     seed: Initial center estimate as ``(x, y)`` tuple.
///     config: Optional RadialCenterConfig. Uses defaults if None.
///
/// Returns:
///     PointRefinementResult: refined center (x, y), status, residual, and iterations.
///
/// Literature: Parthasarathy, Nature Methods 2012.
#[pyfunction]
#[pyo3(name = "radial_center_refine", signature = (gradient, seed, config=None))]
fn radial_center_refine_py(
    gradient: &PyGradientField,
    seed: (f32, f32),
    config: Option<&PyRadialCenterConfig>,
) -> PyResult<PyPointRefinementResult> {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let coord = radsym::PixelCoord::new(seed.0, seed.1);
    let result = radsym::radial_center_refine_from_gradient(&gradient.inner, coord, &cfg)
        .map_err(to_pyerr)?;
    Ok(PyPointRefinementResult {
        point: (result.hypothesis.x, result.hypothesis.y),
        status: result.status,
        residual: result.residual,
        iterations: result.iterations,
    })
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Detect circles in a grayscale image using the full propose-score-refine pipeline.
///
/// This is a one-call convenience wrapper that runs:
/// 1. Sobel gradient computation
/// 2. FRST voting and NMS proposal extraction
/// 3. Support scoring and filtering
/// 4. Iterative circle refinement
///
/// Args:
///     image: 2D numpy array (uint8, H x W) representing a grayscale image.
///     config: Optional DetectCirclesConfig. Uses defaults if None.
///
/// Returns:
///     list[Detection]: Detections sorted by descending support score.
///
/// Raises:
///     ValueError: if the image is empty or has invalid dimensions.
#[pyfunction]
#[pyo3(name = "detect_circles", signature = (image, config=None))]
fn detect_circles_py(
    image: PyReadonlyArray2<u8>,
    config: Option<&PyDetectCirclesConfig>,
) -> PyResult<Vec<PyDetection>> {
    let owned = numpy_to_owned_u8(&image)?;
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let detections = radsym::detect_circles(&owned.view(), &cfg).map_err(to_pyerr)?;
    Ok(detections
        .into_iter()
        .map(|d| PyDetection { inner: d })
        .collect())
}

/// Greedily suppress proposals that are closer than `min_distance`.
///
/// Input proposals should be sorted by descending score so that the strongest
/// candidate in each spatial neighborhood survives.
///
/// Args:
///     proposals: list[Proposal] — candidates to filter.
///     min_distance: Minimum allowed distance between retained proposals (pixels).
///     max_detections: Maximum number of proposals to return.
///
/// Returns:
///     list[Proposal]: Filtered proposals preserving relative input order.
#[pyfunction]
#[pyo3(name = "suppress_proposals_by_distance")]
fn suppress_proposals_by_distance_py(
    proposals: Vec<PyRef<'_, PyProposal>>,
    min_distance: f32,
    max_detections: usize,
) -> Vec<PyProposal> {
    let rust_proposals: Vec<radsym::Proposal> =
        proposals.into_iter().map(|p| p.inner.clone()).collect();
    radsym::suppress_proposals_by_distance(&rust_proposals, min_distance, max_detections)
        .into_iter()
        .map(|p| PyProposal { inner: p })
        .collect()
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Render a response map as a colored heatmap image.
///
/// Args:
///     response: ResponseMap to visualize.
///     colormap: Color scheme — ``"jet"``, ``"hot"``, or ``"magma"`` (default: ``"hot"``).
///
/// Returns:
///     DiagnosticImage: RGBA image that can be saved or converted to numpy.
#[pyfunction]
#[pyo3(name = "response_heatmap", signature = (response, colormap="hot"))]
fn response_heatmap_py(response: &PyResponseMap, colormap: &str) -> PyResult<PyDiagnosticImage> {
    let cmap = colormap_from_str(colormap)?;
    let hm = radsym::diagnostics::heatmap::response_heatmap(response.inner.response(), cmap);
    Ok(PyDiagnosticImage { inner: hm })
}

/// Draw a circle outline onto a diagnostic image (in-place).
///
/// Args:
///     image: DiagnosticImage to draw on.
///     circle: Circle to draw.
///     color: RGBA tuple (default: red).
#[pyfunction]
#[pyo3(name = "overlay_circle", signature = (image, circle, color=(255, 0, 0, 255)))]
fn overlay_circle_py(image: &mut PyDiagnosticImage, circle: &PyCircle, color: (u8, u8, u8, u8)) {
    radsym::diagnostics::overlay::overlay_circle(
        &mut image.inner,
        &circle.inner,
        [color.0, color.1, color.2, color.3],
    );
}

/// Draw an ellipse outline onto a diagnostic image (in-place).
///
/// Args:
///     image: DiagnosticImage to draw on.
///     ellipse: Ellipse to draw.
///     color: RGBA tuple (default: green).
#[pyfunction]
#[pyo3(name = "overlay_ellipse", signature = (image, ellipse, color=(0, 255, 0, 255)))]
fn overlay_ellipse_py(image: &mut PyDiagnosticImage, ellipse: &PyEllipse, color: (u8, u8, u8, u8)) {
    radsym::diagnostics::overlay::overlay_ellipse(
        &mut image.inner,
        &ellipse.inner,
        [color.0, color.1, color.2, color.3],
    );
}

/// Save a diagnostic image (RGBA) to a file.
///
/// The output format is inferred from the file extension (``.png``, ``.jpg``, etc.).
///
/// Args:
///     image: DiagnosticImage to save.
///     path: Output file path.
///
/// Raises:
///     OSError: if the file cannot be written.
#[pyfunction]
#[pyo3(name = "save_diagnostic")]
fn save_diagnostic_py(image: &PyDiagnosticImage, path: &str) -> PyResult<()> {
    radsym::save_diagnostic(&image.inner, path).map_err(to_pyerr)
}

/// Load a grayscale image from a file path.
///
/// Supports any format the ``image`` crate can decode (PNG, JPEG, etc.).
///
/// Args:
///     path: Path to the image file.
///
/// Returns:
///     numpy.ndarray: 2D uint8 array (H x W).
///
/// Raises:
///     OSError: if the file cannot be read or decoded.
#[pyfunction]
#[pyo3(name = "load_grayscale")]
fn load_grayscale_py(py: Python<'_>, path: &str) -> PyResult<Py<numpy::PyArray2<u8>>> {
    let owned = radsym::load_grayscale(path).map_err(to_pyerr)?;
    let w = owned.width();
    let h = owned.height();
    let data = owned.data().to_vec();
    let arr = numpy::ndarray::Array2::from_shape_vec((h, w), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(numpy::PyArray2::from_owned_array(py, arr).unbind())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Python bindings for radsym — radial symmetry detection.
///
/// Provides a complete pipeline for detecting radially symmetric structures
/// in grayscale images: gradient computation, center voting (FRST/RSD),
/// proposal extraction, support scoring, and iterative refinement.
///
/// Quick start::
///
///     import numpy as np
///     import radsym
///
///     # Create a synthetic bright disk
///     image = np.zeros((64, 64), dtype=np.uint8)
///     for y in range(64):
///         for x in range(64):
///             if ((x - 32)**2 + (y - 32)**2)**0.5 <= 10:
///                 image[y, x] = 255
///
///     gradient = radsym.sobel_gradient(image)
///     config = radsym.FrstConfig(radii=[9, 10, 11])
///     response = radsym.frst_response(gradient, config)
///     proposals = radsym.extract_proposals(response)
///     print(proposals[0])
#[pymodule]
#[pyo3(name = "radsym")]
fn radsym_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Geometry types
    m.add_class::<PyCircle>()?;
    m.add_class::<PyEllipse>()?;
    m.add_class::<PyHomography>()?;
    m.add_class::<PyRectifiedGrid>()?;

    // Config types
    m.add_class::<PyFrstConfig>()?;
    m.add_class::<PyRsdConfig>()?;
    m.add_class::<PyNmsConfig>()?;
    m.add_class::<PyScoringConfig>()?;
    m.add_class::<PyCircleRefineConfig>()?;
    m.add_class::<PyEllipseRefineConfig>()?;
    m.add_class::<PyHomographyRerankConfig>()?;
    m.add_class::<PyHomographyEllipseRefineConfig>()?;
    m.add_class::<PyRadialCenterConfig>()?;
    m.add_class::<PyDetectCirclesConfig>()?;

    // Result types
    m.add_class::<PySupportScore>()?;
    m.add_class::<PyProposal>()?;
    m.add_class::<PyHomographyProposal>()?;
    m.add_class::<PyRerankedProposal>()?;
    m.add_class::<PyCircleRefinementResult>()?;
    m.add_class::<PyEllipseRefinementResult>()?;
    m.add_class::<PyHomographyRefinementResult>()?;
    m.add_class::<PyPointRefinementResult>()?;
    m.add_class::<PyDetection>()?;

    // Opaque handles
    m.add_class::<PyGradientField>()?;
    m.add_class::<PyResponseMap>()?;
    m.add_class::<PyRectifiedResponseMap>()?;
    m.add_class::<PyPyramidLevelImage>()?;
    m.add_class::<PyDiagnosticImage>()?;

    // Functions — exposed without _py suffix via #[pyo3(name = "...")]
    m.add_function(wrap_pyfunction!(sobel_gradient_py, m)?)?;
    m.add_function(wrap_pyfunction!(pyramid_level_image_py, m)?)?;
    m.add_function(wrap_pyfunction!(frst_response_py, m)?)?;
    m.add_function(wrap_pyfunction!(multiradius_response_py, m)?)?;
    m.add_function(wrap_pyfunction!(frst_response_homography_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsd_response_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsd_response_fused_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_proposals_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_rectified_proposals_py, m)?)?;
    m.add_function(wrap_pyfunction!(rerank_proposals_homography_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_circles_py, m)?)?;
    m.add_function(wrap_pyfunction!(suppress_proposals_by_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_circle_support_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_ellipse_support_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_rectified_circle_support_py, m)?)?;
    m.add_function(wrap_pyfunction!(refine_circle_py, m)?)?;
    m.add_function(wrap_pyfunction!(refine_ellipse_py, m)?)?;
    m.add_function(wrap_pyfunction!(refine_ellipse_homography_py, m)?)?;
    m.add_function(wrap_pyfunction!(rectified_circle_to_image_ellipse_py, m)?)?;
    m.add_function(wrap_pyfunction!(radial_center_refine_py, m)?)?;
    m.add_function(wrap_pyfunction!(response_heatmap_py, m)?)?;
    m.add_function(wrap_pyfunction!(overlay_circle_py, m)?)?;
    m.add_function(wrap_pyfunction!(overlay_ellipse_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_diagnostic_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_grayscale_py, m)?)?;

    Ok(())
}
