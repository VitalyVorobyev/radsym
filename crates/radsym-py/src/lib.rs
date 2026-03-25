//! Python bindings for the radsym radial symmetry detection library.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use radsym::core::gradient::sobel_gradient;
use radsym::propose::extract::extract_proposals;
use radsym::propose::seed::ProposalSource;

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
    Ok(PyResponseMap {
        inner: radsym::ResponseMap::new(response, ProposalSource::Frst),
    })
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
    Ok(PyResponseMap {
        inner: radsym::ResponseMap::new(response, ProposalSource::Rsd),
    })
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
) -> PyCircleRefinementResult {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = radsym::refine_circle(&gradient.inner, &circle.inner, &cfg);
    PyCircleRefinementResult {
        circle: result.hypothesis,
        status: result.status,
        residual: result.residual,
        iterations: result.iterations,
    }
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
) -> PyEllipseRefinementResult {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let result = radsym::refine_ellipse(&gradient.inner, &ellipse.inner, &cfg);
    PyEllipseRefinementResult {
        ellipse: result.hypothesis,
        status: result.status,
        residual: result.residual,
        iterations: result.iterations,
    }
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
) -> PyPointRefinementResult {
    let cfg = config.map(|c| c.inner.clone()).unwrap_or_default();
    let coord = radsym::PixelCoord::new(seed.0, seed.1);
    let result = radsym::radial_center_refine_from_gradient(&gradient.inner, coord, &cfg);
    PyPointRefinementResult {
        point: (result.hypothesis.x, result.hypothesis.y),
        status: result.status,
        residual: result.residual,
        iterations: result.iterations,
    }
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

    // Config types
    m.add_class::<PyFrstConfig>()?;
    m.add_class::<PyRsdConfig>()?;
    m.add_class::<PyNmsConfig>()?;
    m.add_class::<PyScoringConfig>()?;
    m.add_class::<PyCircleRefineConfig>()?;
    m.add_class::<PyEllipseRefineConfig>()?;
    m.add_class::<PyRadialCenterConfig>()?;

    // Result types
    m.add_class::<PySupportScore>()?;
    m.add_class::<PyProposal>()?;
    m.add_class::<PyCircleRefinementResult>()?;
    m.add_class::<PyEllipseRefinementResult>()?;
    m.add_class::<PyPointRefinementResult>()?;

    // Opaque handles
    m.add_class::<PyGradientField>()?;
    m.add_class::<PyResponseMap>()?;
    m.add_class::<PyDiagnosticImage>()?;

    // Functions — exposed without _py suffix via #[pyo3(name = "...")]
    m.add_function(wrap_pyfunction!(sobel_gradient_py, m)?)?;
    m.add_function(wrap_pyfunction!(frst_response_py, m)?)?;
    m.add_function(wrap_pyfunction!(rsd_response_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_proposals_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_circle_support_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_ellipse_support_py, m)?)?;
    m.add_function(wrap_pyfunction!(refine_circle_py, m)?)?;
    m.add_function(wrap_pyfunction!(refine_ellipse_py, m)?)?;
    m.add_function(wrap_pyfunction!(radial_center_refine_py, m)?)?;
    m.add_function(wrap_pyfunction!(response_heatmap_py, m)?)?;
    m.add_function(wrap_pyfunction!(overlay_circle_py, m)?)?;
    m.add_function(wrap_pyfunction!(overlay_ellipse_py, m)?)?;
    m.add_function(wrap_pyfunction!(save_diagnostic_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_grayscale_py, m)?)?;

    Ok(())
}
