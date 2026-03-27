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

/// Homography-aware FRST response on a rectified grid.
#[pyclass(name = "RectifiedResponseMap")]
pub struct PyRectifiedResponseMap {
    pub inner: radsym::RectifiedResponseMap,
}

#[pymethods]
impl PyRectifiedResponseMap {
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let response = self.inner.response();
        let w = response.width();
        let h = response.height();
        let data = response.data();
        let arr = Array2::from_shape_fn((h, w), |(y, x)| data[y * w + x]);
        PyArray2::from_owned_array(py, arr)
    }

    fn scale_hints_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let response = self.inner.scale_hints();
        let w = response.width();
        let h = response.height();
        let data = response.data();
        let arr = Array2::from_shape_fn((h, w), |(y, x)| data[y * w + x]);
        PyArray2::from_owned_array(py, arr)
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.response().width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.response().height()
    }

    fn __repr__(&self) -> String {
        format!(
            "RectifiedResponseMap({}x{})",
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
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::error::to_pyerr;
