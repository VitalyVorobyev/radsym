use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_pyerr;

fn parse_homography_input(matrix: &Bound<'_, PyAny>) -> PyResult<radsym::Homography> {
    if let Ok(flat) = matrix.extract::<Vec<f32>>() {
        if flat.len() != 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "homography flat sequence must contain exactly 9 elements",
            ));
        }
        return radsym::Homography::from_flat([
            flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6], flat[7], flat[8],
        ])
        .map_err(to_pyerr);
    }

    if let Ok(rows) = matrix.extract::<Vec<Vec<f32>>>() {
        if rows.len() != 3 || rows.iter().any(|row| row.len() != 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "homography nested sequence must have shape 3x3",
            ));
        }
        return radsym::Homography::new([
            [rows[0][0], rows[0][1], rows[0][2]],
            [rows[1][0], rows[1][1], rows[1][2]],
            [rows[2][0], rows[2][1], rows[2][2]],
        ])
        .map_err(to_pyerr);
    }

    if let Ok(array) = matrix.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = array.shape();
        if shape != [3, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "homography numpy array must have shape (3, 3)",
            ));
        }
        let slice = array.as_slice()?;
        return radsym::Homography::from_flat([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
            slice[8],
        ])
        .map_err(to_pyerr);
    }

    if let Ok(array) = matrix.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape != [3, 3] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "homography numpy array must have shape (3, 3)",
            ));
        }
        let slice = array.as_slice()?;
        return radsym::Homography::from_flat([
            slice[0] as f32,
            slice[1] as f32,
            slice[2] as f32,
            slice[3] as f32,
            slice[4] as f32,
            slice[5] as f32,
            slice[6] as f32,
            slice[7] as f32,
            slice[8] as f32,
        ])
        .map_err(to_pyerr);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "homography must be a flat 9-element sequence, nested 3x3 sequence, or 3x3 numpy array",
    ))
}

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

/// Validated image-to-rectified homography.
#[pyclass(name = "Homography")]
#[derive(Clone)]
pub struct PyHomography {
    pub inner: radsym::Homography,
}

#[pymethods]
impl PyHomography {
    #[new]
    fn new(matrix: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: parse_homography_input(matrix)?,
        })
    }

    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: radsym::Homography::identity(),
        }
    }

    #[getter]
    fn matrix(&self) -> Vec<Vec<f32>> {
        let flat = self.inner.to_flat();
        vec![
            vec![flat[0], flat[1], flat[2]],
            vec![flat[3], flat[4], flat[5]],
            vec![flat[6], flat[7], flat[8]],
        ]
    }

    fn __repr__(&self) -> String {
        let flat = self.inner.to_flat();
        format!(
            "Homography([[{:.4}, {:.4}, {:.4}], [{:.4}, {:.4}, {:.4}], [{:.6}, {:.6}, {:.4}]])",
            flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6], flat[7], flat[8]
        )
    }
}

/// Caller-defined rectified raster domain.
#[pyclass(name = "RectifiedGrid")]
#[derive(Clone)]
pub struct PyRectifiedGrid {
    pub inner: radsym::RectifiedGrid,
}

#[pymethods]
impl PyRectifiedGrid {
    #[new]
    fn new(width: usize, height: usize) -> PyResult<Self> {
        Ok(Self {
            inner: radsym::RectifiedGrid::new(width, height).map_err(to_pyerr)?,
        })
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height
    }

    fn __repr__(&self) -> String {
        format!(
            "RectifiedGrid(width={}, height={})",
            self.inner.width, self.inner.height
        )
    }
}
