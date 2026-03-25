use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use radsym::RadSymError;

pub fn to_pyerr(e: RadSymError) -> PyErr {
    match &e {
        RadSymError::InvalidDimensions { .. }
        | RadSymError::BufferTooSmall { .. }
        | RadSymError::InvalidStride { .. }
        | RadSymError::InvalidConfig { .. }
        | RadSymError::DegenerateHypothesis { .. } => PyValueError::new_err(e.to_string()),
        RadSymError::RefinementFailed { .. } => PyRuntimeError::new_err(e.to_string()),
        RadSymError::ImageIo { .. } => PyOSError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}
