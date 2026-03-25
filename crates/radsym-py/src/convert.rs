use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use radsym::Polarity;

pub fn polarity_from_str(s: &str) -> PyResult<Polarity> {
    match s.to_lowercase().as_str() {
        "bright" => Ok(Polarity::Bright),
        "dark" => Ok(Polarity::Dark),
        "both" => Ok(Polarity::Both),
        _ => Err(PyValueError::new_err(format!(
            "unknown polarity '{s}', expected 'bright', 'dark', or 'both'"
        ))),
    }
}

pub fn polarity_to_str(p: Polarity) -> &'static str {
    match p {
        Polarity::Bright => "bright",
        Polarity::Dark => "dark",
        Polarity::Both => "both",
        _ => "unknown",
    }
}

pub fn colormap_from_str(s: &str) -> PyResult<radsym::diagnostics::heatmap::Colormap> {
    use radsym::diagnostics::heatmap::Colormap;
    match s.to_lowercase().as_str() {
        "jet" => Ok(Colormap::Jet),
        "hot" => Ok(Colormap::Hot),
        "magma" => Ok(Colormap::Magma),
        _ => Err(PyValueError::new_err(format!(
            "unknown colormap '{s}', expected 'jet', 'hot', or 'magma'"
        ))),
    }
}

pub fn status_to_str(s: &radsym::RefinementStatus) -> &'static str {
    match s {
        radsym::RefinementStatus::Converged => "converged",
        radsym::RefinementStatus::MaxIterations => "max_iterations",
        radsym::RefinementStatus::Degenerate => "degenerate",
        radsym::RefinementStatus::OutOfBounds => "out_of_bounds",
        _ => "unknown",
    }
}
