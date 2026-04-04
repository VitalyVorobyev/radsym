#[pyclass(name = "FrstConfig", skip_from_py_object)]
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
#[pyclass(name = "RsdConfig", skip_from_py_object)]
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
#[pyclass(name = "NmsConfig", skip_from_py_object)]
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
#[pyclass(name = "ScoringConfig", skip_from_py_object)]
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
#[pyclass(name = "CircleRefineConfig", skip_from_py_object)]
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
///     max_iterations: Maximum refinement iterations. Default: 5.
///     convergence_tol: Stop when center shift is below this (pixels). Default: 0.1.
///     annulus_margin: Fractional annulus margin around the ellipse. Default: 0.3.
///     ray_count: Number of angular sectors used for edge acquisition. Default: 96.
///     radial_search_inner: Inner radial search factor for the initial seed. Default: 0.6.
///     radial_search_outer: Outer radial search factor for the initial seed. Default: 1.45.
///     normal_search_half_width: Half-width of the normal search window. Default: 6.0.
///     min_inlier_coverage: Minimum inlier sector coverage. Default: 0.6.
///     max_center_shift_fraction: Maximum center shift from the seed as a radius fraction. Default: 0.4.
///     max_axis_ratio: Maximum allowed ellipse axis ratio. Default: 1.8.
#[pyclass(name = "EllipseRefineConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEllipseRefineConfig {
    pub inner: radsym::EllipseRefineConfig,
}

#[pymethods]
impl PyEllipseRefineConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        max_iterations=5,
        convergence_tol=0.1,
        annulus_margin=0.3,
        ray_count=96,
        radial_search_inner=0.6,
        radial_search_outer=1.45,
        normal_search_half_width=6.0,
        min_inlier_coverage=0.6,
        max_center_shift_fraction=0.4,
        max_axis_ratio=1.8
    ))]
    fn new(
        max_iterations: usize,
        convergence_tol: f32,
        annulus_margin: f32,
        ray_count: usize,
        radial_search_inner: f32,
        radial_search_outer: f32,
        normal_search_half_width: f32,
        min_inlier_coverage: f32,
        max_center_shift_fraction: f32,
        max_axis_ratio: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: radsym::EllipseRefineConfig {
                max_iterations,
                convergence_tol,
                annulus_margin,
                ray_count,
                radial_search_inner,
                radial_search_outer,
                normal_search_half_width,
                min_inlier_coverage,
                max_center_shift_fraction,
                max_axis_ratio,
                ..Default::default()
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "EllipseRefineConfig(max_iterations={}, convergence_tol={}, ray_count={})",
            self.inner.max_iterations, self.inner.convergence_tol, self.inner.ray_count,
        )
    }
}

/// Configuration for homography-aware proposal reranking.
#[pyclass(name = "HomographyRerankConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PyHomographyRerankConfig {
    pub inner: radsym::HomographyRerankConfig,
}

#[pymethods]
impl PyHomographyRerankConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        min_radius=6.0,
        max_radius=0.0,
        radius_step=2.0,
        ray_count=64,
        radial_search_inner=0.6,
        radial_search_outer=1.45,
        size_prior_sigma=0.22,
        center_prior_sigma_fraction=0.45
    ))]
    fn new(
        min_radius: f32,
        max_radius: f32,
        radius_step: f32,
        ray_count: usize,
        radial_search_inner: f32,
        radial_search_outer: f32,
        size_prior_sigma: f32,
        center_prior_sigma_fraction: f32,
    ) -> Self {
        Self {
            inner: radsym::HomographyRerankConfig {
                min_radius,
                max_radius,
                radius_step,
                ray_count,
                radial_search_inner,
                radial_search_outer,
                size_prior_sigma,
                center_prior_sigma_fraction,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HomographyRerankConfig(min_radius={}, max_radius={}, ray_count={})",
            self.inner.min_radius, self.inner.max_radius, self.inner.ray_count
        )
    }
}

/// Configuration for homography-aware refinement.
#[pyclass(name = "HomographyEllipseRefineConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PyHomographyEllipseRefineConfig {
    pub inner: radsym::HomographyEllipseRefineConfig,
}

#[pymethods]
impl PyHomographyEllipseRefineConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        max_iterations=5,
        convergence_tol=0.1,
        ray_count=96,
        radial_search_inner=0.6,
        radial_search_outer=1.45,
        normal_search_half_width=6.0,
        min_inlier_coverage=0.45,
        max_center_shift_fraction=0.4,
        max_radius_change_fraction=0.6
    ))]
    fn new(
        max_iterations: usize,
        convergence_tol: f32,
        ray_count: usize,
        radial_search_inner: f32,
        radial_search_outer: f32,
        normal_search_half_width: f32,
        min_inlier_coverage: f32,
        max_center_shift_fraction: f32,
        max_radius_change_fraction: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: radsym::HomographyEllipseRefineConfig {
                max_iterations,
                convergence_tol,
                ray_count,
                radial_search_inner,
                radial_search_outer,
                normal_search_half_width,
                min_inlier_coverage,
                max_center_shift_fraction,
                max_radius_change_fraction,
                ..Default::default()
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "HomographyEllipseRefineConfig(max_iterations={}, ray_count={})",
            self.inner.max_iterations, self.inner.ray_count,
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
#[pyclass(name = "RadialCenterConfig", skip_from_py_object)]
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

/// Aggregated configuration for the one-call :func:`detect_circles` pipeline.
///
/// Args:
///     frst: Optional FrstConfig. Uses defaults if None.
///     nms: Optional NmsConfig. Uses defaults if None.
///     scoring: Optional ScoringConfig. Uses defaults if None.
///     refinement: Optional CircleRefineConfig. Uses defaults if None.
///     polarity: Which polarity to detect — ``"bright"``, ``"dark"``, or ``"both"``.
///         Default: ``"both"``.
///     radius_hint: Approximate expected radius in pixels used as the initial
///         circle hypothesis. Default: 10.0.
///     min_score: Minimum support score to keep a detection (in ``[0, 1]``).
///         Default: 0.0.
#[pyclass(name = "DetectCirclesConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDetectCirclesConfig {
    pub inner: radsym::DetectCirclesConfig,
}

#[pymethods]
impl PyDetectCirclesConfig {
    #[new]
    #[pyo3(signature = (
        frst=None,
        nms=None,
        scoring=None,
        refinement=None,
        polarity="both",
        radius_hint=10.0,
        min_score=0.0,
        gradient_operator="sobel"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        frst: Option<&PyFrstConfig>,
        nms: Option<&PyNmsConfig>,
        scoring: Option<&PyScoringConfig>,
        refinement: Option<&PyCircleRefineConfig>,
        polarity: &str,
        radius_hint: f32,
        min_score: f32,
        gradient_operator: &str,
    ) -> PyResult<Self> {
        let defaults = radsym::DetectCirclesConfig::default();
        Ok(Self {
            inner: radsym::DetectCirclesConfig {
                frst: frst.map(|c| c.inner.clone()).unwrap_or(defaults.frst),
                nms: nms.map(|c| c.inner.clone()).unwrap_or(defaults.nms),
                scoring: scoring.map(|c| c.inner.clone()).unwrap_or(defaults.scoring),
                refinement: refinement
                    .map(|c| c.inner.clone())
                    .unwrap_or(defaults.refinement),
                polarity: polarity_from_str(polarity)?,
                radius_hint,
                min_score,
                gradient_operator: gradient_operator_from_str(gradient_operator)?,
            },
        })
    }

    /// Approximate expected radius hint in pixels.
    #[getter]
    fn radius_hint(&self) -> f32 {
        self.inner.radius_hint
    }

    /// Minimum support score to keep a detection.
    #[getter]
    fn min_score(&self) -> f32 {
        self.inner.min_score
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectCirclesConfig(radius_hint={}, min_score={})",
            self.inner.radius_hint, self.inner.min_score,
        )
    }
}

use pyo3::prelude::*;

use crate::convert::{gradient_operator_from_str, polarity_from_str, polarity_to_str};
