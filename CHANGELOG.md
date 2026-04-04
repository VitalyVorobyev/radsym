# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-04-04

### Changed

- **Gradient computation 5x faster**: replaced bounds-checked `.get()` pixel
  access with direct slice indexing in all four gradient functions
  (`sobel_gradient`, `sobel_gradient_f32`, `scharr_gradient`,
  `scharr_gradient_f32`). Loop bounds guarantee in-bounds access.
- **FRST normalization ~10% faster**: special-case `powf(alpha)` for alpha=1
  and alpha=2 (the paper's default) to avoid the general `exp(a*ln(x))` path.
- **FRST voting loop tighter**: hoisted polarity checks and `n as Scalar`
  conversion out of the inner loop; use `mag.recip()` instead of division.
- Enabled `lto = "thin"` and `codegen-units = 1` in the release profile for
  better cross-crate inlining (~1.6x improvement on multiradius at 1024px).
- Pre-allocate annulus sampling `Vec` to avoid incremental growth.

## [0.1.2] - 2026-04-04

### Added

- `scharr_gradient` — Scharr 3×3 gradient operator for `u8` images, providing
  better rotational isotropy than Sobel for circular structure detection.
- `scharr_gradient_f32` — Scharr operator variant accepting `f32` images.
- `GradientOperator` — enum (`Sobel` | `Scharr`) for runtime operator selection;
  derives `serde` under the `serde` feature flag.
- `compute_gradient` — dispatcher that routes a `u8` image to `sobel_gradient`
  or `scharr_gradient` based on `GradientOperator`.
- `compute_gradient_f32` — same dispatcher for `f32` images.
- `gradient_operator` field on `DetectCirclesConfig` (default: `GradientOperator::Sobel`)
  so users can switch the pipeline gradient operator without touching internal code.
- `multiradius_response` — fused multi-radius magnitude-only FRST voting that
  processes all radii in a single pixel pass with one Gaussian blur, instead
  of FRST's per-radius passes. The `alpha` config field is ignored.
- `rsd_response_fused` — same fused single-pass strategy for RSD. Shares the
  voting kernel with `multiradius_response` via an internal helper.
- Criterion benchmarks for both fused variants vs their per-radius originals.
- Python bindings: `multiradius_response` and `rsd_response_fused` exposed
  via `radsym-py`.

## [0.1.1] - 2026-04-03

### Added

- `OwnedImage::into_data()` — consume an image and return the backing `Vec<T>`.
- `ResponseMap::into_response()` — consume a response map and return the
  underlying `OwnedImage<Scalar>`.
- `GradientField::max_magnitude()` — maximum gradient magnitude across the
  entire field (useful for relative-to-absolute threshold conversion).

### Fixed

- `detect_circles` now validates `CircleRefineConfig` eagerly, so invalid
  refinement settings (e.g. `max_iterations = 0`) return `Err(InvalidConfig)`
  instead of being silently swallowed and returning `Ok([])`.

### Changed

- Adapted release CI workflows (`release.yml`, `release-pypi.yml`) for the
  `radsym` package name and workspace version layout.

## [0.1.0] - 2026-04-02

### Added

- **Center proposal generation** via FRST (Loy & Zelinsky, TPAMI 2003) and RSD
  (Barnes, Zelinsky, Fletcher, IEEE T-ITS 2008), with optional rayon parallelism.
- **Homography-aware FRST** (`frst_response_homography`): votes in a rectified
  coordinate frame to handle oblique viewpoints.
- **Non-maximum suppression** (`extract_proposals`) with configurable radius,
  threshold, and detection budget.
- **Spatial deduplication** (`suppress_proposals_by_distance`) for downstream
  post-processing.
- **Annular support scoring** (`score_circle_support`, `score_ellipse_support`)
  measuring gradient alignment and angular coverage.
- **Iterative circle refinement** (`refine_circle`) combining the Parthasarathy
  radial center method (Nature Methods 2012) with annulus-based radius estimation.
- **Iterative ellipse refinement** (`refine_ellipse`) using Fitzgibbon et al.
  direct least-squares fitting (TPAMI 1999) with robust trimming.
- **Homography-aware ellipse refinement** (`refine_ellipse_homography`): fits a
  circle in rectified space and back-projects to an image-space ellipse.
- **One-call pipeline** (`detect_circles`) for the common propose-score-refine
  workflow.
- **Pyramid support** (`pyramid_level_owned`) for coarse-to-fine processing.
- **Kåsa circle fit** (`fit_circle`, `fit_circle_weighted`) for algebraic circle
  fitting (IEEE T-IM 1976).
- **Diagnostics**: response heatmaps (`response_heatmap`) and shape overlays
  (`overlay_circle`, `overlay_ellipse`).
- **Python bindings** (`radsym-py`) via PyO3/maturin exposing the full pipeline
  with numpy I/O.
- Feature flags: `rayon`, `image-io`, `tracing`, `affine`, `serde`.
- Zero unsafe code; zero clippy warnings; 138 unit and integration tests.
- mdBook documentation with full mathematical derivations.

[0.1.3]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/VitalyVorobyev/radsym/releases/tag/v0.1.0
