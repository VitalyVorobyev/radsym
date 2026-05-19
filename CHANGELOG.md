# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-18

A coordinated public-API revision (see the migration notes in each entry).
Breaking for the `radsym` crate and both binding packages.

### Added

- **`DetectCirclesConfig` builder**: new `DetectCirclesConfig::for_radii(radii)`
  constructor plus chainable `polarity`, `radius_hint`, `min_score`, and
  `gradient_operator` setters, so the common detection case can be expressed
  without assembling the nested stage configs by hand. Purely additive —
  existing struct-literal construction keeps working.
- **`CircleDetection` type alias**: `pub type CircleDetection = Detection<Circle>;`
  for the circle-detection result produced by `detect_circles`, re-exported at
  the crate root and from the prelude. Purely additive — it is the same type as
  `Detection<Circle>`, which continues to work identically.
- **Diagnostics channel**: new `detect_circles_with_diagnostics` returns the
  `Vec<CircleDetection>` result plus a `CircleDetectionDiagnostics` carrying the
  response map, the raw proposals, the rejected candidates (`RejectedProposal` /
  `RejectionReason`), and a per-detection `SupportScoreBreakdown`. The new
  diagnostic types live under `radsym::diagnostics`. Purely additive.

### Changed

- **`multiradius_response` renamed to `frst_response_fused` (breaking)**: the
  fused single-pass FRST variant now uses a `_fused` suffix, matching
  `rsd_response_fused`. Re-exported at the crate root and in the prelude. The
  Python `radsym` package and the WASM `RadSymProcessor` class now also expose
  this as `frst_response_fused` (see below).
- **Python bindings — `multiradius_response` renamed to `frst_response_fused`
  (breaking)**: the Python `radsym.multiradius_response` function is renamed to
  `radsym.frst_response_fused`, matching the Rust crate. Migration: replace
  `radsym.multiradius_response(...)` with `radsym.frst_response_fused(...)`.
- **WASM bindings — `RadSymProcessor.multiradius_response` renamed to
  `frst_response_fused` (breaking)**: the WASM `RadSymProcessor` method for the
  fused single-pass FRST response is renamed to `frst_response_fused`, matching
  the Rust crate and the Python package. The method signature and the
  `Float32Array` output format are unchanged. Migration: replace
  `processor.multiradius_response(...)` with `processor.frst_response_fused(...)`.
- **Python bindings — `Detection.score` is now a `float` (breaking)**:
  `detect_circles` returns `Detection` objects whose `.score` attribute is the
  headline support total as a plain `float` rather than a structured score
  object. Migration: read `detection.score` directly; the per-component
  breakdown (`ringness`, `angular_coverage`, `is_degenerate`) is no longer on
  the detection. The standalone `SupportScore` returned by
  `score_circle_support` / `score_ellipse_support` is unchanged and still
  exposes `total`, `ringness`, `angular_coverage`, and `is_degenerate`.
- **Config split: stable core + `advanced` sub-config (breaking)**: the five
  tuning-heavy config structs now keep only their stable, user-intent fields
  and move their algorithm-acquisition knobs into a dedicated `advanced`
  sub-config:
  - `CircleRefineConfig` keeps `max_iterations`, `convergence_tol`,
    `max_center_drift`; the rest moves to `CircleRefineAdvanced`.
  - `EllipseRefineConfig` keeps `max_iterations`, `convergence_tol`,
    `max_center_shift_fraction`, `max_axis_ratio`; the rest moves to
    `EllipseRefineAdvanced`.
  - `HomographyRerankConfig` keeps `min_radius`, `max_radius`, `radius_step`;
    the rest moves to `HomographyRerankAdvanced`.
  - `HomographyEllipseRefineConfig` keeps `max_iterations`, `convergence_tol`,
    `max_center_shift_fraction`, `max_radius_change_fraction`; the rest moves to
    `HomographyEllipseRefineAdvanced`.
  - `DetectCirclesConfig` keeps `polarity`, `radius_hint`, `min_score`,
    `gradient_operator`, and a new top-level `radii: Vec<u32>` field (the source
    of truth for FRST voting radii); the `frst`, `nms`, `scoring`, and
    `refinement` stage configs move to `DetectCirclesAdvanced`.

  Each config gains a `pub advanced: <Name>Advanced` field; `<Config>::default()`
  is unchanged in effect, and pipeline/refinement results are identical.
  Migration: struct literals that set a moved field must nest it under
  `advanced: <Name>Advanced { .. }`. The new advanced structs are re-exported at
  the crate root and in the prelude.
- **Growth-prone config and result structs are now `#[non_exhaustive]`
  (breaking)**: the public configuration, advanced sub-config, result, and
  detection-diagnostics structs are now `#[non_exhaustive]`, so fields can be
  added in future releases without breaking downstream code. External crates can
  no longer construct these types with a struct literal (including the
  `..Default::default()` functional-update form) or exhaustively pattern-match
  them without `..`. Affected types:
  - Configs: `DetectCirclesConfig`, `DetectCirclesAdvanced`, `FrstConfig`,
    `RsdConfig`, `NmsConfig`, `ScoringConfig`, `AnnulusSamplingConfig`,
    `CircleRefineConfig`, `CircleRefineAdvanced`, `EllipseRefineConfig`,
    `EllipseRefineAdvanced`, `HomographyRerankConfig`, `HomographyRerankAdvanced`,
    `HomographyEllipseRefineConfig`, `HomographyEllipseRefineAdvanced`,
    `RadialCenterConfig`.
  - Results: `Detection<T>` (and its alias `CircleDetection`),
    `RefinementResult<H>`, `RerankedProposal`, `HomographyRefinementResult`.
  - Detection diagnostics: `CircleDetectionDiagnostics`, `RejectedProposal`,
    `SupportScoreBreakdown`.

  Migration: construct configs via `Config::default()` followed by field
  assignment instead of a struct literal, e.g.
  `let mut cfg = FrstConfig::default(); cfg.radii = vec![9, 10, 11];`. The
  builders and chainable setters on `DetectCirclesConfig` (`for_radii`,
  `polarity`, …) are unaffected and remain the recommended entry point. Result
  and diagnostics structs are produced by the library, so most consumers need no
  change beyond adding `..` to any exhaustive pattern match.
- **`SupportScore` split into a compact result score and a diagnostic
  breakdown (breaking)**: `SupportScore` is now `{ total }` only — the compact,
  headline result-tier score. The `ringness`, `angular_coverage`, and
  `is_degenerate` evidence components moved to a dedicated
  `SupportScoreBreakdown` type (also `#[non_exhaustive]`). The scoring functions
  `score_circle_support`, `score_ellipse_support`, and
  `score_rectified_circle_support` now return `SupportScoreBreakdown` instead of
  `SupportScore`; `SupportScoreBreakdown::score()` and
  `From<SupportScoreBreakdown> for SupportScore` reduce it to the compact form.
  `Detection::score` stays typed `SupportScore` and now carries only `total`;
  the per-detection breakdown is available through the diagnostics channel
  (`CircleDetectionDiagnostics::score_breakdowns`), and `RejectedProposal::score`
  is now a `SupportScoreBreakdown`. `SupportScore` remains re-exported at the
  crate root and in the prelude; `SupportScoreBreakdown` is reachable via
  `radsym::support::score` and `radsym::diagnostics`. Migration: direct callers
  of the `score_*_support` functions keep `ringness`/`angular_coverage`/
  `is_degenerate` (the functions return the breakdown); code reading those
  components off a `Detection`'s `score` must instead read them from the
  diagnostics' `score_breakdowns` vec, which is index-aligned with the
  detections.

### Removed

- **Root-facade trim (breaking)**: these items are no longer re-exported at the
  crate root — use the module path instead:
  - `radsym::fit_circle`, `radsym::fit_circle_weighted` →
    `radsym::core::circle_fit::*`
  - `radsym::PyramidWorkspace`, `PyramidLevelView`, `OwnedPyramidLevel`,
    `pyramid_level_owned` → `radsym::core::pyramid::*`
  - `radsym::SupportEvidence` → `radsym::support::evidence::SupportEvidence`
  - `radsym::AnnulusHypothesis`, `CircleHypothesis`, `ConcentricPairHypothesis`,
    `EllipseHypothesis` → `radsym::support::hypothesis::*`
  - `radsym::frst_response_single` →
    `radsym::propose::frst::frst_response_single`
  - `radsym::Colormap`, `radsym::DiagnosticImage`, `radsym::response_heatmap`,
    `radsym::overlay_circle`, `radsym::overlay_ellipse` → `radsym::diagnostics::*`
- **Support stage internals narrowed (breaking)**: `support::score::score_at` is
  removed (use `score_circle_support`); `support::coverage` is now a crate-private
  module; `support::annulus::{sample_annulus, sample_elliptical_annulus}` are now
  crate-private. `support::annulus::AnnulusSamplingConfig` remains public.

## [0.1.4] - 2026-04-09

### Added

- **RSD algorithm in WASM**: `rsd_response` and `rsd_response_fused` methods
  exposed via `RadSymProcessor`, providing ~2x faster magnitude-only proposal
  generation alongside FRST.
- **Fused FRST in WASM**: `multiradius_response` method for single-pass
  multi-radius FRST voting.
- **Proposal extraction in WASM**: `extract_proposals` method returns NMS seed
  proposals as stride-3 `[x, y, score]` arrays for any algorithm.
- **Detailed detection output**: `detect_circles_detailed` returns stride-8
  arrays with score breakdown (ringness, angular coverage) and refinement
  status per detection.
- **Algorithm selector in demo**: choose between FRST, FRST (fused), RSD, and
  RSD (fused) for heatmap and proposal visualization.
- **Seed Proposals panel** in demo showing NMS-extracted proposal locations.
- **Interactive demo in mdBook**: new "Interactive Demo" chapter with embedded
  WASM demo and `book/build.sh` for asset management.
- **demo/README.md** documenting scope, config reference, and overlay features.

### Changed

- **Circle refinement stability**: added `max_center_drift` parameter to
  `CircleRefineConfig` (default: 0.5x radius). Refinement now stops with
  `OutOfBounds` if the center drifts beyond `max_center_drift * radius` from
  its initial position, preventing cascading divergence over multiple
  iterations.
- **`response_heatmap` API**: now accepts an `algorithm` parameter (`"frst"`,
  `"frst_fused"`, `"rsd"`, `"rsd_fused"`) to select the proposal method.
- Demo layout: sidebar and main area scroll independently; Run button moved to
  top of sidebar; gradient magnitude panel restored alongside proposals.
- All 19 config parameters (including `max_center_drift`) exposed in demo UI.

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

[0.2.0]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/VitalyVorobyev/radsym/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/VitalyVorobyev/radsym/releases/tag/v0.1.0
