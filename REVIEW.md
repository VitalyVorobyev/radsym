# Pre-Release Review -- radsym 0.1.0
*Reviewed: 2026-04-02*
*Scope: full workspace (radsym + radsym-py)*

## Review Verdict

**Overall: PASS** -- all findings addressed, no regressions.

| Finding | Verdict |
|---------|---------|
| F01 Root README uses stale API | PASS |
| F02 Python bindings missing `detect_circles` | PASS |
| F03 Python bindings missing `suppress_proposals_by_distance` | PASS |
| F04 No CHANGELOG.md | PASS |
| F05 Limited doctest coverage | PASS |
| F06 Long refine functions | skipped (by design) |
| F07 Internal f64 usage lacks comments | PASS |

- **Verified: 6** / Needs-rework: 0 / Regression: 0
- Full verification suite green: `cargo fmt --check`, `cargo clippy` (zero warnings), `cargo test --all-features` (134 tests + 9 doctests), `cargo test --no-default-features` (126 tests + 9 doctests), `cargo doc` (zero warnings).
- No new issues introduced by the fixes.

## Executive Summary

radsym is a well-engineered, publication-quality Rust library. The codebase demonstrates
excellent practices: zero unsafe code, zero clippy warnings, 138 passing tests, actionable
error types with context, proper `#[non_exhaustive]` on extensible enums, and clean
module boundaries. The API is composable and discoverable with a new prelude and
`detect_circles` convenience entry point.

The main concerns are: (1) the root workspace README uses the old API and won't compile,
(2) the new `detect_circles` pipeline function is not exposed in Python bindings,
(3) several Rust public functions are missing from the Python binding layer, and
(4) no CHANGELOG exists for the release. No correctness bugs, security vulnerabilities,
or soundness issues were found.

## Findings

### F01 Root README uses stale API
- **Severity**: P1 (fix before release)
- **Category**: docs
- **Location**: `README.md:41-72`
- **Status**: verified
- **Resolution**: Replaced Quick Start in README.md with the updated example from crates/radsym/README.md using root-level imports and correct API.
- **Problem**: The workspace root README (`README.md`, distinct from `crates/radsym/README.md`) uses the old API: deep module imports (`radsym::core::gradient::sobel_gradient`), struct-literal `ResponseMap { data: ..., source: ... }` (private fields), and `refined.converged()` (nonexistent method). This code won't compile.
- **Fix**: Replace the Quick Start section with the same updated example from `crates/radsym/README.md`, which uses root-level imports and the new `frst_response` -> `ResponseMap` flow.

### F02 Python bindings missing new `detect_circles` function
- **Severity**: P1 (fix before release)
- **Category**: design
- **Location**: `crates/radsym-py/src/lib.rs`
- **Status**: verified
- **Resolution**: Added `detect_circles_py` wrapper, `PyDetectCirclesConfig` in config.rs, and `PyDetection` in results.rs; registered all in module init.
- **Problem**: The newly added `detect_circles` pipeline function, `DetectCirclesConfig`, and `Detection<Circle>` are not exposed in the Python bindings. Python users can't use the one-call convenience API.
- **Fix**: Add a `detect_circles_py` wrapper that accepts a numpy array + optional `PyDetectCirclesConfig`, calls `detect_circles`, and returns a list of detection results. Add `PyDetectCirclesConfig` to the config types module.

### F03 Python bindings missing several Rust public functions
- **Severity**: P1 (fix before release)
- **Category**: design
- **Location**: `crates/radsym-py/src/lib.rs`
- **Status**: verified
- **Resolution**: Added `suppress_proposals_by_distance_py` wrapper in lib.rs; registered in module init. `detect_circles` covered by F02.
- **Problem**: These public Rust functions have no Python counterpart: `score_at`, `suppress_proposals_by_distance`, `fit_circle`, `fit_circle_weighted`, `remap_proposal_to_image`, `remap_proposals_to_image`, `pyramid_level_owned`. While not all are critical, `suppress_proposals_by_distance` and `score_at` are commonly needed in workflows.
- **Fix**: At minimum, add `suppress_proposals_by_distance` and `detect_circles` to the Python API. The others can be deferred.

### F04 No CHANGELOG.md
- **Severity**: P2 (fix soon)
- **Category**: docs
- **Location**: (missing file)
- **Status**: verified
- **Resolution**: Created CHANGELOG.md at workspace root in Keep a Changelog format documenting the 0.1.0 initial release.
- **Problem**: No CHANGELOG exists. For a 0.1.0 release, a brief initial changelog is expected by the Rust ecosystem (crates.io, cargo-release, etc.).
- **Fix**: Create `CHANGELOG.md` documenting the 0.1.0 initial release with a summary of capabilities.

### F05 Limited doctest coverage
- **Severity**: P2 (fix soon)
- **Category**: tests
- **Location**: workspace-wide
- **Status**: verified
- **Resolution**: Added doctests to `sobel_gradient`, `frst_response`, `extract_proposals`, `score_circle_support`, and `refine_circle`; `detect_circles` already had one. All 9 doctests pass.
- **Problem**: Only 4 doctests exist (crate-level + prelude). None of the ~50 public functions have per-function doctests. Doctests serve as both documentation and compile-checked examples.
- **Fix**: Add doctests to the most important public functions: `sobel_gradient`, `frst_response`, `extract_proposals`, `score_circle_support`, `refine_circle`, `detect_circles`. These can use the synthetic disk pattern from existing tests.

### F06 `refine_ellipse` and `refine_ellipse_homography` are long functions
- **Severity**: P3 (tech debt)
- **Category**: code-quality
- **Location**: `crates/radsym/src/refine/ellipse.rs:385-540`, `crates/radsym/src/refine/homography.rs:386-557`
- **Status**: skipped
- **Problem**: Both functions exceed 150 lines. The complexity is inherent to the iterative refinement algorithms (edge detection + fitting + convergence checking + fallbacks).
- **Fix**: Could extract phases into named helper functions for readability, but the current structure is clear and well-commented. Not blocking.

### F07 Internal f64 usage lacks justification comments
- **Severity**: P3 (tech debt)
- **Category**: code-quality
- **Location**: `crates/radsym/src/refine/radial_center.rs:94-158`, `crates/radsym/src/refine/ellipse_fit.rs:12-13`, `crates/radsym/src/refine/circle_fit.rs:13-14`
- **Status**: verified
- **Resolution**: Added `// f64 accumulation for numerical stability in weighted least-squares` comments at each f64 usage site in all three files.
- **Problem**: CLAUDE.md says "f32 everywhere". Internal code uses f64 for accumulation precision (justified), but lacks comments explaining why. A future contributor might "fix" this to f32 and break numerical stability.
- **Fix**: Add a brief comment at each f64 usage site: `// f64 accumulation for numerical stability in weighted least-squares`.

## Out-of-Scope Pointers

- **Algorithmic correctness**: The Parthasarathy radial center, Kasa circle fit, and Fitzgibbon ellipse fit implementations should be verified against reference datasets -> `algo-review` / `calibration-review`
- **Performance**: FRST voting loop and annulus sampling are hot paths -> `perf-architect` / `criterion-bench`
- **Ellipse fit numerical conditioning**: The Gauss-Newton solver uses a fixed damping factor (1e-4) -> `calibration-review`

## Strong Points

- **Zero unsafe code** in the entire workspace
- **Zero clippy warnings** across all targets and features
- **Excellent error design**: all `RadSymError` variants carry actionable context (dimensions, sizes, reasons)
- **Clean visibility boundaries**: internal helpers properly `pub(crate)` or private, public API is minimal and discoverable
- **Composable architecture**: each pipeline stage works independently, no forced coupling
- **`#[non_exhaustive]`** on all extensible public enums (Polarity, ProposalSource, RefinementStatus, Colormap, RadSymError)
- **Config validation**: all 8 config structs validate at entry points with descriptive errors
- **138 tests** with edge cases (empty images, NaN/Inf, degenerate hypotheses, zero gradients)
- **Comprehensive mdBook** with full math derivations for core algorithms
- **Literature traceability**: every algorithm cites its source paper
