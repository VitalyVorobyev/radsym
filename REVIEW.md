# Pre-Release Review -- radsym
*Reviewed: 2026-04-04*
*Scope: full workspace (radsym + radsym-py), focused on recent additions*

## Review Verdict

**Overall: PASS**

| Item | Verdict |
|------|---------|
| F01 | PASS |
| F02 | PASS |
| F03 | PASS |
| F04 | PASS |
| F05 | PASS |
| F06 | PASS |
| F07 | PASS |
| F08 | PASS (skipped, no action needed) |

- Verified: 7
- Needs-rework: 0
- Regression: 0
- New issues: none

Full verification suite results:
- `cargo fmt --all -- --check`: clean
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: clean
- `cargo test --workspace --all-features`: 146 unit + 5 integration + 13 doc-tests = all pass
- `cargo test --workspace --no-default-features`: 138 unit + 5 integration + 13 doc-tests = all pass
- `cargo doc -p radsym --all-features --no-deps`: zero warnings

## Executive Summary

The workspace is in good shape overall. Zero unsafe code, clean module boundaries,
`#[non_exhaustive]` on all public enums, actionable error types, and strong test
coverage for core algorithms. The main issues are: (1) clippy fails on unused
variables in a test, (2) the new Scharr/compute_gradient functions need CHANGELOG
entries and doctests, (3) `Detection<T>` lacks serde support, and (4) one unwrap
in the experimental affine module could panic on NaN.

## Findings

### F01 Clippy failure: unused variables in gradient test
- **Severity**: P0 (blocking)
- **Category**: contracts
- **Location**: `crates/radsym/src/core/gradient.rs:417-418`
- **Status**: verified
- **Resolution**: Changed `let (sx, sy)` and `let (cx, cy)` to `let (sx, _)` and `let (cx, _)` in `compute_gradient_dispatches_correctly`.
- **Problem**: `let (sx, sy)` and `let (cx, cy)` unpack gy components that are never used. `cargo clippy -- -D warnings` fails.
- **Fix**: Change to `let (sx, _sy)` and `let (cx, _cy)`.

### F02 CHANGELOG missing Scharr/GradientOperator entries
- **Severity**: P1 (fix before release)
- **Category**: docs
- **Location**: `CHANGELOG.md`
- **Status**: verified
- **Resolution**: Added `[Unreleased]` section listing all five new public items and `gradient_operator` field.
- **Problem**: `scharr_gradient`, `scharr_gradient_f32`, `GradientOperator`, `compute_gradient`, `compute_gradient_f32` are public API items not mentioned in any CHANGELOG version.
- **Fix**: Add to the current unreleased version or create a new version entry.

### F03 `Detection<T>` missing serde support
- **Severity**: P1 (fix before release)
- **Category**: design
- **Location**: `crates/radsym/src/pipeline.rs:55-62`
- **Status**: verified
- **Resolution**: Added `cfg_attr` serde derive with explicit `serialize`/`deserialize` bounds on `Detection<T>`.
- **Problem**: `Detection<T>` is a primary user-facing result type but has no serde derive. Users cannot serialize detection results.
- **Fix**: Add `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` with appropriate bounds.

### F04 Unwrap on f32 partial_cmp in affine module
- **Severity**: P2 (fix soon)
- **Category**: code-quality
- **Location**: `crates/radsym/src/affine/propose.rs:158`
- **Status**: verified
- **Resolution**: Replaced `.unwrap()` with `.unwrap_or(std::cmp::Ordering::Equal)` in `affine_frst_responses` sort.
- **Problem**: `.partial_cmp(&a.peak_value).unwrap()` will panic if `peak_value` is NaN.
- **Fix**: Use `.unwrap_or(std::cmp::Ordering::Equal)`.

### F05 Missing doctests for new gradient functions
- **Severity**: P2 (fix soon)
- **Category**: tests
- **Location**: `crates/radsym/src/core/gradient.rs`
- **Status**: verified
- **Resolution**: Added 16×16 step-edge doctests to `scharr_gradient` and `compute_gradient`, mirroring `sobel_gradient`'s pattern.
- **Problem**: `scharr_gradient` and `compute_gradient` lack doctests. `sobel_gradient` has one.
- **Fix**: Add short doctests using the step-edge pattern.

### F06 Missing unit tests for f32 Scharr/compute variants
- **Severity**: P2 (fix soon)
- **Category**: tests
- **Location**: `crates/radsym/src/core/gradient.rs`
- **Status**: verified
- **Resolution**: Added `scharr_gradient_f32_matches_u8` (u8 vs f32 agreement at step edge) and `compute_gradient_f32_dispatches` (both operators produce non-zero gx on f32 input).
- **Problem**: `scharr_gradient_f32` and `compute_gradient_f32` have no unit tests.
- **Fix**: Add test that verifies f32 and u8 variants produce the same output on equivalent input.

### F07 CLAUDE.md doesn't mention Scharr
- **Severity**: P3 (tech debt)
- **Category**: docs
- **Location**: `CLAUDE.md:26-27`
- **Status**: verified
- **Resolution**: Updated `core/` description to read "gradient (Sobel/Scharr)".
- **Problem**: The core/ module description doesn't mention Scharr as an alternative gradient operator.
- **Fix**: Update to "gradient (Sobel/Scharr)".

### F08 Python compute_gradient not exposed
- **Severity**: P3 (tech debt)
- **Category**: design
- **Location**: `crates/radsym-py/src/lib.rs`
- **Status**: skipped
- **Problem**: `compute_gradient` dispatch function not in Python. Users have `sobel_gradient` and `scharr_gradient` separately which is sufficient. f32 variants not needed for Python (arrays are u8).
- **Fix**: Not needed — two separate functions are idiomatic Python.

## Out-of-Scope Pointers

- Scharr vs Sobel accuracy on circular targets -> `algo-review`
- Fused voting performance profiling -> `perf-architect`
- Ellipse fit Gauss-Newton damping -> `calibration-review`

## Strong Points

- Zero unsafe code across the entire workspace
- `#[non_exhaustive]` on all 6 public enums
- Clean `pub(crate)` boundaries (fused.rs, blur.rs)
- All production `.clone()` calls semantically necessary
- No TODO/FIXME/dead code markers
- Comprehensive serde on config/result types (except Detection<T>)
