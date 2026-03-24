# radsym — development backlog

## Status legend

- [ ] not started
- [x] done
- [~] in progress

---

## Epic 1: Foundation (core module + workspace)

### Milestone 1.1: Workspace skeleton

- [x] Workspace `Cargo.toml` with metadata, edition 2021, MSRV 1.80
- [x] `crates/radsym/Cargo.toml` with feature flags: rayon, image-io, tracing, affine, serde
- [x] `lib.rs` with module declarations and feature gates
- [x] `CLAUDE.md` with project conventions
- [x] `.gitignore`, `rustfmt.toml`

### Milestone 1.2: Core types

- [x] `scalar.rs` — `Scalar = f32` alias
- [x] `coords.rs` — `PixelCoord`, `PixelIndex`, `ImagePoint` newtype
- [x] `geometry.rs` — `Circle`, `Ellipse`, `Annulus`, `Circle→Ellipse` conversion
- [x] `polarity.rs` — `Polarity` enum (Bright/Dark/Both)
- [x] `error.rs` — `RadSymError` enum via thiserror, `Result<T>` alias
- [x] Unit tests for all core types

### Milestone 1.3: Image abstraction

- [x] `ImageView<'a, T>` struct with stride, from_slice, get, row, roi
- [x] `OwnedImage<T>` with zeros, from_vec, view, get, get_mut
- [x] Bilinear `sample()` for f32 and u8 views
- [x] Tests: construct, stride, out-of-bounds, ROI, bilinear sampling

### Milestone 1.4: Gradient computation

- [x] `GradientField` type (gx, gy as OwnedImage)
- [x] `sobel_gradient` for u8 images (3x3 Sobel, /8 normalization)
- [x] `sobel_gradient_f32` for float images
- [x] `gradient_magnitude` computation
- [x] Tests: step edge, uniform image, dimensions

### Milestone 1.5: Non-maximum suppression

- [x] `NmsConfig` (radius, threshold, max_detections)
- [x] `Peak` type (position + score)
- [x] `non_maximum_suppression` with deterministic ordering (score, then y, then x)
- [x] Tests: single peak, two peaks, budget cap, threshold, suppression

---

## Epic 2: Proposal generation

### Milestone 2.1: FRST reference implementation

- [x] `SeedPoint` type (position + score)
- [x] `Proposal` type (seed + scale_hint + polarity + source)
- [x] `FrstConfig` with defaults (radii, alpha, gradient threshold, polarity, sigma)
- [x] `frst_response_single` — single-radius O_n/M_n accumulation + Gaussian smoothing
- [x] `frst_response` — multi-radius summation
- [x] Tests: synthetic bright disk, dark disk, concentric rings, multi-target, dimensions, threshold
- [ ] Benchmark: FRST response time vs image size (512², 1024², 2048²)

Literature: Loy & Zelinsky, ECCV 2002 / TPAMI 2003

### Milestone 2.2: Proposal extraction

- [x] `ResponseMap` wrapper type
- [x] `extract_proposals` using NMS
- [x] Tests: correct count from multi-peak response, budget enforcement

### Milestone 2.3: FRST production tuning

- [x] Gradient magnitude thresholding (beta parameter)
- [x] Polarity-selective voting (bright-only, dark-only)
- [ ] Optional rayon parallelism for multi-radius computation
- [ ] Benchmarks: gradient threshold effect, rayon vs sequential

### Milestone 2.4: Basic proposal example

- [ ] `examples/basic_proposal.rs` — load image, compute FRST, extract proposals

---

## Epic 3: Local support analysis

### Milestone 3.1: Annulus sampling

- [ ] `AnnulusSamplingConfig` (angular samples, radial samples)
- [ ] `sample_annulus` — circular annulus gradient sampling
- [ ] `sample_elliptical_annulus` — elliptical variant
- [ ] Tests: sampling on synthetic ring, angular coverage verification

### Milestone 3.2: Profiles

- [ ] `RadialProfile` type (values, radii, center)
- [ ] `compute_radial_profile` — azimuthally-averaged radial profile
- [ ] `compute_normal_profile` — profile along ellipse normal
- [ ] Tests: profile of synthetic ring shows peaks at edges

### Milestone 3.3: Hypothesis types

- [ ] `CircleHypothesis`, `EllipseHypothesis` (geometry + confidence)
- [ ] `AnnulusHypothesis`, `ConcentricPairHypothesis`
- [ ] Conversion: `CircleHypothesis` → `EllipseHypothesis`

### Milestone 3.4: Support scoring

- [ ] `SupportEvidence` type (gradient samples, angular coverage, alignment)
- [ ] `SupportScore` type (total, ringness, polarity consistency, coverage, degeneracy)
- [ ] `score_circle_support` — gradient alignment + coverage scoring
- [ ] `score_ellipse_support`
- [ ] `angular_coverage` function
- [ ] Tests: high score for centered circle, low for offset, degeneracy flags

---

## Epic 4: Local refinement

### Milestone 4.1: Parthasarathy radial center (reference)

- [ ] `RadialCenterConfig` (patch_radius)
- [ ] `radial_center_refine` — closed-form weighted LS intersection of gradient lines
- [ ] Roberts-cross gradient on half-pixel grid (per original paper)
- [ ] Tests: subpixel accuracy on synthetic Gaussian blob, synthetic ring
- [ ] Benchmark: refinement time per seed

Literature: Parthasarathy, Nature Methods 2012

### Milestone 4.2: Radial center (production)

- [ ] Production variant with Sobel gradient
- [ ] Gradient magnitude weighting
- [ ] Tests: compare reference vs production accuracy

### Milestone 4.3: RefinementResult

- [ ] `RefinementResult<H>` with status, residual, uncertainty
- [ ] `RefinementStatus` enum (Converged, MaxIterations, Degenerate, OutOfBounds)
- [ ] Tests: degeneracy detection

### Milestone 4.4: Circle refinement

- [ ] `CircleRefineConfig`
- [ ] `refine_circle` — iterative gradient-based circle parameter update
- [ ] Tests: convergence from noisy initial estimate

### Milestone 4.5: Ellipse refinement

- [ ] `EllipseRefineConfig`
- [ ] `refine_ellipse` — iterative ellipse parameter update
- [ ] Tests: convergence on synthetic ellipse

---

## Epic 5: Fast proposal variants (RSD lineage)

Literature: Barnes, Zelinsky, Fletcher, IEEE T-ITS 2008

### Milestone 5.1: RSD-style proposal

- [ ] `RsdConfig` with proposal budget
- [ ] `rsd_response` — simplified magnitude-only voting
- [ ] Budget-aware early termination
- [ ] Benchmark: RSD vs FRST speed and recall comparison

### Milestone 5.2: Dense-scene tuning

- [ ] Multi-scale proposal fusion (coarse-to-fine)
- [ ] Benchmark: dense repeated patterns (50+ targets)

---

## Epic 6: Affine/projective extensions (experimental)

Literature: Ni, Singh, Bahlmann, CVPR 2012

### Milestone 6.1: Affine types

- [ ] `AffineMap` type behind `affine` feature gate
- [ ] Affine parameter sampling strategy

### Milestone 6.2: GFRS-style response

- [ ] `affine_frst_response` — voting with affine-transformed gradient offsets
- [ ] Tests: detect synthetic ellipse that isotropic FRST misses
- [ ] Benchmark: affine-aware vs isotropic cost/benefit

---

## Epic 7: Diagnostics

### Milestone 7.1: Heatmap export

- [ ] `DiagnosticImage` type (RGBA buffer)
- [ ] `response_heatmap` — map f32 response to colormap RGBA
- [ ] Feature-gated PNG export via `image` crate

### Milestone 7.2: Overlays

- [ ] `overlay_proposals` — draw seed markers
- [ ] `overlay_circle`, `overlay_ellipse`
- [ ] `overlay_annulus_samples`

### Milestone 7.3: Data export

- [ ] JSON export of proposals and scores (serde feature)
- [ ] Radial profile data export

---

## Epic 8: Documentation, CI, publishing

### Milestone 8.1: Rustdoc

- [ ] Module-level doc comments for every module
- [ ] Type-level and function-level doc comments
- [ ] Crate-level doc with quick-start example in `lib.rs`
- [ ] Literature traceability doc comments on algorithm implementations

### Milestone 8.2: Examples

- [ ] `examples/basic_proposal.rs`
- [ ] `examples/support_scoring.rs`
- [ ] `examples/radial_center_refine.rs`
- [ ] `examples/diagnostics_export.rs`

### Milestone 8.3: CI

- [ ] GitHub Actions: fmt, clippy, test (all features), test (no default features)
- [ ] MSRV check (1.80)

### Milestone 8.4: crates.io preparation

- [ ] Fill README.md with badges, overview, usage
- [ ] Create CHANGELOG.md
- [ ] License files (MIT + Apache-2.0)

---

## Epic 9: Python bindings

### Milestone 9.1: radsym-py crate

- [ ] `crates/radsym-py/Cargo.toml` with PyO3 + maturin
- [ ] Numpy ndarray image interop
- [ ] Expose proposal generation
- [ ] Expose support scoring
- [ ] Expose refinement
- [ ] Python tests

---

## Algorithm inventory

| Algorithm | Module | Source | Fidelity | Status |
|-----------|--------|--------|----------|--------|
| FRST | `propose::frst` | Loy & Zelinsky 2002/2003 | reference + production | not started |
| RSD | `propose::rsd` | Barnes et al. 2008 | production | not started |
| Parthasarathy radial center | `refine::radial_center` | Parthasarathy 2012 | reference + production | not started |
| GST | — | Reisfeld et al. 1995 | reference | deferred |
| Iterative voting | — | Parvin et al. 2007 | experimental | deferred |
| GFRS | `affine::propose` | Ni et al. 2012 | experimental | not started |

---

## ADR backlog

Architectural decisions to document formally as the project matures:

1. **Coordinate convention** — (x=col, y=row), matching nalgebra Point2. Decided.
2. **Image abstraction** — concrete `ImageView<'a, T>` struct, not trait. Decided.
3. **Scalar precision** — f32 everywhere. Decided.
4. **Feature flags** — rayon, image-io, tracing, affine, serde. Decided.
5. **Determinism policy** — deterministic ordering by default. Decided.
6. **Single crate vs multi-crate** — single crate with modules. Decided.
7. **Diagnostics format** — TBD (likely PNG heatmaps + JSON data)
8. **Stable vs experimental boundary** — affine module is experimental. Decided.
