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
- [x] `coords.rs` — `PixelCoord`, `PixelIndex` (ImagePoint removed as unused)
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
- [x] Benchmark: FRST response time vs image size (256², 512², 1024²)

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

### Milestone 2.4: Examples

- [x] `examples/detect_rings.rs` — load image, compute FRST, extract proposals, score, refine
- [x] `examples/detect_highlight.rs` — single target with ellipse refinement
- [x] JSON config files for all examples (`examples/configs/`)

---

## Epic 3: Local support analysis

### Milestone 3.1: Annulus sampling

- [x] `AnnulusSamplingConfig` (angular samples, radial samples)
- [x] `sample_annulus` — circular annulus gradient sampling
- [x] `sample_elliptical_annulus` — elliptical variant (accepts `&Ellipse`)
- [x] Tests: sampling on synthetic ring, angular coverage verification

### Milestone 3.2: Profiles

- [x] `RadialProfile` type (values, radii, center)
- [x] `compute_radial_profile` — azimuthally-averaged radial profile
- [x] `compute_normal_profile` — profile along ellipse normal (accepts `&Ellipse`)
- [x] Tests: profile of synthetic ring shows peaks at edges

### Milestone 3.3: Hypothesis types

- [x] `CircleHypothesis`, `EllipseHypothesis` (geometry + confidence)
- [x] `AnnulusHypothesis`, `ConcentricPairHypothesis`
- [x] Conversion: `CircleHypothesis` → `EllipseHypothesis`

### Milestone 3.4: Support scoring

- [x] `SupportEvidence` type (gradient samples, angular coverage, alignment)
- [x] `SupportScore` type (total, ringness, polarity consistency, coverage, degeneracy)
- [x] `score_circle_support` — gradient alignment + coverage scoring
- [x] `score_ellipse_support`
- [x] `angular_coverage` function
- [x] Tests: high score for centered circle, low for offset, degeneracy flags

---

## Epic 4: Local refinement

### Milestone 4.1: Parthasarathy radial center (reference)

- [x] `RadialCenterConfig` (patch_radius, gradient_threshold)
- [x] `radial_center_refine` — closed-form weighted LS intersection of gradient lines
- [x] Roberts-cross gradient on half-pixel grid (per original paper)
- [x] Tests: subpixel accuracy on synthetic Gaussian blob, synthetic ring
- [x] Benchmark: refinement time per seed (radial_center + refine_circle)

Literature: Parthasarathy, Nature Methods 2012

### Milestone 4.2: Radial center (production)

- [x] `radial_center_refine_from_gradient` — production variant with Sobel gradient
- [x] Gradient magnitude squared weighting (Parthasarathy scheme)
- [x] Tests: compare reference vs production accuracy and consistency

### Milestone 4.3: RefinementResult

- [x] `RefinementResult<H>` with status, residual, iterations
- [x] `RefinementStatus` enum (Converged, MaxIterations, Degenerate, OutOfBounds)
- [x] Tests: convergence and degeneracy detection

### Milestone 4.4: Circle refinement

- [x] `CircleRefineConfig`
- [x] `refine_circle` — iterative center + radius update via radial center + annulus sampling
- [x] Tests: convergence from noisy initial estimate, degeneracy on empty image

### Milestone 4.5: Ellipse refinement

- [x] `EllipseRefineConfig`
- [x] `refine_ellipse` — iterative ellipse parameter update via covariance fitting
- [x] Tests: convergence on synthetic circle-as-ellipse, degeneracy on empty image

---

## Epic 5: Fast proposal variants (RSD lineage)

Literature: Barnes, Zelinsky, Fletcher, IEEE T-ITS 2008

### Milestone 5.1: RSD-style proposal

- [x] `RsdConfig` with gradient threshold, polarity, smoothing
- [x] `rsd_response` — simplified magnitude-only voting (single + multi-radius)
- [x] Tests: bright disk detection, multi-target, dimensions, gradient threshold
- [x] Benchmark: RSD response time vs image size (256², 512², 1024²)

### Milestone 5.2: Dense-scene tuning

- [ ] Multi-scale proposal fusion (coarse-to-fine)
- [ ] Benchmark: dense repeated patterns (50+ targets)

---

## Epic 6: Affine/projective extensions (experimental)

Literature: Ni, Singh, Bahlmann, CVPR 2012

### Milestone 6.1: Affine types

- [x] `AffineMap` type behind `affine` feature gate (2×2 matrix with compose, inverse, apply)
- [x] `sample_affine_maps` — rotation × anisotropic scaling parameter sampling
- [x] Tests: identity, rotation, inverse, composition, singularity

### Milestone 6.2: GFRS-style response

- [x] `affine_frst_response_single` — voting with affine-warped gradient offsets
- [x] `affine_frst_responses` — multi-map responses sorted by peak
- [x] Tests: detect synthetic ellipse center, response ordering
- [ ] Benchmark: affine-aware vs isotropic cost/benefit (deferred)

---

## Epic 7: Diagnostics

### Milestone 7.1: Heatmap export

- [x] `DiagnosticImage` type (RGBA buffer with set/get pixel)
- [x] `response_heatmap` — map f32 response to colormap RGBA (Jet, Hot, Magma)
- [x] Feature-gated PNG export via `image` crate (`diagnostics/export.rs`, `core/io.rs`)

### Milestone 7.2: Overlays

- [x] `overlay_proposals` — draw seed markers
- [x] `overlay_circle`, `overlay_ellipse` — shape outlines
- [x] `overlay_marker` — cross marker at position

### Milestone 7.3: Data export

- [x] Serde derives on all public config, result, and geometry types
- [ ] JSON export of proposals and scores (convenience functions)
- [ ] Radial profile data export

---

## Epic 8: Documentation, CI, publishing

### Milestone 8.1: Rustdoc

- [x] Module-level doc comments for every module
- [x] Type-level and function-level doc comments
- [x] Crate-level doc with quick-start example in `lib.rs` (tested)
- [x] Literature traceability doc comments on algorithm implementations

### Milestone 8.2: Examples

- [x] `examples/detect_rings.rs` — FRST + scoring + circle refinement on ring grid
- [x] `examples/detect_highlight.rs` — single target with ellipse refinement
- [x] `examples/radial_center_demo.rs` — subpixel center refinement accuracy
- [x] `examples/diagnostics_demo.rs` — heatmap + circle overlay export

### Milestone 8.3: CI

- [x] GitHub Actions: fmt, clippy, test (all features), test (no default features)
- [x] MSRV check (1.80)
- [x] Documentation build with warnings-as-errors

### Milestone 8.4: crates.io preparation

- [ ] Fill README.md with badges, overview, usage
- [ ] Create CHANGELOG.md
- [ ] License files (MIT + Apache-2.0)

---

## Epic 9: Python bindings

### Milestone 9.1: radsym-py crate

- [x] `crates/radsym-py/Cargo.toml` with PyO3 0.25 + maturin
- [x] Numpy ndarray image interop (uint8 input, float32/uint8 output)
- [x] Expose gradient computation (sobel_gradient)
- [x] Expose proposal generation (frst_response, rsd_response, extract_proposals)
- [x] Expose support scoring (score_circle_support, score_ellipse_support)
- [x] Expose refinement (refine_circle, refine_ellipse, radial_center_refine)
- [x] Expose diagnostics (response_heatmap, overlay_circle/ellipse, save_diagnostic, load_grayscale)
- [x] Full Python class wrappers with docstrings and __repr__ for all types
- [x] 27 Python tests (pipeline, types, diagnostics) — all passing

---

## Algorithm inventory

| Algorithm | Module | Source | Fidelity | Status |
|-----------|--------|--------|----------|--------|
| FRST | `propose::frst` | Loy & Zelinsky 2002/2003 | reference + production | done |
| RSD | `propose::rsd` | Barnes et al. 2008 | production | done |
| Parthasarathy radial center | `refine::radial_center` | Parthasarathy 2012 | reference + production | done |
| GST | — | Reisfeld et al. 1995 | reference | deferred |
| Iterative voting | — | Parvin et al. 2007 | experimental | deferred |
| GFRS | `affine::propose` | Ni et al. 2012 | experimental | done |

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
