# radsym

Rust library for radial symmetry detection: center proposal generation, local
circular/elliptical support analysis, support scoring, and local image-space
refinement. CPU-first, deterministic, composable tools — not a monolithic pipeline.

## Build Commands

```sh
cargo fmt --all                                                    # format
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lint
cargo test --workspace --all-features                              # test (all)
cargo test --workspace --no-default-features                       # test (minimal)
cargo bench                                                        # benchmarks
```

## Architecture

Single library crate at `crates/radsym/` with internal modules:

- `core/`         — types, image view, errors, NMS, geometry, gradient
- `propose/`      — FRST, RSD, center voting, proposal extraction
- `support/`      — annulus sampling, profiles, scoring, hypothesis types
- `refine/`       — Parthasarathy radial center, circle/ellipse refinement
- `affine/`       — (feature-gated) affine-aware extensions
- `diagnostics/`  — heatmaps, overlays, export

## Coordinate Convention

**(x=col, y=row)** everywhere. Matches `nalgebra::Point2<f32>`. x increases
rightward, y increases downward. `PixelCoord = nalgebra::Point2<f32>`.

## Scalar Precision

`f32` everywhere for pixel-level computation. No `f64` in public API.

## Feature Flags

- `default = []` (no default features)
- `rayon` — parallel execution
- `image-io` — file I/O via `image` crate
- `tracing` — structured logging
- `affine` — experimental affine-aware extensions
- `serde` — serialization for configs/results

## Dependency Rules

- `core/` must not depend on algorithm modules
- `propose/`, `support/`, `refine/` depend only on `core/`
- `affine/` depends on `core/` and `propose/`
- `diagnostics/` may depend on all other modules
- No opencv or heavy vision deps. `nalgebra` only for linear algebra.

## Key Conventions

- `thiserror` for all error types
- Concrete `ImageView<'a, T>` struct for image abstraction (not a trait)
- No allocations in hot loops where avoidable
- Deterministic output ordering
- Every public item must have rustdoc
- Literature traceability: algorithm implementations must cite source paper

## Testing

- Synthetic test generators for circles, rings, ellipses
- Property tests: deterministic ordering, translation sanity
- Benchmark with criterion before optimizing hot paths

## Algorithm Families

| Family | Module | Source |
|--------|--------|--------|
| FRST | `propose::frst` | Loy & Zelinsky, ECCV 2002 / TPAMI 2003 |
| RSD | `propose::rsd` | Barnes, Zelinsky, Fletcher, IEEE T-ITS 2008 |
| Radial center | `refine::radial_center` | Parthasarathy, Nature Methods 2012 |
| GFRS | `affine::propose` | Ni, Singh, Bahlmann, CVPR 2012 |
