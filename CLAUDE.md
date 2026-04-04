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

Workspace with two crates:

- `crates/radsym/`    — core Rust library
- `crates/radsym-py/` — Python bindings via PyO3

### radsym modules

- `core/`         — types, image view, errors, NMS, geometry, gradient (Sobel/Scharr),
                     homography, circle fitting (Kåsa)
- `propose/`      — FRST, RSD, center voting, proposal extraction,
                     homography-aware FRST and reranking
- `support/`      — annulus sampling, profiles, scoring, hypothesis types
- `refine/`       — Parthasarathy radial center, circle/ellipse refinement,
                     homography-aware ellipse refinement
- `affine/`       — (feature-gated) affine-aware extensions (GFRS)
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
- `propose/`, `support/` depend only on `core/`
- `refine/` depends on `core/` and `support/`
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

## Versioning

- **Single source of truth**: `[workspace.package] version` in root `Cargo.toml`
- `crates/radsym/Cargo.toml` and `crates/radsym-py/Cargo.toml` use
  `version.workspace = true`
- `crates/radsym-py/pyproject.toml` uses `dynamic = ["version"]` so maturin
  reads the version from `Cargo.toml` automatically — **never hardcode a
  version in pyproject.toml**
- When bumping the version: update **only** root `Cargo.toml` and `CHANGELOG.md`

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
| Kåsa circle fit | `core::circle_fit` | Kåsa, IEEE T-IM 1976 |
| Ellipse refinement | `refine::ellipse` | Fitzgibbon et al., TPAMI 1999 |
| Homography FRST | `propose::homography` | Novel: FRST in rectified space |
| Homography refinement | `refine::homography` | Novel: circle fit in rectified space |
