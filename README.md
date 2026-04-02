# radsym

[![CI](https://github.com/VitalyVorobyev/radsym/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/radsym/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/radsym.svg)](https://crates.io/crates/radsym)
[![docs.rs](https://docs.rs/radsym/badge.svg)](https://docs.rs/radsym)
[![License](https://img.shields.io/crates/l/radsym.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://blog.rust-lang.org/2025/06/26/Rust-1.88.0/)

Radial symmetry detection in Rust. Fast gradient-based center voting,
local support scoring, and subpixel refinement — composable tools for
finding circles and ellipses in images.

## Why radsym?

- **Fast** — FRST and RSD proposals run in a single pass over the gradient field
- **Accurate** — Parthasarathy radial center achieves subpixel localization
- **Composable** — use any stage independently: propose, score, refine
- **Minimal deps** — `nalgebra` + `thiserror`, nothing else required
- **Deterministic** — same input always produces the same output

## Pipeline overview

```
Image --> Gradient --> Proposals --> Support scoring --> Refinement
           Sobel       FRST/RSD     annulus sampling     radial center
                                    coverage analysis    circle/ellipse fit
```

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
radsym = "0.1"
```

Detect circles in an image:

```rust
use radsym::{
    ImageView, FrstConfig, Circle, Polarity, ScoringConfig, NmsConfig,
    sobel_gradient, frst_response, extract_proposals, score_circle_support,
};

let size = 64;
let mut data = vec![0u8; size * size];
for y in 0..size {
    for x in 0..size {
        let dx = x as f32 - 32.0;
        let dy = y as f32 - 32.0;
        if (dx * dx + dy * dy).sqrt() <= 10.0 {
            data[y * size + x] = 255;
        }
    }
}

let image = ImageView::from_slice(&data, size, size).unwrap();
let gradient = sobel_gradient(&image).unwrap();

let config = FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() };
let response = frst_response(&gradient, &config).unwrap();

let nms = NmsConfig { radius: 5, threshold: 0.0, max_detections: 5 };
let proposals = extract_proposals(&response, &nms, Polarity::Bright);

if let Some(best) = proposals.first() {
    let circle = Circle::new(best.seed.position, 10.0);
    let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
    assert!(score.total > 0.0);
}
```

## Modules

| Module | Purpose |
|--------|---------|
| `core` | Types, image views, geometry, gradient, NMS |
| `propose` | Center voting: FRST (Loy & Zelinsky 2002), RSD (Barnes et al. 2008) |
| `support` | Annulus gradient sampling, angular coverage, support scoring |
| `refine` | Parthasarathy radial center (2012), iterative circle/ellipse fit |
| `diagnostics` | Heatmap rendering, shape overlays |
| `affine` | GFRS elliptical voting (Ni et al. 2012) -- feature-gated |

## Feature flags

All features are opt-in. The default build has zero optional dependencies.

| Feature | What it enables |
|---------|----------------|
| `rayon` | Parallel multi-radius proposal computation |
| `image-io` | PNG/JPEG image loading via the `image` crate |
| `tracing` | Structured logging |
| `affine` | Experimental affine-aware extensions (GFRS) |
| `serde` | Serialization for configs and results |

## Conventions

- **Coordinates**: `(x=col, y=row)` everywhere, matching `nalgebra::Point2<f32>`
- **Precision**: `f32` for all pixel-level computation
- **Images**: borrow-based `ImageView<'a, T>` with stride support

## Algorithm references

| Algorithm | Paper |
|-----------|-------|
| FRST | Loy, G. & Zelinsky, A. *Fast Radial Symmetry for Detecting Points of Interest.* IEEE TPAMI 25(8), 2003 |
| RSD | Barnes, N., Zelinsky, A., Fletcher, L.S. *Real-time speed sign detection using the radial symmetry detector.* IEEE T-ITS 9(2), 2008 |
| Radial center | Parthasarathy, R. *Rapid, accurate particle tracking by calculation of radial symmetry centers.* Nature Methods 9, 2012 |
| GFRS | Ni, K., Singh, M., Bahlmann, C. *Fast Radial Symmetry Detection Under Affine Transformations.* CVPR, 2012 |

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

at your option.
