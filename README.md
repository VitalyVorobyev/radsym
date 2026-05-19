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
radsym = "0.2"
```

Detect circles in an image with the one-call pipeline:

```rust
use radsym::{detect_circles, DetectCirclesConfig, ImageView, Polarity};

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

let config = DetectCirclesConfig::for_radii([9, 10, 11])
    .polarity(Polarity::Bright)
    .radius_hint(10.0)
    .min_score(0.2);
let detections = detect_circles(&image, &config).unwrap();

for det in &detections {
    println!(
        "center=({:.1}, {:.1}) r={:.1} score={:.3}",
        det.hypothesis.center.x, det.hypothesis.center.y,
        det.hypothesis.radius, det.score.total,
    );
}
```

`detect_circles` returns `Vec<CircleDetection>` sorted by descending support
score. Each `CircleDetection` carries the refined `Circle`, a `SupportScore`,
and a `RefinementStatus`. Need the response map, raw proposals, rejected
candidates, and per-detection score breakdowns? Call
`detect_circles_with_diagnostics`, which returns the same result plus a
`CircleDetectionDiagnostics`.

### Composable stages

For per-stage control, drive the propose-score-refine stages directly:

```rust
use radsym::{
    sobel_gradient, frst_response, extract_proposals, score_circle_support,
    Circle, FrstConfig, NmsConfig, Polarity, ScoringConfig,
};

let image = ImageView::from_slice(&data, size, size).unwrap();
let gradient = sobel_gradient(&image).unwrap();

let mut config = FrstConfig::default();
config.radii = vec![9, 10, 11];
let response = frst_response(&gradient, &config).unwrap();

let mut nms = NmsConfig::default();
nms.radius = 5;
let proposals = extract_proposals(&response, &nms, Polarity::Bright);

if let Some(best) = proposals.first() {
    let circle = Circle::new(best.seed.position, 10.0);
    // score_circle_support returns a SupportScoreBreakdown; `.total` is the
    // headline score, `.ringness` / `.angular_coverage` the components.
    let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
    assert!(score.total >= 0.0);
}
```

The fused, magnitude-only proposal variant is `frst_response_fused` (likewise
`rsd_response_fused`). Most config and result structs are `#[non_exhaustive]`,
so construct them via `Config::default()` plus field assignment rather than
struct literals.

## Modules

| Module | Purpose |
|--------|---------|
| `core` | Types, image views, geometry, gradient, NMS, pyramids, Kåsa circle fit |
| `propose` | Center voting: FRST (Loy & Zelinsky 2002), RSD (Barnes et al. 2008) |
| `support` | Annulus gradient sampling, support scoring |
| `refine` | Parthasarathy radial center (2012), iterative circle/ellipse fit |
| `diagnostics` | Heatmap rendering, shape overlays, detection diagnostics |
| `affine` | GFRS elliptical voting (Ni et al. 2012) -- feature-gated |

The crate root re-exports the common detect-score-refine contract. Low-level
helpers stay under their module paths: circle fitting under
`core::circle_fit`, pyramids under `core::pyramid`, support evidence under
`support::evidence`, and visualization under `diagnostics`.

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
