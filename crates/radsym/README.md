# radsym

`radsym` is a Rust library for radial symmetry detection: center proposal
generation, local circular and elliptical support analysis, support scoring,
and local image-space refinement.

The crate is CPU-first, deterministic, and composable. You can use proposal
generation, scoring, and refinement independently instead of committing to a
single end-to-end pipeline.

## Installation

```toml
[dependencies]
radsym = "0.2"
```

## Quick start

The one-call `detect_circles` runs the full propose-score-refine pipeline.
Configure it with the `DetectCirclesConfig::for_radii` builder:

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
score. Use `detect_circles_with_diagnostics` to also receive a
`CircleDetectionDiagnostics` with the response map, raw proposals, rejected
candidates, and per-detection `SupportScoreBreakdown`.

For per-stage control, the propose-score-refine stages remain available as
free functions: `sobel_gradient`, `frst_response` / `rsd_response` (and the
fused `frst_response_fused` / `rsd_response_fused`), `extract_proposals`,
`score_circle_support` / `score_ellipse_support`, and `refine_circle` /
`refine_ellipse`.

## Modules

- `core`: image views, geometry, gradients, NMS, pyramids, Kåsa circle fit,
  errors
- `propose`: FRST, RSD, proposal extraction, homography-aware proposal tools
- `support`: annulus sampling, support evidence, support scoring
- `refine`: radial center, circle refinement, ellipse refinement
- `diagnostics`: response heatmaps, overlays, detection diagnostics
- `affine`: experimental affine-aware extensions behind the `affine` feature

The crate root carries the common detect-score-refine contract. Low-level
helpers stay under their module paths — circle fitting under
`core::circle_fit`, pyramids under `core::pyramid`, raw support evidence under
`support::evidence`, and visualization under `diagnostics`.

Most config and result structs are `#[non_exhaustive]`: construct them via
`Config::default()` plus field assignment rather than struct literals.

## Feature flags

- `rayon`: parallel multi-radius proposal computation
- `image-io`: PNG and JPEG image loading via `image`
- `tracing`: structured instrumentation
- `affine`: experimental affine-aware extensions
- `serde`: serialization for configs and results

## Conventions

- Coordinates use `(x=col, y=row)` everywhere.
- Pixel-level computation uses `f32`.
- `ImageView<'a, T>` is the core borrowed image abstraction.

## References

- FRST: Loy and Zelinsky, IEEE TPAMI 2003
- RSD: Barnes, Zelinsky, Fletcher, IEEE T-ITS 2008
- Radial center: Parthasarathy, Nature Methods 2012
- GFRS: Ni, Singh, Bahlmann, CVPR 2012

## License

Licensed under MIT OR Apache-2.0.
