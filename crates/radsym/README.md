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
radsym = "0.1"
```

## Quick start

```rust
use radsym::{Circle, FrstConfig, ImageView, Polarity, ScoringConfig};
use radsym::core::gradient::sobel_gradient;
use radsym::core::nms::NmsConfig;
use radsym::propose::extract::{extract_proposals, ResponseMap};
use radsym::propose::seed::ProposalSource;
use radsym::support::score::score_circle_support;

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

let config = FrstConfig {
    radii: vec![9, 10, 11],
    ..FrstConfig::default()
};
let response = radsym::frst_response(&gradient, &config).unwrap();

let response_map = ResponseMap::new(response, ProposalSource::Frst);
let nms = NmsConfig {
    radius: 5,
    threshold: 0.0,
    max_detections: 5,
};
let proposals = extract_proposals(&response_map, &nms, Polarity::Bright);

if let Some(best) = proposals.first() {
    let circle = Circle::new(best.seed.position, 10.0);
    let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
    assert!(score.total > 0.0);
}
```

## Modules

- `core`: image views, geometry, gradients, NMS, errors
- `propose`: FRST, RSD, proposal extraction, homography-aware proposal tools
- `support`: annulus sampling, evidence extraction, support scoring
- `refine`: radial center, circle refinement, ellipse refinement
- `diagnostics`: response heatmaps and overlays
- `affine`: experimental affine-aware extensions behind the `affine` feature

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
