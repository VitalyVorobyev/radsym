# Installation and Quick Start

## Adding radsym to Your Project

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
radsym = "0.1"
```

By default, no optional features are enabled. The core detection pipeline works
out of the box with zero additional dependencies beyond `nalgebra` and
`thiserror`.

## Feature Flags

| Flag | What it enables |
|------|-----------------|
| `rayon` | Parallel multi-radius FRST voting and batch scoring via Rayon |
| `image-io` | `load_grayscale` file I/O and `save_diagnostic` export via the `image` crate |
| `tracing` | Structured log spans and events via the `tracing` crate |
| `affine` | Experimental affine-aware extensions (GFRS, Ni et al. CVPR 2012) |
| `serde` | `Serialize` / `Deserialize` derives on all config and result types |

Enable features in `Cargo.toml` as needed:

```toml
[dependencies]
radsym = { version = "0.1", features = ["rayon", "image-io"] }
```

## Quick Start: the Composable Pipeline

The standard workflow uses root-level imports and explicit function calls for
each stage:

```rust
use radsym::{
    ImageView, FrstConfig, Circle, Polarity, ScoringConfig, NmsConfig,
    sobel_gradient, frst_response, extract_proposals, score_circle_support,
    refine_circle, CircleRefineConfig,
};

// 1. Build an ImageView from your pixel buffer (row-major, single-channel u8).
let image = ImageView::from_slice(&pixel_data, width, height).unwrap();

// 2. Compute the Sobel gradient field.
let gradient = sobel_gradient(&image).unwrap();

// 3. Run FRST voting across a range of candidate radii.
let frst_cfg = FrstConfig {
    radii: vec![15, 16, 17, 18, 19, 20],
    ..FrstConfig::default()
};
let response = frst_response(&gradient, &frst_cfg).unwrap();

// 4. Extract proposals via non-maximum suppression.
let nms = NmsConfig { radius: 5, threshold: 0.0, max_detections: 10 };
let proposals = extract_proposals(&response, &nms, Polarity::Bright);

// 5. Score and refine each proposal.
for proposal in &proposals {
    let circle = Circle::new(proposal.seed.position, 18.0);
    let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
    if score.total > 0.3 {
        let result = refine_circle(&gradient, &circle, &CircleRefineConfig::default());
        // result.hypothesis contains the refined Circle
    }
}
```

## Quick Start: the One-Call API

If you do not need per-stage control, `detect_circles` runs the entire
pipeline in a single call:

```rust
use radsym::{ImageView, FrstConfig, Polarity, detect_circles, DetectCirclesConfig};

let image = ImageView::from_slice(&pixel_data, width, height).unwrap();
let config = DetectCirclesConfig {
    frst: FrstConfig { radii: vec![15, 16, 17, 18, 19, 20], ..FrstConfig::default() },
    polarity: Polarity::Bright,
    radius_hint: 18.0,
    min_score: 0.3,
    ..DetectCirclesConfig::default()
};

let detections = detect_circles(&image, &config).unwrap();
for det in &detections {
    println!("center=({:.1}, {:.1}) r={:.1} score={:.3}",
        det.hypothesis.center.x, det.hypothesis.center.y,
        det.hypothesis.radius, det.score.total);
}
```

`detect_circles` returns a `Vec<Detection<Circle>>` sorted by descending
support score. Each `Detection` carries the refined `Circle`, a `SupportScore`,
and a `RefinementStatus` indicating convergence.

## Coordinate Convention

radsym uses **(x = column, y = row)** everywhere.
$x$ increases rightward, $y$ increases downward — consistent with
`nalgebra::Point2<f32>`, which is type-aliased as `PixelCoord`. All public API
functions expect and return coordinates in this convention.
