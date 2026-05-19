# Installation and Quick Start

## Adding radsym to Your Project

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
radsym = "0.2"
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
radsym = { version = "0.2", features = ["rayon", "image-io"] }
```

## Quick Start: the One-Call API

For the common case, `detect_circles` runs the entire propose-score-refine
pipeline in a single call. Build its config with the
`DetectCirclesConfig::for_radii` builder:

```rust
use radsym::{detect_circles, DetectCirclesConfig, ImageView, Polarity};

// Build an ImageView from your pixel buffer (row-major, single-channel u8).
let image = ImageView::from_slice(&pixel_data, width, height).unwrap();

let config = DetectCirclesConfig::for_radii([15, 16, 17, 18, 19, 20])
    .polarity(Polarity::Bright)
    .radius_hint(18.0)
    .min_score(0.3);

let detections = detect_circles(&image, &config).unwrap();
for det in &detections {
    println!("center=({:.1}, {:.1}) r={:.1} score={:.3}",
        det.hypothesis.center.x, det.hypothesis.center.y,
        det.hypothesis.radius, det.score.total);
}
```

`detect_circles` returns a `Vec<CircleDetection>` (an alias for
`Vec<Detection<Circle>>`) sorted by descending support score. Each
`CircleDetection` carries the refined `Circle`, a `SupportScore`, and a
`RefinementStatus` indicating convergence.

`DetectCirclesConfig` exposes the stable detection knobs — `radii`, `polarity`,
`radius_hint`, `min_score`, `gradient_operator` — plus an `advanced` field
holding the per-stage FRST/NMS/scoring/refinement configs for power users. The
struct is `#[non_exhaustive]`; build it through `for_radii` (or
`DetectCirclesConfig::default()` plus field assignment), not a struct literal.

To inspect the response map, raw proposals, rejected candidates, and
per-detection score breakdowns, call `detect_circles_with_diagnostics`, which
returns the same `Vec<CircleDetection>` plus a `CircleDetectionDiagnostics`.

## Quick Start: the Composable Pipeline

When you need per-stage control, drive the propose-score-refine stages with
explicit function calls:

```rust
use radsym::{
    Circle, CircleRefineConfig, FrstConfig, ImageView, NmsConfig, Polarity,
    ScoringConfig, extract_proposals, frst_response, refine_circle,
    score_circle_support, sobel_gradient,
};

// 1. Build an ImageView from your pixel buffer (row-major, single-channel u8).
let image = ImageView::from_slice(&pixel_data, width, height).unwrap();

// 2. Compute the Sobel gradient field.
let gradient = sobel_gradient(&image).unwrap();

// 3. Run FRST voting across a range of candidate radii. FrstConfig is
//    `#[non_exhaustive]`: start from the default and assign fields.
let mut frst_cfg = FrstConfig::default();
frst_cfg.radii = vec![15, 16, 17, 18, 19, 20];
let response = frst_response(&gradient, &frst_cfg).unwrap();

// 4. Extract proposals via non-maximum suppression.
let mut nms = NmsConfig::default();
nms.radius = 5;
nms.max_detections = 10;
let proposals = extract_proposals(&response, &nms, Polarity::Bright);

// 5. Score and refine each proposal. score_circle_support returns a
//    SupportScoreBreakdown; `.total` is the headline score.
for proposal in &proposals {
    let circle = Circle::new(proposal.seed.position, 18.0);
    let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
    if score.total > 0.3 {
        let result = refine_circle(&gradient, &circle, &CircleRefineConfig::default());
        // result.hypothesis contains the refined Circle
    }
}
```

## Coordinate Convention

radsym uses **(x = column, y = row)** everywhere.
$x$ increases rightward, $y$ increases downward — consistent with
`nalgebra::Point2<f32>`, which is type-aliased as `PixelCoord`. All public API
functions expect and return coordinates in this convention.
