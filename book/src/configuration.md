# Configuration Guide

All radsym algorithms are controlled through config structs with public fields
and sensible defaults. This page collects every config struct, organized by
pipeline stage.

Most config structs are `#[non_exhaustive]`: construct them with
`Config::default()` plus field assignment (or a dedicated builder) rather than
a struct literal. The detector and refinement configs are split into a small
*stable* config plus an `advanced` sub-config that holds algorithm-tuning
knobs; the tables below mark which fields live where.

## Detection stage

### DetectCirclesConfig

The aggregated config for the one-call `detect_circles` pipeline. Build it with
the `DetectCirclesConfig::for_radii` builder:

```rust
use radsym::{DetectCirclesConfig, Polarity};

let config = DetectCirclesConfig::for_radii([9, 10, 11])
    .polarity(Polarity::Bright)
    .radius_hint(10.0)
    .min_score(0.2);
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `radii` | `Vec<u32>` | `[3, 5, 7, 9, 11]` | Candidate FRST voting radii (pixels). Source of truth for the radii the pipeline votes over |
| `polarity` | `Polarity` | `Both` | Which polarity to detect |
| `radius_hint` | `f32` | `10.0` | Expected radius used as the initial circle hypothesis |
| `min_score` | `f32` | `0.0` | Minimum support score to keep a detection |
| `gradient_operator` | `GradientOperator` | `Sobel` | Gradient operator for the pipeline |
| `advanced` | `DetectCirclesAdvanced` | (see below) | Advanced per-stage configuration |

`DetectCirclesAdvanced` bundles the per-stage configs assembled by the
pipeline: `frst: FrstConfig`, `nms: NmsConfig`, `scoring: ScoringConfig`, and
`refinement: CircleRefineConfig`. The pipeline overrides `advanced.frst.radii`
and `advanced.frst.polarity` with the top-level `radii` and `polarity`, so most
callers only touch the stable top-level fields.

## Proposal stage

### FrstConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `radii` | `Vec<u32>` | `[3, 5, 7, 9, 11]` | Discrete radii to test (pixels) |
| `alpha` | `f32` | `2.0` | Radial strictness exponent |
| `gradient_threshold` | `f32` | `0.0` | Minimum gradient magnitude for voting |
| `polarity` | `Polarity` | `Both` | Target polarity (Bright, Dark, Both) |
| `smoothing_factor` | `f32` | `0.5` | Gaussian sigma = factor * radius |

### RsdConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `radii` | `Vec<u32>` | `[3, 5, 7, 9, 11]` | Discrete radii to test (pixels) |
| `gradient_threshold` | `f32` | `0.0` | Minimum gradient magnitude for voting |
| `polarity` | `Polarity` | `Both` | Target polarity |
| `smoothing_factor` | `f32` | `0.5` | Gaussian sigma = factor * radius |

### NmsConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `radius` | `usize` | `5` | Suppression half-window size (pixels) |
| `threshold` | `f32` | `0.0` | Minimum response to consider |
| `max_detections` | `usize` | `1000` | Budget cap on returned peaks |

### HomographyRerankConfig

Stable fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_radius` | `f32` | `6.0` | Minimum image-space radius to evaluate |
| `max_radius` | `f32` | `0.0` | Maximum radius (`<= min_radius` = auto) |
| `radius_step` | `f32` | `2.0` | Coarse radius sweep step (pixels) |
| `advanced` | `HomographyRerankAdvanced` | (see below) | Advanced edge-acquisition and prior knobs |

`HomographyRerankAdvanced` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ray_count` | `usize` | `64` | Image-space rays for edge acquisition |
| `radial_search_inner` | `f32` | `0.6` | Inner factor for radial search |
| `radial_search_outer` | `f32` | `1.45` | Outer factor for radial search |
| `size_prior_sigma` | `f32` | `0.22` | Gaussian size prior width |
| `center_prior_sigma_fraction` | `f32` | `0.45` | Center prior sigma as fraction of image size |

## Scoring stage

### AnnulusSamplingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_angular_samples` | `usize` | `64` | Angular sample count around annulus |
| `num_radial_samples` | `usize` | `9` | Radial sample count across annulus width |

### ScoringConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sampling` | `AnnulusSamplingConfig` | (see above) | Annulus sampling parameters |
| `annulus_margin` | `f32` | `0.3` | Fractional width around hypothesized radius |
| `min_samples` | `usize` | `8` | Minimum samples to avoid degeneracy |
| `weight_ringness` | `f32` | `0.6` | Weight for gradient alignment component |
| `weight_coverage` | `f32` | `0.4` | Weight for angular coverage component |

## Refinement stage

### RadialCenterConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patch_radius` | `usize` | `12` | Half-width of the analysis patch (pixels) |
| `gradient_threshold` | `f32` | `1e-4` | Minimum gradient magnitude for fit |

### CircleRefineConfig

Stable fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | `usize` | `10` | Maximum refinement iterations |
| `convergence_tol` | `f32` | `0.1` | Stop when center shift < this (pixels) |
| `max_center_drift` | `f32` | `0.5` | Max total center drift as a fraction of the initial radius |
| `advanced` | `CircleRefineAdvanced` | (see below) | Advanced acquisition and sampling knobs |

`CircleRefineAdvanced` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `annulus_margin` | `f32` | `0.3` | Annulus width as fraction of radius |
| `radial_center` | `RadialCenterConfig` | (see above) | Sub-step center config |
| `sampling` | `AnnulusSamplingConfig` | (see above) | Sub-step sampling config |

### EllipseRefineConfig

Stable fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | `usize` | `5` | Maximum refinement iterations |
| `convergence_tol` | `f32` | `0.1` | Convergence threshold (pixels) |
| `max_center_shift_fraction` | `f32` | `0.4` | Max center shift / seed radius |
| `max_axis_ratio` | `f32` | `1.8` | Maximum allowed a/b ratio |
| `advanced` | `EllipseRefineAdvanced` | (see below) | Advanced edge-acquisition and sampling knobs |

`EllipseRefineAdvanced` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `annulus_margin` | `f32` | `0.3` | Annulus margin for downstream diagnostics |
| `radial_center` | `RadialCenterConfig` | (see above) | Seed stabilization config |
| `sampling` | `AnnulusSamplingConfig` | (see above) | Legacy sampling config |
| `min_alignment` | `f32` | `0.3` | Minimum alignment for legacy annulus diagnostics |
| `ray_count` | `usize` | `96` | Angular sectors for edge acquisition |
| `radial_search_inner` | `f32` | `0.6` | Inner radius factor for initial search |
| `radial_search_outer` | `f32` | `1.45` | Outer radius factor for initial search |
| `normal_search_half_width` | `f32` | `6.0` | Normal search window (pixels) |
| `min_inlier_coverage` | `f32` | `0.6` | Minimum sector coverage of inliers |

### HomographyEllipseRefineConfig

Stable fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | `usize` | `5` | Maximum refinement iterations |
| `convergence_tol` | `f32` | `0.1` | Convergence threshold (rectified pixels) |
| `max_center_shift_fraction` | `f32` | `0.4` | Max center shift / rectified radius |
| `max_radius_change_fraction` | `f32` | `0.6` | Max radius change / rectified radius |
| `advanced` | `HomographyEllipseRefineAdvanced` | (see below) | Advanced edge-acquisition and sampling knobs |

`HomographyEllipseRefineAdvanced` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `radial_center` | `RadialCenterConfig` | (see above) | Image-space seed stabilization |
| `ray_count` | `usize` | `96` | Image-space rays for edge acquisition |
| `radial_search_inner` | `f32` | `0.6` | Inner radius factor |
| `radial_search_outer` | `f32` | `1.45` | Outer radius factor |
| `normal_search_half_width` | `f32` | `6.0` | Normal search window (pixels) |
| `min_inlier_coverage` | `f32` | `0.45` | Minimum rectified angular coverage |

## Tuning tips

The advanced edge-acquisition knobs (`ray_count`, `normal_search_half_width`,
the radial-search factors) live on the `advanced` sub-config — set them as,
e.g., `config.advanced.ray_count = ...`.

**Small targets (radius < 10 px):**
Use tight radii in FrstConfig (e.g., `[3, 5, 7]`), reduce `patch_radius` in
RadialCenterConfig, and lower `advanced.normal_search_half_width` in
EllipseRefineConfig.

**Noisy images:**
Increase `gradient_threshold` in FrstConfig/RsdConfig to suppress votes from
noise. Increase `smoothing_factor` for broader peaks that survive noise.

**High-precision localization:**
Increase `advanced.ray_count` in EllipseRefineConfig for denser angular
sampling. Decrease `convergence_tol` to force more refinement iterations. Use a
larger `patch_radius` in RadialCenterConfig.

**Real-time budgets:**
Use RsdConfig instead of FrstConfig for faster proposal generation, or the
fused `frst_response_fused` / `rsd_response_fused` variants. Set
`max_detections` in NmsConfig to limit downstream work. Reduce
`advanced.ray_count` and `max_iterations` in refinement configs.

**Strong perspective distortion:**
Use the homography-aware path (HomographyEllipseRefineConfig) when a
calibrated homography is available. Otherwise, increase `max_axis_ratio` in
EllipseRefineConfig to permit more eccentric ellipses.
