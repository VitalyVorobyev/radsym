# radsym Design Document

## Goals

radsym provides composable building blocks for detecting radially symmetric
structures (circles, rings, ellipses) in grayscale images. It is not a
monolithic pipeline — each stage (proposal, scoring, refinement) is a
standalone function that can be used independently or composed.

Primary use cases:

- Calibration target detection (fiducial rings, circular markers)
- Particle tracking (fluorescence microscopy)
- Industrial inspection (circular features, holes, gaskets)

## Architecture

### Single-crate module layout

```
radsym/
  core/         -- types, image, geometry, gradient, NMS (no algorithm deps)
  propose/      -- center voting: FRST, RSD, proposal extraction
  support/      -- annulus sampling, coverage, scoring
  refine/       -- radial center, circle/ellipse iterative refinement
  diagnostics/  -- heatmaps, overlays (visualization only)
  affine/       -- GFRS affine-aware voting (feature-gated)
```

The dependency graph is strictly layered:

```
core  <--  propose
core  <--  support
core  <--  refine
core + propose  <--  affine
all modules  <--  diagnostics
```

This was a deliberate choice over a multi-crate workspace. A single crate
with modules avoids premature API boundary decisions while keeping clean
internal separation. See [ADR-006](decisions/006-single-crate.md).

### Data flow

The canonical pipeline for circle detection:

1. **Image input**: `ImageView<'_, u8>` — borrowed, zero-copy, stride-aware
2. **Gradient**: `sobel_gradient()` -> `GradientField` (gx, gy as `OwnedImage`)
3. **Proposals**: `frst_response()` -> `OwnedImage` -> `extract_proposals()` -> `Vec<Proposal>`
4. **Scoring**: `score_circle_support()` -> `SupportScore` (ringness + coverage)
5. **Refinement**: `refine_circle()` -> `RefinementResult<Circle>`

Each step is a pure function. No global state, no configuration singletons.

### Image abstraction

`ImageView<'a, T>` is a concrete struct (not a trait) that borrows pixel
data with an explicit stride. This follows the pattern from the
[corrmatch-rs](https://github.com/VitalyVorobyev/corrmatch-rs) sibling
project. Benefits:

- No trait-object overhead or monomorphization explosion
- Stride support enables ROI views without copying
- Bilinear `sample()` for subpixel access (f32 and u8)

`OwnedImage<T>` provides the owned counterpart for algorithm outputs
(gradient fields, response maps).

### Coordinate system

`(x=col, y=row)` everywhere, using `nalgebra::Point2<f32>` as the
coordinate type (`PixelCoord`). This matches image processing convention
where x increases rightward and y increases downward.

## Algorithm families

### FRST — Fast Radial Symmetry Transform

Source: Loy & Zelinsky, IEEE TPAMI 2003

Each pixel votes along its gradient direction at a fixed radius offset.
Two accumulators per radius:
- `O_n` — orientation projection (consistency of voting direction)
- `M_n` — magnitude projection (strength of evidence)

Combined as `F_n = |O_n|^alpha * M_n`, smoothed, summed across radii.
Polarity-selective (bright-on-dark, dark-on-bright, or both).

### RSD — Radial Symmetry Detector

Source: Barnes, Zelinsky, Fletcher, IEEE T-ITS 2008

Simplified FRST: drops the orientation accumulator, votes magnitude only.
~2x faster at the cost of weaker discrimination against non-symmetric
structures. Suitable for real-time applications.

### Parthasarathy radial center

Source: Parthasarathy, Nature Methods 2012

Non-iterative subpixel center refinement. Each gradient pixel defines a
line through the center; the center is the weighted least-squares
intersection of all such lines. Two variants:

- **Reference**: Roberts-cross gradient on half-pixel grid (matching paper)
- **Production**: Uses precomputed Sobel gradient field

### GFRS — Generalized Fast Radial Symmetry

Source: Ni, Singh, Bahlmann, CVPR 2012

Extends FRST by warping gradient directions through a discrete set of
affine maps before voting. Each map produces a separate accumulator;
the map with the strongest peak indicates the affine distortion
(and hence the ellipse parameters). Feature-gated under `affine`.

## Support scoring

The support score quantifies how strongly local image evidence supports
a geometric hypothesis. Two components:

- **Ringness** (weight 0.6): mean radial alignment of gradient vectors
  sampled along an annulus around the hypothesis
- **Angular coverage** (weight 0.4): fraction of angular bins with
  well-aligned gradient evidence

A score is flagged as degenerate if fewer than 8 gradient samples
are found (empty region, edge of image).

## Testing strategy

All algorithms are tested on synthetic images:
- Gaussian blobs for radial center accuracy
- Binary rings for annulus sampling and scoring
- Multi-target images for proposal extraction
- Uniform images for degeneracy detection

Tests verify both correctness (center accuracy, convergence) and
discrimination (on-center vs off-center, full circle vs partial arc).

## Performance considerations

- No allocations in inner voting loops
- Separable Gaussian blur for O(n) smoothing
- Gradient magnitude squared weighting (avoids sqrt in inner loop
  where possible)
- Optional rayon parallelism for multi-radius computation
- Deterministic output ordering for reproducibility
