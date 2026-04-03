# Homography-Aware Extensions

## Motivation

When a known planar homography relates the image to a rectified frame --
for example, a ground-plane homography from camera calibration -- circular
targets in the rectified frame appear as ellipses in the image. Rather than
detecting ellipses directly (which is harder and less reliable), the
homography-aware extensions perform detection in the rectified space where
targets are circular, then transport results back to image coordinates.

## Core types

### Homography

The `Homography` struct wraps a $3 \times 3$ matrix $H$ that maps image
coordinates to rectified coordinates:

$$\mathbf{x}_R \sim H\, \mathbf{x}_I$$

where $\mathbf{x}_I = (x, y, 1)^T$ is a point in the source image and
$\mathbf{x}_R$ is the corresponding point in the rectified frame. The struct
caches the inverse $H^{-1}$ at construction time, so forward and backward
point transformations are both available without recomputation.

### RectifiedGrid

`RectifiedGrid` defines a discrete pixel grid in rectified space. It stores
the grid dimensions and the `Homography`, providing the mapping between
rectified pixel indices and image coordinates. FRST voting and NMS operate on
this grid.

## Proposal: FRST in rectified space

The `propose::homography` module runs FRST voting on a `RectifiedGrid`:

1. For each pixel in the rectified grid, compute the corresponding image
   coordinate via $H^{-1}$ and sample the image gradient using bilinear
   interpolation.
2. Transform the gradient direction into the rectified frame using the local
   Jacobian of $H$.
3. Vote into the rectified accumulator as in standard FRST.
4. Apply Gaussian smoothing and NMS in the rectified grid to extract peaks.

Each peak is a `HomographyProposal` containing the rectified-frame center,
a scale hint (rectified radius), and the corresponding image-space ellipse
obtained by projecting the rectified circle through $H^{-1}$.

### Reranking

The `HomographyRerankConfig` provides a post-proposal reranking step. For each
proposal, it sweeps a range of rectified radii and scores edge evidence along
radial rays in image space (transformed through the homography). A Gaussian
size prior and center prior can be combined with the edge score to reorder
proposals. This is useful when FRST peaks are noisy and the expected target
size is approximately known.

## Refinement: circle fit in rectified space

The `refine::homography` module refines a homography proposal by fitting a
circle in rectified space:

1. Starting from the initial rectified circle, cast rays outward from the
   image-space ellipse along its normals.
2. Detect edge observations via gradient peak search (same mechanism as
   [ellipse refinement](ellipse-refine.md)).
3. Transform each image-space edge point into rectified coordinates via $H$.
4. Fit a circle to the rectified edge points using trimmed Gauss-Newton
   (analogous to the ellipse fitter, but with 3 parameters: $c_x, c_y, r$).
5. Project the refined rectified circle back to image space as an ellipse
   using the conic projection $C_I = H^{-T}\, C_R\, H^{-1}$, where $C_R$
   is the conic matrix of the rectified circle.

Guard constraints (maximum center shift, maximum radius change) are enforced
in rectified space, where the geometry is simpler.

## When to use

Use the homography-aware path when:

- A calibrated homography is available (e.g., road-plane from vehicle
  calibration, table-plane from a fixed camera).
- Targets are known to be circular in the world plane.
- Perspective distortion is significant enough that standard FRST or direct
  ellipse fitting underperforms.

When no homography is available, use standard [FRST](frst.md) or
[GFRS](gfrs.md) for proposal generation and
[ellipse refinement](ellipse-refine.md) for shape fitting.

## Configuration

```rust
pub struct HomographyEllipseRefineConfig {
    pub max_iterations: usize,           // 5
    pub convergence_tol: f32,            // 0.1
    pub radial_center: RadialCenterConfig,
    pub ray_count: usize,               // 96
    pub radial_search_inner: f32,        // 0.6
    pub radial_search_outer: f32,        // 1.45
    pub normal_search_half_width: f32,   // 6.0
    pub min_inlier_coverage: f32,        // 0.45
    pub max_center_shift_fraction: f32,  // 0.4
    pub max_radius_change_fraction: f32, // 0.6
}
```

```rust
pub struct HomographyRerankConfig {
    pub min_radius: f32,                  // 6.0
    pub max_radius: f32,                  // 0.0 (auto)
    pub radius_step: f32,                 // 2.0
    pub ray_count: usize,                // 64
    pub radial_search_inner: f32,         // 0.6
    pub radial_search_outer: f32,         // 1.45
    pub size_prior_sigma: f32,            // 0.22
    pub center_prior_sigma_fraction: f32, // 0.45
}
```
