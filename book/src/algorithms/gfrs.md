# GFRS: Generalized Fast Radial Symmetry

## Overview

GFRS extends [FRST](frst.md) to detect elliptical symmetry under affine
distortion. When a circular target is viewed under perspective, it projects to
an ellipse. Standard FRST misses such targets because gradient directions no
longer converge to a single center at a single radius. GFRS recovers the
detection by warping gradient directions through a set of affine maps before
voting.

## Algorithm

For a sampled set of 2D affine transformations $\{A_k\}$:

1. At each pixel $p$ with gradient direction $\hat{\mathbf{g}}(p)$, compute the
   warped direction $A_k\, \hat{\mathbf{g}}(p)$.
2. Compute the affected pixel offset using the warped direction and vote into a
   per-map accumulator.
3. Smooth each accumulator and record its peak response.

The map $A_k$ that produces the strongest peak gives both the center location
and an estimate of the affine distortion. The distortion matrix directly yields
the ellipse semi-axes and orientation.

## Configuration

```rust
pub struct AffineFrstConfig {
    /// Single voting radius (pixels).
    pub radius: u32,
    /// Minimum gradient magnitude to vote.
    pub gradient_threshold: f32,
    /// Gaussian sigma = smoothing_factor * radius.
    pub smoothing_factor: f32,
    /// Set of affine maps to evaluate.
    pub affine_maps: Vec<AffineMap>,
}
```

The number of affine maps controls the trade-off between angular/scale
resolution and computation time. Typical usage samples 20--50 maps covering
the expected range of perspective distortion.

## Feature gate

GFRS is gated behind the `affine` feature flag:

```toml
radsym = { version = "...", features = ["affine"] }
```

## Reference

Ni, K., Singh, M., Bahlmann, C. "Fast radial symmetry detection under affine
transformations." *IEEE Conference on Computer Vision and Pattern Recognition
(CVPR)*, 932--939 (2012).
doi:[10.1109/CVPR.2012.6247767](https://doi.org/10.1109/CVPR.2012.6247767)
