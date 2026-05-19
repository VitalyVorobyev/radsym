# Annulus Sampling

## Overview

Annulus sampling extracts local gradient evidence from a ring-shaped region
around a hypothesis center. This evidence feeds into the
[support scoring](scoring.md) stage, which determines how strongly the image
supports the hypothesis.

Annulus sampling is an *internal stage* of support scoring: the sampling
routines themselves are crate-private. Callers invoke it indirectly through
`score_circle_support` / `score_ellipse_support`, which is configured via the
`AnnulusSamplingConfig` carried by `ScoringConfig`. The resulting
`SupportEvidence` is a diagnostic type, available under
`radsym::support::evidence`.

## Sampling strategy

The annulus is defined by an inner radius and an outer radius (typically
computed as $(1 - m) \cdot r$ and $(1 + m) \cdot r$, where $r$ is the
hypothesized radius and $m$ is the `annulus_margin`). Sample points are placed
on a regular grid in polar coordinates:

- **Angular axis**: `num_angular_samples` evenly spaced angles in $[0, 2\pi)$.
- **Radial axis**: `num_radial_samples` linearly interpolated offsets between
  the inner and outer radii.

For each sample at angle $\theta$ and radius $\rho$, the image-space
coordinates are

$$x = c_x + \rho \cos\theta, \qquad y = c_y + \rho \sin\theta.$$

## Gradient readout

At each sample point, the gradient $(g_x, g_y)$ is read from the precomputed
`GradientField` using **bilinear interpolation**, giving sub-pixel accuracy.
Samples that fall outside image bounds are silently skipped.

The radial alignment at each sample is

$$a = \frac{|g_x \cos\theta + g_y \sin\theta|}{|\mathbf{g}|},$$

measuring how well the gradient direction agrees with the radial direction from
the center. A perfect ring gives $a = 1$ at every sample.

## Circular and elliptical variants

Two sampling strategies back the scoring functions:

- A **circular annulus** -- parameterized by center and inner/outer radii.
  Used by `score_circle_support`.
- An **elliptical annulus** -- parameterized by an `Ellipse` and margin. The
  radial direction at each angle is computed from the ellipse geometry,
  stretching sample placement along the appropriate axes. Used by
  `score_ellipse_support`.

Both produce a `SupportEvidence` struct containing the individual gradient
samples (`GradientSample`), the total sample count, and the mean gradient
alignment. `SupportEvidence` and `GradientSample` are diagnostic types exposed
under `radsym::support::evidence`.

## Configuration

`AnnulusSamplingConfig` is `#[non_exhaustive]`; construct it from
`AnnulusSamplingConfig::default()` and assign fields. Its fields are:

```rust
pub struct AnnulusSamplingConfig {
    /// Number of angular samples around the annulus. Default: 64.
    pub num_angular_samples: usize,
    /// Number of radial samples across the annulus width. Default: 9.
    pub num_radial_samples: usize,
}
```

The total number of sample points is `num_angular_samples * num_radial_samples`
(576 by default). Increasing angular samples improves coverage estimation;
increasing radial samples captures more of the annulus width, which is useful
for thick rings or uncertain radius estimates.

Validation requires `num_angular_samples >= 4` and `num_radial_samples >= 1`.
