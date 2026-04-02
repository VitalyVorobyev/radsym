# Annulus Sampling

## Overview

Annulus sampling extracts local gradient evidence from a ring-shaped region
around a hypothesis center. This evidence feeds into the
[support scoring](scoring.md) stage, which determines how strongly the image
supports the hypothesis.

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

Two sampling functions are provided:

- `sample_annulus` -- circular annulus, parameterized by center and inner/outer
  radii.
- `sample_elliptical_annulus` -- elliptical annulus, parameterized by an
  `Ellipse` and margin. The radial direction at each angle is computed from the
  ellipse geometry, stretching sample placement along the appropriate axes.

Both return a `SupportEvidence` struct containing the individual gradient
samples, the total sample count, and the mean gradient alignment.

## Configuration

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
