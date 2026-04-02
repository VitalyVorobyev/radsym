# Gaussian Blur

## Overview

Gaussian blur is applied to FRST and RSD response maps before peak extraction.
The blur module provides an in-place implementation with two execution paths
selected automatically based on kernel size.

## Dual-mode execution

- **Direct convolution** (sigma <= 2.0): a separable 1D Gaussian kernel with
  radius $\lceil 3\sigma \rceil$ is applied horizontally then vertically. This
  is straightforward and efficient for small kernels.

- **Stacked box blur** (sigma > 2.0): three successive passes of a uniform
  box filter approximate the Gaussian. Each pass is O(1) per pixel using a
  sliding-window accumulator, making the total cost independent of sigma.
  The box widths are chosen so that the cascade converges to the target
  Gaussian in the limit (Wells 1986; W3C Filter Effects Module Level 1,
  Section 12.2).

When sigma <= 0.5, the kernel radius would be zero, so the function is a
no-op.

## Boundary handling

Both paths use **mirror-clamp** boundary extension: samples outside the image
are reflected back from the nearest edge. This avoids dark halos at image
borders that would bias peak detection near the edges.

## Usage

The blur function is `pub(crate)` and called internally by the FRST and RSD
response computation. It operates in-place on an `OwnedImage<f32>`, requiring
no additional output buffer.
