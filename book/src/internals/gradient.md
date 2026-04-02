# Gradient Computation

## Overview

The gradient module provides Sobel-based image gradient computation, which is
the first step in nearly every radsym pipeline. All proposal and refinement
algorithms operate on the precomputed gradient field rather than the raw image.

## Sobel 3x3 operator

Two functions are provided:

- `sobel_gradient` -- accepts an `ImageView<u8>` (8-bit grayscale).
- `sobel_gradient_f32` -- accepts an `ImageView<f32>` (floating-point).

Both apply the standard 3x3 Sobel kernels:

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \qquad
  G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix},$$

producing horizontal ($\partial I / \partial x$) and vertical
($\partial I / \partial y$) derivatives respectively.

## GradientField

The result is stored in a `GradientField` struct with two `OwnedImage<f32>`
buffers:

- `gx` -- horizontal gradient. Positive values indicate intensity increasing
  rightward.
- `gy` -- vertical gradient. Positive values indicate intensity increasing
  downward.

Both buffers have the same dimensions as the input image. Accessor methods
`gx()` and `gy()` return borrowed `ImageView` slices for zero-copy downstream
use.

## Border handling

Border pixels (the outermost 1-pixel ring) are set to zero, since the 3x3
kernel requires a 1-pixel neighborhood that is not available at the boundary.
This is consistent across both `u8` and `f32` variants.
