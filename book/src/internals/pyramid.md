# Image Pyramids

## Overview

The pyramid module provides multi-scale image access via the `box-image-pyramid`
crate. Downsampled images allow detection of large radial features without
requiring proportionally large voting radii.

## PyramidWorkspace

`PyramidWorkspace` is a reusable buffer that avoids reallocating intermediate
downsample storage on every frame. Construct it once, then call
`workspace.level(image, level)` to extract a borrowed view at the requested
pyramid level. Level 0 is the original resolution; each subsequent level halves
both dimensions using a box filter.

```rust
let mut workspace = PyramidWorkspace::new();
let level2 = workspace.level(image, 2)?;
// level2.image() is an ImageView at 1/4 resolution
```

## One-shot downsampling

For cases where workspace reuse is not needed, `pyramid_level_owned` extracts
a single pyramid level as an owned image:

```rust
let downsampled = pyramid_level_owned(&image, 2)?;
```

## Coordinate remapping

Pyramid levels track the scale factor so that detections at a downsampled level
can be mapped back to the original image frame. `OwnedPyramidLevel` provides
methods to remap `Circle` and `Ellipse` geometries to and from the level
coordinate system.
