# Image Pyramids

## Overview

The pyramid module provides multi-scale image access via the `box-image-pyramid`
crate. Downsampled images allow detection of large radial features without
requiring proportionally large voting radii.

The pyramid types live under `radsym::core::pyramid` — they are not re-exported
at the crate root, since multi-scale access is a power-user tool rather than
part of the common detect-score-refine contract.

## PyramidWorkspace

`PyramidWorkspace` is a reusable buffer that avoids reallocating intermediate
downsample storage on every frame. Construct it once, then call
`workspace.level(image, level)` to extract a borrowed view at the requested
pyramid level. Level 0 is the original resolution; each subsequent level halves
both dimensions using a box filter.

```rust
use radsym::core::pyramid::PyramidWorkspace;

let mut workspace = PyramidWorkspace::new();
let level2 = workspace.level(image, 2)?;
// level2.image() is an ImageView at 1/4 resolution
```

## One-shot downsampling

For cases where workspace reuse is not needed, `pyramid_level_owned` extracts
a single pyramid level as an owned image:

```rust
use radsym::core::pyramid::pyramid_level_owned;

let downsampled = pyramid_level_owned(&image, 2)?;
```

## Coordinate remapping

Pyramid levels track the scale factor so that detections at a downsampled level
can be mapped back to the original image frame. Both `PyramidLevelView` (the
borrowed view) and `OwnedPyramidLevel` provide methods to remap `Circle` and
`Ellipse` geometries from the level coordinate system back to the original
image frame.
