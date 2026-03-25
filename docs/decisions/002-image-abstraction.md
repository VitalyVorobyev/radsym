# ADR-002: Image Abstraction

**Status:** Accepted
**Date:** 2025-01-15

## Context

Image data needs a common abstraction for algorithm inputs and outputs.
Options considered:

1. **Trait-based** (`trait Image<T>`) — maximum flexibility, but incurs
   monomorphization cost or trait-object overhead.
2. **Concrete struct** (`ImageView<'a, T>`) — simple, zero-overhead
   borrowing with explicit stride.
3. **External crate** (`image::ImageBuffer`) — pulls in a heavy dependency.

## Decision

Use concrete structs: `ImageView<'a, T>` for borrowed image data and
`OwnedImage<T>` for algorithm outputs. Both carry width, height, and
stride explicitly.

This follows the proven pattern from the
[corrmatch-rs](https://github.com/VitalyVorobyev/corrmatch-rs) sibling
project.

## Consequences

- No trait-object overhead or monomorphization explosion.
- Stride support enables zero-copy ROI views.
- Bilinear `sample()` method provides subpixel access for f32 and u8.
- Adding new pixel types requires the `ImagePixel` trait but no changes
  to the view struct itself.
- Cannot accept arbitrary image types without conversion — callers must
  provide a slice with known layout.
