# ADR-003: Scalar Precision

**Status:** Accepted
**Date:** 2025-01-15

## Context

Pixel-level computation can use f32 or f64. f64 provides more precision
but doubles memory bandwidth and prevents SIMD-friendly 4-wide f32
operations. Most image processing operates well within f32 range
(pixel values 0-255, image dimensions < 65536).

## Decision

Use **f32** for all pixel-level computation. The `Scalar` type alias is
`f32`. No `f64` appears in the public API.

Internal computations that require higher precision (e.g., solving small
linear systems in the Parthasarathy radial center) may use f64 locally
and cast back to f32 for the result.

## Consequences

- Half the memory bandwidth of f64 for gradient fields and response maps.
- Matches common GPU precision, easing future GPU backend work.
- Subpixel accuracy is limited to ~1e-7 pixels, which is far beyond
  what image noise allows.
- Interop with f64-based libraries requires explicit conversion.
