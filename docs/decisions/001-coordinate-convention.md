# ADR-001: Coordinate Convention

**Status:** Accepted
**Date:** 2025-01-15

## Context

Vision libraries use conflicting coordinate conventions. Row-major array
indexing suggests `(row, col)`, but image processing and graphics
traditionally use `(x, y)` where x is horizontal and y is vertical.
nalgebra's `Point2<f32>` uses `(x, y)` ordering.

## Decision

Use **(x = col, y = row)** everywhere. `PixelCoord` is a type alias for
`nalgebra::Point2<f32>`. x increases rightward, y increases downward.

## Consequences

- Matches nalgebra's `Point2` convention — no mental translation needed.
- Array indexing requires `data[y * stride + x]`, which is explicit but
  correct.
- All geometric computations (gradients, voting offsets, annulus sampling)
  use the same convention without conversion.
- Incompatible with libraries that use `(row, col)` — conversions needed
  at integration boundaries.
