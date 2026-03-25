# ADR-007: Diagnostics Format

**Status:** Accepted
**Date:** 2025-01-15

## Context

Algorithm development and debugging require visual inspection of
intermediate results: response maps, proposal locations, refined shapes.
The diagnostics module needs a format that is easy to generate, view,
and integrate into notebooks and reports.

## Decision

Use an internal **RGBA buffer** (`DiagnosticImage`) as the universal
diagnostics canvas. Supported outputs:

- **Heatmaps**: f32 response maps rendered via colormaps (Jet, Hot, Magma)
- **Overlays**: circle/ellipse outlines and seed markers drawn on the canvas
- **PNG export**: feature-gated behind `image-io` via the `image` crate

No runtime dependency on image formats unless `image-io` is enabled.
The raw RGBA buffer can be consumed by any downstream renderer.

## Consequences

- Zero additional dependencies in the default build.
- Colormaps are self-contained (lookup tables in source).
- PNG export requires opting in to the `image-io` feature.
- No SVG, PDF, or interactive formats — those belong in downstream tools.
