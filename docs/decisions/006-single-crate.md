# ADR-006: Single Crate vs Multi-Crate

**Status:** Accepted
**Date:** 2025-01-15

## Context

The initial concept document proposed 7 separate crates (radsym-core,
radsym-frst, radsym-support, etc.). This forces premature API boundary
decisions and increases maintenance overhead for a greenfield project
where module responsibilities are still solidifying.

## Decision

Use a **single library crate** (`radsym`) with internal modules:

```
core/         -- types, image, geometry, gradient, NMS
propose/      -- center voting (FRST, RSD), proposal extraction
support/      -- annulus sampling, coverage, scoring
refine/       -- radial center, circle/ellipse refinement
affine/       -- (feature-gated) experimental extensions
diagnostics/  -- visualization
```

The dependency graph is enforced by convention (and CI):

```
core  <--  propose, support, refine
core + propose  <--  affine
all modules  <--  diagnostics
```

## Consequences

- No premature crate-boundary decisions. Internal module boundaries can
  shift without breaking downstream.
- Single `Cargo.toml` simplifies versioning, feature flags, and CI.
- Users get one dependency (`radsym`) instead of managing multiple crates.
- Extraction into separate crates remains possible if a clear need
  emerges (e.g., independent versioning for Python bindings).
- Downside: users cannot depend on just `core` without algorithm code.
  In practice this is acceptable — the algorithms are small.
