# ADR-008: Stable vs Experimental Boundary

**Status:** Accepted
**Date:** 2025-01-15

## Context

The library includes both well-established algorithms (FRST, RSD,
Parthasarathy radial center) and experimental extensions (GFRS
affine-aware voting). Experimental code may change or be removed without
a semver major bump if included unconditionally.

## Decision

The **`affine` module is experimental** and gated behind the `affine`
feature flag. All other modules (`core`, `propose`, `support`, `refine`,
`diagnostics`) are considered stable and follow semver.

Criteria for promotion from experimental to stable:

1. Algorithm correctness validated on real-world data (not just synthetics)
2. API reviewed and considered unlikely to change
3. Performance acceptable without special tuning

## Consequences

- Users of the default build get only stable, well-tested algorithms.
- The `affine` feature flag signals "use at your own risk" clearly.
- Breaking changes to `affine` can land in minor releases without
  violating semver (the feature is documented as experimental).
- Future experimental modules (e.g., iterative voting, GST) follow the
  same pattern: feature-gated until promoted.
