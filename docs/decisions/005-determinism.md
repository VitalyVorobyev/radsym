# ADR-005: Determinism Policy

**Status:** Accepted
**Date:** 2025-01-15

## Context

Non-deterministic output ordering causes flaky tests and makes algorithm
comparison difficult. Sources of non-determinism include HashMap
iteration order, floating-point reduction order, and parallel execution.

## Decision

**Deterministic output ordering by default.** Same input always produces
the same output, regardless of platform.

Specific measures:

- NMS peaks are sorted by (score descending, y ascending, x ascending).
- Proposal lists preserve NMS ordering.
- No HashMap-based accumulators in algorithm paths.
- Parallel execution (rayon feature) must produce the same results as
  sequential, though ordering within a single score tier may differ.

## Consequences

- Algorithms are reproducible and testable with exact equality checks.
- Sorting adds O(n log n) overhead to proposal extraction, but n is
  typically small (tens to hundreds of proposals).
- Parallel rayon paths use deterministic reduction (sorted merge) rather
  than non-deterministic concurrent collection.
