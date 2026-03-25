# ADR-004: Feature Flags

**Status:** Accepted
**Date:** 2025-01-15

## Context

The library has optional capabilities that pull in additional dependencies
or enable experimental code. These should not be imposed on users who
need only the core algorithms.

## Decision

All features are opt-in. The default build has zero optional dependencies.

| Feature    | Purpose                                    | Dependency         |
|------------|--------------------------------------------|--------------------|
| `rayon`    | Parallel multi-radius proposal computation | `rayon`            |
| `image-io` | PNG/JPEG image loading                    | `image`            |
| `tracing`  | Structured logging                        | `tracing`          |
| `affine`   | Experimental GFRS extensions              | none (code-gated)  |
| `serde`    | Serialization for configs and results     | `serde`            |

## Consequences

- Minimal compile time and binary size for the default build.
- Users opt in to exactly the capabilities they need.
- CI tests both `--all-features` and `--no-default-features` to ensure
  the feature matrix compiles.
- The `affine` feature gates an entire module, not individual items —
  keeps the feature boundary simple.
