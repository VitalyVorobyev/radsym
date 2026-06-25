# Performance methodology

The published performance page
(`https://VitalyVorobyev.github.io/radsym/performance/`) is rendered from a
single committed data file, `.github/pages/performance/data.json`, produced by a
reproducible harness. This document explains what is measured and how to
regenerate it.

## What the page shows

1. **Per-image stage breakdown** — the cost of `detect_circles` decomposed into
   its five pipeline stages, for a spread of representative images (synthetic
   disks at 256/512/1024, the `ringgrid` test image, and a full-resolution
   surface-hole image). The decomposition mirrors `run_detection`
   (`crates/radsym/src/pipeline.rs`) exactly:

   | Stage | Function | Frequency |
   |-------|----------|-----------|
   | gradient | `compute_gradient` (Sobel) | once |
   | voting | `frst_response` (unfused FRST) | once |
   | extract | `extract_proposals` (NMS) | once |
   | score | `score_circle_support` | every proposal |
   | refine | `refine_circle` | accepted proposals only |

   The gradient is computed once and reused by voting, scoring, and refinement —
   so those stages stay cheap. As a faithfulness check, the sum of an image's
   stage bars equals its end-to-end `detect_circles` time within a few percent
   (the residual is diagnostics bookkeeping).

2. **Voting-backend throughput** — FRST vs RSD, unfused vs fused, across image
   sizes, on synthetic disks with the gradient precomputed (radii `[5,7,9,11,13]`,
   matching `benches/frst_bench.rs` / `benches/rsd_bench.rs`). Reported as p50 ms
   and megapixels/second.

3. **Refinement cost** — per-call p50 (microseconds) of `radial_center`,
   `circle`, `ellipse`, and homography-aware ellipse refinement.

4. **End-to-end latency** — whole-pipeline `detect_circles` p50 per image.

## Methodology

- **Single-threaded.** The harness forces `RAYON_NUM_THREADS=1`. radsym's FRST
  parallelises multi-radius voting over the global rayon pool when the `rayon`
  feature is active (it leaks into the example build via the `criterion`
  dev-dependency); pinning to one thread gives deterministic numbers that match
  the default-build (no-`rayon`) sequential path. With `rayon` enabled and N
  cores, multi-radius voting is roughly N×-but-≤(radii count) faster.
- **Median of repeated runs.** Each measurement warms up (5 iterations) then
  takes the median (p50) of the timed iterations. Inputs and outputs are passed
  through `std::hint::black_box` so the optimizer cannot hoist or elide the work.
- **Adaptive repeat count.** Sub-megapixel images use 50 reps; larger images use
  fewer (20 for ~1 MP, 8 for the multi-megapixel surface-hole image) to keep the
  whole run to a few minutes. Medians stay stable.
- **Release build.** Timings are only meaningful with optimizations on.

## Regenerate

The numbers are generated on a known machine and committed (CI runners are
shared and noisy, so absolute timings there are not comparable over time). The
`meta` block in `data.json` records the CPU, rustc version, and git SHA the
numbers were measured against.

```sh
cargo run --release -p radsym --example perf_export --features image-io,serde \
  -- .github/pages/performance/data.json
```

Pass `-` as the output path to print the JSON to stdout for inspection. Do **not**
add `--features rayon`: the harness pins one thread regardless, but keeping the
default feature set matches what most users build.

The harness lives in `crates/radsym/examples/perf_export.rs`. It is a faithful
re-timing of the public composable stage functions, not a separate
implementation — the same calls a real `detect_circles` makes.

## Refreshing the WASM demo

The interactive demo at `/demo/` is bundled from `book/src/demo/`, whose
`pkg/*.wasm` is committed. To rebuild it against the current source:

```sh
./book/build.sh    # requires wasm-pack + mdbook
```

This rebuilds the WASM package, copies it into `book/src/demo/pkg/`, and rebuilds
the book. The `Docs` workflow then bundles it on the next push to `main`.
