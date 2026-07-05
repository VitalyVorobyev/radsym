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
   hardcoded in `perf_export.rs` and matching `benches/frst_bench.rs`; `benches/rsd_bench.rs`
   uses a different radius set for its own criterion runs). Reported as p50 ms
   and megapixels/second.

3. **Refinement cost** — per-call p50 (microseconds) of `radial_center`,
   `circle`, `ellipse`, and homography-aware ellipse refinement.

4. **End-to-end latency** — whole-pipeline `detect_circles` p50 per image.

## Methodology

- **Single-threaded.** The harness measures the default-build (no-`rayon`)
  sequential path. radsym's `rayon` feature is `dep:rayon`-gated, so it is *not*
  pulled in by the `criterion` dev-dependency — the default example build always
  runs the sequential `.iter()` voting path. The `RAYON_NUM_THREADS=1` line only
  matters if you explicitly add `--features rayon` (then it pins voting to one
  thread for deterministic numbers). With `rayon` enabled and N cores,
  multi-radius voting is roughly min(N, radii-count)× faster with bit-identical
  output (see [Voting performance](./voting-performance.md)).
- **Median of repeated runs.** Each measurement warms up (5 iterations) then
  takes the median (p50) of the timed iterations. Inputs and outputs are passed
  through `std::hint::black_box` so the optimizer cannot hoist or elide the work.
- **Adaptive repeat count.** Sub-megapixel images use 50 reps; larger images use
  fewer (20 for ~1 MP, 8 for the multi-megapixel surface-hole image) to keep the
  whole run to a few minutes. Medians stay stable.
- **Release build.** Timings are only meaningful with optimizations on.

> **Reading the absolute numbers.** On a laptop, a single plain-binary run is
> subject to OS scheduling and frequency/thermal state (on Apple Silicon, the
> P-core/E-core split alone moves an unfused-FRST call by ~4× for *identical*
> machine code). Run-to-run variance of ±10–30% on absolute timings is normal.
> The **cross-algorithm and cross-size ratios** on the page are the stable
> signal — they are measured in the same run, so drift cancels. Treat the
> absolute milliseconds as order-of-magnitude, not precise.

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

The interactive demo's canonical source is `demo/` at the repo root
(HTML/CSS/JS + sample images). The built WASM package (`pkg/*.wasm`) is a
gitignored build artifact both there and in the generated `book/src/demo/`
copy the build script stages it into. To rebuild against the current source:

```sh
./book/build.sh    # requires wasm-pack + mdbook
```

This rebuilds the WASM package, stages it into `book/src/demo/`, and rebuilds
the book. The `Docs` workflow then bundles it on the next push to `main`.
