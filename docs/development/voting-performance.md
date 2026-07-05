# Voting performance (FRST) — profiling & levers

FRST voting is the dominant cost of `detect_circles`. This document records
where the time goes, the in-place optimization applied to the unfused path, and
the two levers available for production throughput (one free, one with a quality
trade-off). All numbers are single-thread on an Apple M4 Pro; treat absolute
milliseconds as order-of-magnitude (see the variance note in
[performance.md](./performance.md)), and the **ratios** as the reliable signal.

## Where the time goes

Across every image on the performance page, voting (`frst_response_scaled`,
including the per-pixel winning-radius map `run_detection` actually pays for)
is now **67–76 %** of end-to-end `detect_circles` (e.g. synthetic-1024: 25.7 ms
of 38.4 ms; surface-hole 2048×1536: 137.3 ms of 180.8 ms) — down from 88–92 %
before the 0.4.1 cache-blocked blur rewrite (see
[CHANGELOG](../../CHANGELOG.md)), which cut the dominant per-radius Gaussian
blur cost by roughly 7×. NMS extraction is now a more significant share of the
remainder (up to ~31 % at 1024²) than before the blur fix; gradient, scoring,
and refinement stay sub-millisecond to low-millisecond.

The unfused path (`frst_response` → `frst_response_scaled`) computes, **per
radius**: a scatter-voting pass into two accumulators (`O_n`, `M_n`), a
normalization-max reduction, a combine pass into `S_n`, and a Gaussian blur,
then sums the per-radius `S_n`. For five radii that is a sequence of large-array
passes whose cost is **memory-bandwidth-bound**, not compute-bound — which is
why the fused variant (one accumulator, one blur) is several× faster despite
casting the same number of votes.

## The optimization: reuse per-radius scratch

The unfused path previously allocated (and first-touch page-faulted) ~four
`w·h` buffers *per radius* — `O_n`, `M_n`, `S_n`, and the blur temporary — and
collected all five `S_n` images before summing. It now reuses a single
`FrstScratch` set across all radii on the serial path, folds each radius into
the running accumulators immediately (no per-radius images retained), reuses the
blur temporary via `gaussian_blur_inplace_buf`, and fuses the two normalization
reductions into one pass.

The output is **bit-for-bit identical** — proven by
`optimized_frst_is_bit_identical_to_reference`, which checks the optimized path
against a verbatim copy of the pre-optimization algorithm across polarities,
alpha branches, all three blur regimes, and a non-square image, on both the
serial and rayon paths.

Measured win (criterion, single-thread, steady-state):

| size | change |
|------|--------|
| frst/256  | ~flat (within the ±2–4 % noise floor) |
| frst/512  | **−15 %** |
| frst/1024 | **−8 %** |

The 512 case gains most: that is the working-set size where the response buffers
spill L2, so avoiding per-radius re-faulting matters most. The win is modest
because the dominant cost is the *number* of bandwidth-bound passes (blur +
combine + reduction + sum), which buffer reuse does not change — only allocation
and fault overhead.

## Lever 1 — `rayon` (free, bit-identical, recommended)

`frst_response_scaled` already parallelizes per-radius voting across the global
rayon pool when the `rayon` feature is enabled. Because each radius is
independent and the sum is reduced in fixed radii order, the output is
**bit-for-bit identical** to the serial path.

Measured (unfused, real images, `rayon` on a 12-core M4 Pro):

| image | serial | rayon | speedup |
|-------|--------|-------|---------|
| ringgrid 720×540 | 13.96 ms | 4.26 ms | **3.3×** |
| surf1 2048×1536  | 501 ms | 123 ms | **4.1×** |
| surf4 2048×1536  | 504 ms | 128 ms | **3.9×** |

> These serial/rayon absolute timings predate the 0.4.1 cache-blocked blur
> rewrite and have not been re-measured against it — the per-radius blur that
> both paths now share got roughly 7× faster, so both columns (and likely the
> speedup ratio) have shifted. Treat this table as showing that the lever
> exists and roughly how it scales, not as current absolute numbers.

This is the recommended production lever: ~min(cores, radii-count)× throughput
with **no change in detection results**. Enable it with:

```toml
radsym = { version = "0.4", features = ["rayon"] }
```

## Lever 2 — fused voting (faster, but diverges on real images)

`frst_response_fused` collapses all radii into one voting pass and one blur. It
is now **~2.4–2.6× faster** than serial unfused (down from 4–6× before the
0.4.1 blur rewrite — unfused paid for five per-radius blurs where fused pays
for one, so it benefited proportionally more from the blur speedup, closing
most of the gap; see the throughput table on the
[performance page](https://VitalyVorobyev.github.io/radsym/performance/)).
It still **drops the orientation (`|O_n|^α`)
term** that gives FRST its selectivity. On clean synthetic disks the peaks agree
within a few pixels; on real cluttered images they do **not**:

| image | speedup (pre-0.4.1) | fused proposals landing within 2 px of unfused |
|-------|---------|------------------------------------------------|
| ringgrid | 4.3× | 3 / 60 (max displacement 97 px) |
| surf1 | 6.4× | 0 / 10 (max 125 px) |
| surf2 | 5.6× | counts even differ (6 vs 10), max 334 px |

> The divergence columns (which proposals move, by how much) are unaffected by
> the blur speedup — cache-blocking is bit-identical, so it changes timing
> only, not results. The per-image speedup column predates it and has not been
> re-measured; expect it to have come down like the synthetic throughput
> numbers above.

So fused is **not a safe default and not a drop-in** for unfused quality. It is
appropriate only as an opt-in for callers who have validated it for their own
imagery (e.g. high-contrast targets on a clean background). For
quality-preserving speed, prefer Lever 1.

### Pipeline exposure

`detect_circles` currently hardcodes the unfused backend (`frst_response_scaled`),
because it also yields the per-pixel winning-radius map that drives per-proposal
`scale_hint`. Fused produces a single accumulator with no per-radius separation,
so exposing it through the pipeline would mean proposals fall back to the global
`radius_hint`. Whether to add a voting-backend selector to `DetectCirclesConfig`
is an open product decision (it trades the scale-hint feature for throughput on
validated imagery) — deliberately left for a follow-up rather than changed
silently.

## Done since (0.4.1)

- **Cache-blocked box blur** — the vertical pass's stride-`w` traversal was
  cache-hostile; it's now tiled into 16-column strips, bit-identical output,
  ~7× faster alone (see [CHANGELOG](../../CHANGELOG.md)). This is what moved
  voting's end-to-end share from 88–92 % down to 67–76 % above and closed most
  of the fused/unfused gap in Lever 2.

## Not pursued (and why)

- **Fewer blur passes / data-independent normalization** — would cut bandwidth
  but changes the response numerically (a quality decision, not a free win).
- **Making fused or rayon the default** — both change behavior (results or
  build/runtime profile) and need an explicit decision.
