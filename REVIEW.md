# radsym workspace review

_Reviewed 2026-06-25 against `main` @ `961a32a` (v0.2.0). Scope: practical value,
design (SOLID/DRY/best practices), and completeness/correctness of tests,
comments, and documentation. Every claim below was checked against the source or
a run — `cargo test --workspace --all-features` passes (exit 0); the performance
numbers come from the new `perf_export` harness._

## Verdict

radsym is a **well-architected, genuinely useful library in good shape**. The
0.2.0 API revision (builder, config split, `#[non_exhaustive]`, diagnostics
channel) is mature; the module layering is clean and the documented dependency
rules are honoured. The main gaps are in **test depth** (no property tests, thin
isolated coverage of the hardest numerical code) and a **performance default**
worth revisiting. Nothing here blocks use; the items below are prioritised for
the two follow-on phases.

Two issues flagged by an initial automated skim did **not** hold up on
inspection and are noted so they are not re-raised: config struct **fields are
fully documented** (FrstConfig, RsdConfig, ScoringConfig, CircleRefineConfig,
NmsConfig all carry field rustdoc), and `radsym-wasm` is **not** untested — it is
excluded from the host clippy/test run but covered by a dedicated `wasm-build`
job (`cargo check --target wasm32-unknown-unknown`).

## Practical value — strong

Pure-Rust radial-symmetry / circle detection with **no OpenCV or heavy vision
dependency** (only `nalgebra`) is a real niche. The pipeline (FRST/RSD proposals
→ support scoring → Parthasarathy/circle/ellipse refinement, plus homography-
and affine-aware variants) maps directly onto industrial targets (ring/dot
boards, surface holes) and scientific imaging (Parthasarathy radial center is
from Nature Methods particle tracking). Python (PyO3) and WASM bindings plus an
in-browser demo broaden reach. CPU-first and deterministic suits reproducible
metrology.

## Design — SOLID / DRY: good

- **Layering is clean and enforced.** `core/` carries no dependency on the
  algorithm modules; `propose/`/`support/` depend only on `core/`; `refine/`
  adds `support/`; `diagnostics/` sits on top. This matches the stated
  dependency rules in `CLAUDE.md`.
- **API maturity (0.2.0).** `DetectCirclesConfig` builder + nested
  `DetectCirclesAdvanced` cleanly separates "intent" from "power-user knobs";
  `#[non_exhaustive]` future-proofs configs and enums; `detect_circles_with_
  diagnostics` keeps the inspection channel out of the result type. Good ISP/OCP.
- **DRY where it counts.** `radial_center_refine_from_gradient` is reused as a
  sub-step of both circle and ellipse refinement; homography mapping lives once
  in `core/homography.rs` and is reused by proposal + refinement; annulus
  sampling is shared by scoring and refinement.
- **Minor smells (low priority):**
  - `DetectCirclesAdvanced.frst.{radii,polarity}` are silently overridden by the
    top-level fields (`pipeline.rs:269-271`). This is now documented on the
    fields, but the foot-gun remains for anyone who sets them directly.
  - `Detection<T>` is generic but only ever instantiated as `Detection<Circle>`
    — mild YAGNI, harmless.
  - `FrstConfig`/`RsdConfig` overlap on radii/threshold/polarity/smoothing; the
    duplication is justified (RSD drops `alpha`) but worth keeping an eye on.

## Tests — the weakest area

Counts: ~141 unit tests across 27 modules, a handful of integration tests
(`pipeline_e2e`, `ringgrid_regression`, `surf_hole_regression`), and 15 doc
tests. All pass. Synthetic generators (disks, ellipses, projective disks, the
surf-hole synthetic case) and regression tests with quantitative thresholds
(center error, axis error, IoU) are a real strength.

Gaps, in priority order:

1. **No property tests at all**, despite `CLAUDE.md` listing "deterministic
   output ordering" and "translation sanity" as conventions. The two properties
   the library most advertises (determinism, translation invariance of
   detection) are not actually asserted anywhere. → Phase 2.
2. **The hardest numerical code is only covered end-to-end.** Ellipse refinement
   (Gauss-Newton, `refine/ellipse.rs`, ~830 lines) and homography ellipse
   refinement have no isolated numeric-accuracy tests — a risk for a numerical
   library where small regressions hide behind a passing E2E threshold. → Phase 2.
3. **No degenerate / edge-case tests:** empty or uniform gradient, out-of-bounds
   seeds, extreme aspect ratios, single-sample annuli. The code has handling for
   these (`is_degenerate`, `OutOfBounds`), but the handling is untested. → Phase 2.

## Documentation & comments — strong

README (with runnable examples), an excellent Keep-a-Changelog `CHANGELOG.md`, a
comprehensive mdBook (every algorithm + internals + configuration guide), crate-,
module- and item-level rustdoc, **documented config fields**, and literature
citations in the algorithm modules. `cargo doc` runs with `-D warnings`.

Minor nits:
- `detect_circles` rustdoc (and `lib.rs`) describe step 1 as "Sobel gradient
  computation", but the operator is configurable (`gradient_operator`, Scharr
  available). Clarified this round.
- No `CONTRIBUTING.md`. Added this round.
- No Python `.pyi` stub (the WASM `.d.ts` is generated by wasm-pack). → Phase 2.

## Bindings & CI

`radsym-py` (PyO3 + maturin, wheels for Py 3.10–3.14) and `radsym-wasm`
(wasm-bindgen, npm `@vitavision/radsym`, committed demo) are both mature. CI
covers fmt, clippy, a test matrix (all-features + no-default-features), an MSRV
1.88 check, doc with `-D warnings`, and a wasm-target check — solid. There is no
benchmark-regression job (intentional this round; the performance page is
generated locally and committed).

## Performance (measured, single-thread, Apple M4 Pro)

> Addressed in 0.4.1 — see the backlog row below. Voting's share of
> end-to-end cost dropped to ~67-76% after cache-blocking the blur; the
> absolute numbers in this section predate that fix.

The new `perf_export` harness produced a concrete, actionable finding:

- **Voting dominates** end-to-end cost (>90% of every per-image stage breakdown).
- **The pipeline uses _unfused_ FRST, which is ~3.5× slower than fused FRST** for
  the same radii: 256 px 4.25 ms vs 1.23 ms; 1024 px 147 ms vs 28 ms. Evaluating
  fused FRST as the `detect_circles` default (subject to a quality check) is the
  highest-leverage optimization. → Phase 3.
- FRST and RSD unfused are within ~5% of each other single-threaded; RSD's
  advantage is smaller than the "≈2× faster" lore suggests at these radii.
- Multi-radius voting parallelizes over rayon (`propose/frst.rs:260`); with the
  `rayon` feature, voting is several× faster (the published numbers pin one
  thread for reproducibility).

## Backlog

| Pri | Item | Phase |
|-----|------|-------|
| ✅ | `CONTRIBUTING.md`; `docs/development/performance.md`; fix "Sobel" doc wording | done this round |
| P1 | Property tests: deterministic ordering + translation invariance | 2 |
| P1 | Isolated numeric-accuracy tests for ellipse & homography refinement | 2 |
| P2 | Degenerate / edge-case tests (empty gradient, OOB seed, extreme aspect) | 2 |
| P2 | Verify each algorithm against its source paper (FRST α/normalization, Kåsa bias, GN guards, homography pullback) | 2 |
| P3 | Python `.pyi` stub for IDE discoverability | 2 |
| P1 | Evaluate fused FRST as the pipeline default (~3.5× faster, measured) | 3 |
| ✅ | Profile + optimize voting + per-radius blur hot path — cache-blocked the vertical box-blur pass (the actual bottleneck: ~96% of `rsd_response_fused`), bit-identical, ~7.4× on the blur alone. See `CHANGELOG.md` `[0.4.1]`. SIMD/allocation-audit/rayon-scaling beyond this were not pursued as part of that fix. | done (0.4.1) |
| P3 | Optional benchmark-regression CI job | 3 |
