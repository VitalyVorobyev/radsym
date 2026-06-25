# radsym production assessment (Phase 2)

_Lens: radsym as a **production industrial machine-vision library**, not a research
project. The question is not "does this match the source paper" but "is this a
drop-in, robust, ergonomic component for industrial inspection." Findings are
evidence-backed (file:line or a reproduced run). Audited 2026-06-25 @ v0.2.0._

## Headline

The algorithms are sound, the architecture is clean, and — verified empirically —
**the pipeline never panics on degenerate input** and is **deterministic and
reentrant**. The distance to "industrial-grade" is almost entirely in the
**input/operating model**, not the math: the pipeline is welded to **8-bit
full-frame input with caller-supplied radii**, and the Python binding (the
dominant industrial integration surface) **holds the GIL** and is **u8-only**.
The good news, confirmed by tracing the data flow, is that everything below
`compute_gradient` already runs on an `f32 GradientField`, so the largest gaps
are cheap to close.

The big three opportunities — **u16/f32 ingestion**, **ROI**, **auto-radius** —
plus two robustness/binding fixes (**config validation**, **GIL release**) are
the highest leverage.

---

## 1. Robustness — strong, with one real gap

**Verified strengths (reproduced):**
- **No panics on degenerate input.** A battery through `detect_circles` (uniform,
  all-zero, 3×3, 1×1, radius > image, impossible `min_score`) all return 0
  detections cleanly; stage functions (`sobel_gradient`, `frst_response`) handle
  1×1/2×2/tiny gradients without panicking. NMS early-returns `Vec::new()` when
  the window exceeds the image (`core/nms.rs:190-192`).
- **Small, guarded panic surface.** The production `unwrap`/`expect` count is
  ~10 (the 318 raw hits are overwhelmingly in inline `#[cfg(test)]` modules). The
  remaining ones are invariant-guarded (`nms.rs:198/208/223/230` bounded by the
  loop + the window guard; `pyramid.rs` `expect("validated …")`). The
  NaN-unsafe `partial_cmp().unwrap()` pattern exists **only in test code**
  (frst.rs test mod starts at line 344; all hits are after it); production
  edge-selection already uses the NaN-safe `partial_cmp().unwrap_or(Equal)`
  (`refine/edge_profiles.rs:113`).
- **Deterministic & reentrant.** FRST parallelizes only across radii, each into
  its own accumulator, summed back in fixed radii order (`propose/frst.rs`), so
  the float reduction is rayon-independent. NMS breaks score ties by `(y,x)`
  (`nms.rs:249-265`); no global/static state. Safe to call concurrently per-frame
  and to lock down with golden-output regression tests.

**Problem R1 — inconsistent config validation (silent wrong results).** `Medium`,
cheap fix. `run_detection` validates **only** refinement (`pipeline.rs:266`).
`FrstConfig` self-validates (empty radii → clean `ValueError: radii must be
non-empty`, reproduced), but **`NmsConfig` and `ScoringConfig` are never
validated** on the detection path — their `validate()` methods exist
(`nms.rs:22`, `score.rs:94`) but aren't called. Reproduced: `nms.max_detections =
0` **silently returns 0 detections** instead of erroring; `annulus_margin = 0`
samples a degenerate ring. An integrator gets wrong output, not an actionable
error. **Fix:** call `nms.validate()` + `scoring.validate()` up front, as
refinement already is. (~S)

---

## 2. Image input model — the headline gap

**Problem/Opportunity I1 — u16/f32 ingestion.** `High` impact, `S–M` cost,
**top priority.** Industrial cameras are overwhelmingly 10/12/16-bit. Today the
front door is u8-only: `detect_circles`/`compute_gradient` take `ImageView<u8>`
(`pipeline.rs:207`, `gradient.rs:334`); reproduced — a `uint16` or `float32`
numpy array is rejected (with a cryptic `'ndarray' is not an instance of
'ndarray'`). Yet `ImageView<'a,T>` is generic and the `f32` gradient kernels
already exist (`sobel_gradient_f32`/`scharr_gradient_f32`/`compute_gradient_f32`,
`gradient.rs:136/270/345`), and **every stage below the gradient consumes the
`f32 GradientField`** — so the type lock lives only in the front-end. **Plan:** a
`trait SourcePixel: Copy { fn to_f32(self)->f32 }` (u8/u16/f32), generalize the
gradient load, make `compute_gradient`/`detect_circles` generic over it; the u8
hot path is unchanged. **One decision:** `gradient_threshold` is an absolute
magnitude (`frst.rs:53`) that scales ~256× for u16 — either document the
bit-depth dependence or normalize against `GradientField::max_magnitude()`
(exists, `gradient.rs:61`).

---

## 3. Detection capabilities — missing industrial primitives

**Opportunity D1 — ROI / region masking.** `High` impact, `S` (rect) / `M`
(mask). Inspection almost always has a known search window. `ImageView::roi`
exists and is zero-copy (`image_view.rs:98`) but the pipeline never uses it and
`DetectCirclesConfig` has no ROI field. Today a caller can pass `view.roi(...)`,
but results come back in **ROI-local coordinates with no offset-restoration
helper** — a silent off-by-origin trap. **Plan:** `roi: Option<Rect>` on the
config; internally crop + translate output centers back to full-frame. A boolean
mask (annular gauges, keep-outs) is `M`.

**Opportunity D2 — unknown-radius / multi-scale.** `High` impact, `M`. Detection
requires caller-supplied `radii`, and two structural facts make this sharp: the
winning radius is **discarded** (`extract.rs` hardcodes `scale_hint: None`
despite `Proposal::scale_hint` existing), and every proposal is scored/refined
from a **single global `radius_hint`** (`pipeline.rs:281`). Because scoring gates
on `radius_hint ± annulus_margin` (default ±30%), **you can only detect circles
within ~±30% of the hint** — mixed-size scenes silently lose detections. A full
pyramid (`core/pyramid.rs` + the `box-image-pyramid` dependency) exists **but is
not wired into detection** (you pay for an unused dependency). **Plan:** (a)
propagate the per-peak winning radius into `scale_hint` and seed scoring/refine
per-proposal — `M`, widens the detectable range immediately; (b) wire the pyramid
for coarse-to-fine "r ∈ [rmin,rmax]" — `M–L`.

**Problem D3 — no one-call `detect_ellipses`.** `Medium`, `S–M`. Under
perspective, real circles image as ellipses; `refine_ellipse` exists but there is
no one-call ellipse entry point, so tilted-part users must hand-assemble the
stages. `Detection<T>` is generic but only ever `Detection<Circle>`.

---

## 4. Performance & throughput

**Opportunity P1 — reusable-buffer / streaming detector.** `Medium`, `M`.
`run_detection` allocates fresh every call (gradient pair + per-radius
accumulators + blur + response map). For line-scan / high-FPS inspection this is
steady allocator pressure. **Plan:** a native `CircleDetector { …scratch }` with
`detect_into(&mut self, image, config, &mut out)`, mirroring the existing
`PyramidWorkspace` pattern and the WASM `RadSymProcessor`. (This is also the right
home for the Phase-3 SIMD work and pairs with the "unfused FRST is ~3.5× slower"
finding from the perf page.)

**Opportunity P2 — accuracy characterization + per-detection uncertainty.**
`Medium`, `M`. `RefinementResult` carries `residual`/`iterations` but **no
covariance**, while its own doc promises an "optional uncertainty estimate"
(`result.rs:4`) — a stale claim. Metrology customers need a stated, benchmarked
subpixel accuracy and ideally a 2×2 center covariance (cheap to expose from the
Parthasarathy normal equations). Either honor the doc or drop the claim, and add
accuracy benches.

---

## 5. API ergonomics / foot-guns

- **A1 — silent override of `advanced.frst.{radii,polarity}`** (`pipeline.rs:269-271`).
  Documented, but "set it and we ignore it" is the classic surprise. Split the
  config so the overridden fields aren't present, or error on inconsistency. `S`.
- **A2 — composable-stage path under-exposed.** It's the *only* way to reach
  u16/f32, ROI translation, or ellipses today, yet it's prose-only — it deserves
  a runnable `examples/` program. `S`.
- **A3 — confidence is uncalibrated.** `SupportScore.total` is a usable relative
  threshold but not a calibrated detection probability, so a fixed `min_score`
  doesn't transfer across part types/lighting. Document operating points now;
  ROC calibration later. `S`/`M`.

---

## 6. Bindings (Python is the main industrial surface)

**Problem B1 — GIL is held for the whole computation.** `High`, `S`, **top
binding priority.** Zero `allow_threads` in `radsym-py/src`; `detect_circles` and
every stage run inline under the GIL (`lib.rs:449`). The dominant industrial
pattern — fan frames across a `ThreadPoolExecutor` — gets **no parallelism**, and
long calls freeze asyncio/UI threads. **Fix:** copy the input (already done), then
wrap compute in `py.allow_threads(|| …)`.

**Problem B2 — u8-only Python input** (same root as I1) with **cryptic errors**.
`PyReadonlyArray2<u8>` everywhere; u16/f32/3D all fail with `'ndarray' is not an
instance of 'ndarray'`, and a non-contiguous ROI slice fails with "not contiguous
or is misaligned" and **no hint to call `np.ascontiguousarray`** (reproduced).
**Fix:** dtype-dispatch ingest (u8/u16/f32 → f32) + `ascontiguousarray` fallback +
clear `ValueError`s, in one shared helper.

**Problem B3 — `detect_circles` is untested and awkward from Python.** No Python
test or example calls `detect_circles` (grep), and radii must be set indirectly
via `DetectCirclesConfig(frst=FrstConfig(radii=[…]))` — there's no top-level
`radii`/`for_radii`. The one-call headline API is the least-exercised binding.
**Fix:** add a top-level `radii` convenience + a test + an example. `S`.

**Problem B4 — diagnostics channel missing in Python, but docs claim it.** No
`detect_circles_with_diagnostics` binding, yet `Detection.score` docs and the
README say the breakdown is "available through the diagnostics channel"
(`results.rs:451`, `README.md:36`). WASM already exposes it
(`detect_circles_detailed`). Parity gap + untrue docs. `M`.

**Problem B5 — no `.pyi` / `py.typed`.** `Medium`. The shipped artifact is an
opaque `.so`; integrators get no autocomplete or mypy/pyright checking. The
rustdoc already on every `#[pyfunction]` maps straight into a stub. `M`.

**Problem B6 — WASM npm identity is stale.** `Low`, `S`. Generated
`pkg/package.json` is `radsym-wasm` `0.1.4`, but the declared package is
`@vitavision/radsym` at 0.2.0. wasm-pack derives the name from `[package].name` —
needs a name override or a CI rename step.

**Smaller binding items:** Python `ScoringConfig` can't set annulus sampling
density (WASM can) — `S`; configs are construct-only (no setters); `Circle` has
`__eq__` but no `__hash__` (unhashable). `Low`.

---

## Recommended "develop now" slate (highest leverage, lowest risk)

| # | Item | Why | Cost |
|---|------|-----|------|
| 1 | **R1** validate `nms`/`scoring` in `run_detection` | silent-wrong-result → actionable error | S |
| 2 | **B1** release the GIL in Python | unblocks multi-frame throughput | S |
| 3 | **I1/B2** u16 + f32 ingestion (generic `SourcePixel`) + dtype-dispatch Python ingest with clear errors | the headline industrial gap; cheap given f32 internals | S–M |
| 4 | **D1** first-class ROI (rectangle) + output coordinate restoration | inspection always has a search window | S |
| 5 | **D2a** propagate winning radius → `scale_hint`, per-proposal scoring/refine | widens detectable size range without a pyramid | M |
| 6 | **B6** fix WASM npm name/version | real packaging bug | S |
| 7 | **B3** `detect_circles` Python ergonomics + test + example | exercise the headline API | S |

Larger follow-ons (own change-sets): D2b pyramid auto-radius, P1 reusable-buffer
detector, P2 accuracy/covariance, B4 Python diagnostics, B5 `.pyi`, D3
`detect_ellipses`. Phase 3 (perf profiling/optimization, incl. the unfused→fused
FRST default) comes after this slate.
