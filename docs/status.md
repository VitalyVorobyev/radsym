# radsym — Workspace Status Report

**Date:** 2026-07-02 · **Version:** 0.3.0 · **Workspace:** `radsym`, `radsym-py`, `radsym-wasm`

> **Dated snapshot.** This report predates the 0.4.0 API revision and the
> 0.4.1 performance release (cache-blocked blur, `thin_gradient`, `unsafe-opt`
> voting) — see `CHANGELOG.md` for what has changed since. Not refreshed here;
> treat the gaps/recommendations below as historical context, not current
> status.

## 1. Executive summary

`radsym` is unusually well-documented and mature for its size: the core
radial-symmetry algorithm work (FRST, RSD, radial-center, circle/ellipse
refinement, homography-aware variants) is complete, cited to source papers, and
stable. Recent effort has shifted away from new algorithms toward
production-hardening, multi-language bindings (Python via PyO3, WASM via
wasm-bindgen), and a public web presence (GitHub Pages landing + performance
pages, a 20-chapter mdBook, and published crates). Overall health is **strong**,
with a small set of low-risk gaps — several of which are being closed this
session.

## 2. Strengths (with evidence)

| Area | Evidence |
|------|----------|
| Architecture decisions | 8 Accepted ADRs, `docs/decisions/001-coordinate-convention.md` … `008-stable-experimental-boundary.md`, all in Context/Decision format |
| Design documentation | `docs/design.md` — goals, module layout, layered dependency graph, per-algorithm write-ups, support-scoring rationale |
| Backlog hygiene | `docs/backlog.md` — 9 Epics with checkboxes, nearly all complete |
| Long-form docs | Complete 20-chapter mdBook under `book/` (~1,900 lines) with rustdoc API integration and an embedded WASM demo |
| Changelog | `CHANGELOG.md` follows Keep a Changelog + SemVer, with migration notes and compare links |
| CI / release | 6 workflows in `.github/workflows/`: `ci.yml`, `docs.yml`, `publish-crates.yml`, `release-pypi.yml`, `release-npm.yml`, `release.yml` |
| Test coverage | ~143 unit tests across 27 `#[cfg(test)]` modules; 13 integration test fns; 15 doctests; ~27 Python tests; wasm `web.rs`; 6 criterion benches |
| Safety / hygiene | Zero `unsafe` outside the opt-in `unsafe-opt` feature (added 0.4.1, gated to the voting scatter loops, indices proven in-bounds); zero `TODO`/`FIXME`/`XXX` in `src/` |

**CI detail (`ci.yml`):** `cargo fmt` + `clippy -D warnings`; test matrix over
all-features and `--no-default-features`; MSRV 1.88; doc build; wasm check.

**Release detail:** `publish-crates.yml` publishes to crates.io via OIDC trusted
publishing; `release-pypi.yml` builds maturin wheels for Python 3.10–3.14;
`release-npm.yml` publishes `@vitavision/radsym`; `release.yml` cuts the GitHub
release.

**Integration tests:** `pipeline_e2e`, `ringgrid_regression`,
`surf_hole_regression`, `source_pixel`.

## 3. Gaps & discrepancies (prioritized)

| Pri | Status | Gap |
|-----|--------|-----|
| **P1** | FIXED | No `LICENSE-MIT` / `LICENSE-APACHE` files existed despite README badges and `license = "MIT OR Apache-2.0"` — added this session |
| **P1** | FIXED | Version not bumped: the `[Unreleased]` additive batch (multi-bit-depth `SourcePixel`, ROI, per-proposal scale, config validation, PyO3 0.29, FRST scratch-reuse perf) warranted **0.3.0** — bumped this session |
| **P1** | RECOMMENDED | Determinism & translation-invariance are advertised (`README.md`, `CLAUDE.md`, `docs/design.md`) but have **no property tests**. Top testing gap — add proptest-based tests |
| **P2** | RECOMMENDED | The hardest numerical code (ellipse Gauss-Newton refinement ~830 lines; homography refinement) is covered only end-to-end — no isolated numerical-accuracy or degenerate/edge-case tests |
| **P2** | FIXED | Stale docs: `AGENTS.md` described a single crate, the wrong `refine/` dependency rule, and only 4 of 8 algorithm families; `docs/backlog.md` Epic 8.4 (badges/CHANGELOG/licenses) and rayon-voting items were marked open though done — refreshed this session |
| **P2** | FIXED | Stale doc promise: `RefinementResult` rustdoc promised an "optional uncertainty estimate" that does not exist — removed this session |
| **P3** | RECOMMENDED | No single `RELEASING.md`; the tag-driven release flow is documented only implicitly across `CLAUDE.md`, `CONTRIBUTING.md`, and the four tag-triggered workflows |
| **P3** | RECOMMENDED | Config schema is triplicated across core Rust defaults, PyO3 `#[new]` defaults, and wasm `set_*` — drift risk (a live example existed: wasm `set_max_detections` doc said 50 vs actual 1000). Tracked in `API_REVISION.md` |
| **P3** | RECOMMENDED | mdBook uses the stock theme; does not match the dark vitavision brand of the landing/performance pages (cosmetic) |

## 4. Open backlog highlights

From `docs/backlog.md` and `docs/development/production-assessment.md`, still open:

- **Proposal generation:** multi-scale proposal fusion (coarse-to-fine)
- **Benchmarks:** dense-pattern benchmark (50+ targets); affine-vs-isotropic benchmark
- **Export:** JSON export of proposals/scores; radial-profile export
- **Deferred algorithms:** GST (Reisfeld 1995); iterative voting (Parvin 2007)
- **Production items:** D2b pyramid auto-radius; D3 one-call `detect_ellipses`; P1 reusable-buffer / streaming detector; accuracy / covariance estimate; Python diagnostics channel; `.pyi` + `py.typed` stubs

## 5. Recommendations (next steps, prioritized)

1. Add determinism / translation-invariance **property tests** (closes the top P1 testing gap).
2. Add **isolated numerical tests** for ellipse and homography refinement (degenerate + accuracy cases).
3. Execute the **API revision plan** (`API_REVISION.md`) before 1.0 — resolve the triplicated config schema.
4. Ship **`.pyi` + `py.typed`** stubs and a **Python diagnostics channel**.
5. Consider a **`RELEASING.md`** consolidating the tag-driven release flow.
