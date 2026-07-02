# radsym WASM demo

Interactive browser demo for the [radsym](https://github.com/VitalyVorobyev/radsym)
radial symmetry detection library. It runs FRST, RSD, and their fused variants
entirely in the browser via WebAssembly — no image ever leaves the device.

`demo/` is the **single canonical source** for the demo. The GitHub Pages copy
is staged from here into `book/src/demo/` by [`book/build.sh`](../book/build.sh);
both use identical relative paths (`./pkg/…`, `./samples/…`).

## Files

| File | Role |
|------|------|
| `index.html` | Layout and controls |
| `styles.css` | Visual system, matched to the vitavision landing/perf pages |
| `app.js` | WASM driver + the pixel-accurate overlay engine |
| `samples/`, `samples.json` | Predefined images + captions, suggested params, ground truth |
| `vv-logo.svg` | Brand mark |
| `pkg/` | Built WASM package (generated, git-ignored) |

## Build & run locally

```bash
# 1. Build the WASM package (from the repository root)
wasm-pack build crates/radsym-wasm --target web --release

# 2. Place the built package next to the demo
mkdir -p demo/pkg
cp crates/radsym-wasm/pkg/radsym_wasm.js crates/radsym-wasm/pkg/radsym_wasm_bg.wasm demo/pkg/

# 3. (Optional) regenerate the sample images + manifest
cargo run -p radsym --example gen_demo_samples --features image-io

# 4. Serve the repository root and open the demo
python3 -m http.server 8080
open http://localhost:8080/demo/
```

## What it shows

- **Sample gallery** — six predefined scenes (calibration ring grid, concentric
  rings, a dense field, perspective ellipses, a low-contrast/noisy scene, and a
  mixed-radii scene) plus your own uploads. Each sample applies suggested
  parameters and, for the synthetic ones, carries ground-truth centers.
- **Comprehensive controls** with progressive disclosure — primary knobs
  (algorithm, radii, polarity, min score) are always visible; voting, NMS,
  scoring, refinement, and overlay options live in collapsible groups.
- **Pipeline stages** — gradient magnitude, response heatmap, and pre-NMS
  proposals for the selected algorithm.
- **Detected circles** on a result-forward, zoomable canvas with a live
  inspector (subpixel center, radius, ringness, coverage, status).

## Overlay correctness

The library reports centers in a **pixel-center** convention (integer
coordinate = pixel center; see
[`docs/decisions/001-coordinate-convention.md`](../docs/decisions/001-coordinate-convention.md)).
On a canvas, `drawImage(img, 0, 0)` puts that pixel's center at
`(x + 0.5, y + 0.5)`, so every positional overlay maps an image coordinate `c`
to display coordinate `(c + 0.5) · displayScale`. Overlays are drawn on a
separate, device-pixel-ratio-aware canvas so vectors stay crisp while the image
stays pixel-sharp. Enable **Ground truth** on a synthetic sample and zoom in to
confirm the overlays sit dead-center on the features.

## Color modes

Detections can be colored by **support score**, **ringness**, **angular
coverage** (a blue→teal→green ramp), or **refinement status** (green =
converged, amber = max iterations, rose = degenerate, muted = out of bounds).

## Notes

- The **algorithm selector drives the whole pipeline**, including the final
  detection panel (via `detect_circles_detailed_with`). FRST uses per-proposal
  scale selection; the other proposers score and refine at the radius hint.
- Single-page static app — no JS build step or bundler.
