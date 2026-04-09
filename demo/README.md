# radsym WASM Demo

Interactive browser demo for the [radsym](https://github.com/VitalyVorobyev/radsym)
radial symmetry detection library. Runs FRST, RSD, and their fused variants
entirely in the browser via WebAssembly. All configuration parameters are
exposed in the UI.

## Prerequisites

- Rust toolchain with the `wasm32-unknown-unknown` target
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- A local HTTP server (e.g., `python3 -m http.server`)

## Build & Run

```bash
# 1. Build the WASM package (from repository root)
wasm-pack build crates/radsym-wasm --target web --release

# 2. Serve the repository root
python3 -m http.server 8080

# 3. Open in your browser
open http://localhost:8080/demo/
```

The demo loads `testdata/ringgrid.png` by default. Use the file picker to try
your own images.

## Output Panels

| Panel | Description |
|-------|-------------|
| **Original** | Source image (loaded or uploaded) |
| **Response Heatmap** | Response from the selected algorithm, colorized |
| **Seed Proposals** | NMS-extracted proposal locations (cross markers) |
| **Detected Circles** | Full pipeline output with overlay annotations |

## Proposal Algorithms

| Algorithm | WASM Method | Description |
|-----------|------------|-------------|
| **FRST** | `frst_response` | Multi-radius with orientation accumulator (Loy & Zelinsky 2002) |
| **FRST (fused)** | `multiradius_response` | Single-pass fused variant, faster |
| **RSD** | `rsd_response` | Magnitude-only voting, ~2x faster (Barnes et al. 2008) |
| **RSD (fused)** | `rsd_response_fused` | Single-pass fused RSD |

## Configuration Reference

### Algorithm Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| Proposal method | FRST | Algorithm for heatmap and proposals |
| Radii | 9,11,13,15,17 | Discrete radii (px) to test |
| Alpha | 2.0 | Radial strictness exponent (FRST only) |
| Gradient threshold | 0 | Minimum gradient magnitude for voting |
| Smoothing factor | 0.5 | Gaussian smoothing sigma = factor x radius |

### Detection Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| Polarity | both | Detect bright, dark, or both structures |
| Gradient operator | sobel | Sobel or Scharr gradient kernel |
| Radius hint | 13 | Initial radius hypothesis for refinement |
| Min score | 0 | Minimum support score threshold |

### NMS Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| NMS radius | 5 | Suppression radius in pixels |
| NMS threshold | 0 | Minimum response for a peak |
| Max detections | 200 | Maximum number of detections |

### Support Scoring (collapsed)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Angular samples | 64 | Samples around the annulus circumference |
| Radial samples | 9 | Samples across the annulus width |
| Annulus margin | 0.3 | Annulus width as a fraction of radius |
| Min samples | 8 | Minimum gradient samples for a valid score |
| Weight: ringness | 0.6 | Weight of gradient alignment in the total score |
| Weight: coverage | 0.4 | Weight of angular coverage in the total score |

### Refinement (collapsed)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max iterations | 10 | Iteration limit for circle refinement |
| Convergence tolerance | 0.1 | Convergence threshold in pixels |
| Max center drift | 0.5 | Maximum drift from initial center (x radius) |

## Overlay Features

### Color Modes

| Mode | Description |
|------|-------------|
| **Total score** | Green (low) to red (high) combined score |
| **Ringness** | Blue (low) to orange (high) gradient alignment |
| **Angular coverage** | Purple (low) to cyan (high) annulus coverage |
| **Refinement status** | Categorical: green=Converged, yellow=MaxIter, red=Degenerate, gray=OutOfBounds |

### Toggleable Layers

Score labels, annulus rings, status icons (C/M/D/O), legend.

### Click to Inspect

Click any detected circle to see its full detail: center coordinates, radius,
score breakdown, and refinement status.

## Browser Compatibility

Tested in Safari, Chrome, and Firefox. Requires WebAssembly and ES modules.

## Architecture

- Single-page static app: `index.html` + `app.js`
- No JS build step or bundler
- WASM module loaded from `../crates/radsym-wasm/pkg/`
- All processing happens client-side
