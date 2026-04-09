# Interactive Demo

The interactive demo runs the full radsym detection pipeline in your browser
via WebAssembly. You can switch between proposal algorithms (FRST, RSD, and
their fused variants), adjust all configuration parameters, and inspect
individual detections.

<iframe
  id="radsym-demo"
  src="demo/index.html"
  style="width: 100%; height: 900px; border: 1px solid #ddd; border-radius: 4px;"
  loading="lazy">
</iframe>

<noscript>
The interactive demo requires JavaScript and WebAssembly support.
</noscript>

## Running locally

If the embedded demo above does not load, you can run it locally:

```bash
# Build the WASM package
wasm-pack build crates/radsym-wasm --target web --release

# Serve the repository root
python3 -m http.server 8080

# Open the demo
open http://localhost:8080/demo/
```

## What the demo shows

| Panel | Description |
|-------|-------------|
| **Original** | Source image (default: ringgrid.png calibration target) |
| **Response Heatmap** | Response map from the selected algorithm, colorized |
| **Seed Proposals** | NMS-extracted proposal locations (crosses) |
| **Detected Circles** | Full pipeline output with overlay annotations |

## Available algorithms

| Algorithm | Method | Description |
|-----------|--------|-------------|
| **FRST** | `frst_response` | Multi-radius voting with orientation accumulator |
| **FRST (fused)** | `multiradius_response` | Single-pass fused variant, faster |
| **RSD** | `rsd_response` | Magnitude-only voting, ~2x faster than FRST |
| **RSD (fused)** | `rsd_response_fused` | Single-pass fused RSD |
