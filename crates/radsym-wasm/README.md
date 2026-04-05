# radsym

WebAssembly bindings for the [radsym](https://github.com/VitalyVorobyev/radsym)
radial symmetry detection library.

Detect circles and visualize FRST response heatmaps directly in the browser.

## Building

```bash
wasm-pack build crates/radsym-wasm --target web --release
```

## Installation

From the built package:

```bash
npm install ./crates/radsym-wasm/pkg
```

Or from npm (when published):

```bash
npm install @vitavision/radsym
```

## Usage

### Initialization

```js
import init, { RadSymProcessor } from '@vitavision/radsym';

await init();
const processor = new RadSymProcessor();
```

### Detect circles from a canvas

```js
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

processor.set_radii(new Uint32Array([8, 10, 12]));
processor.set_polarity('bright');
processor.set_radius_hint(10.0);

const result = processor.detect_circles(imageData.data, canvas.width, canvas.height);
// result is Float32Array with stride 4: [x, y, radius, score, ...]
for (let i = 0; i < result.length; i += 4) {
  console.log(`Circle at (${result[i]}, ${result[i+1]}) r=${result[i+2]} score=${result[i+3]}`);
}
```

### Visualize FRST response heatmap

```js
const heatmap = processor.response_heatmap(imageData.data, w, h, 'hot');
// heatmap is Uint8Array of RGBA pixels (length w * h * 4)
const img = ctx.createImageData(w, h);
img.data.set(new Uint8ClampedArray(heatmap));
ctx.putImageData(img, 0, 0);
```

### Raw FRST response

```js
const response = processor.frst_response(imageData.data, w, h);
// Float32Array of length w * h, row-major
```

### Gradient field

```js
const gradient = processor.gradient_field(imageData.data, w, h);
// Float32Array with stride 2: [gx, gy, ...], length w * h * 2
```

## Configuration

All `DetectCirclesConfig` fields are exposed as flat setters:

| Method | Default | Description |
|--------|---------|-------------|
| `set_radii(Uint32Array)` | `[5..15]` | Radii to test (pixels) |
| `set_alpha(f32)` | `2.0` | Radial strictness exponent |
| `set_gradient_threshold(f32)` | `0.0` | Min gradient magnitude for voting |
| `set_smoothing_factor(f32)` | `0.5` | Gaussian smoothing sigma factor |
| `set_nms_radius(usize)` | `5` | NMS suppression radius |
| `set_nms_threshold(f32)` | `0.0` | NMS minimum response |
| `set_max_detections(usize)` | `50` | Max detection budget |
| `set_num_angular_samples(usize)` | `36` | Angular samples in annulus |
| `set_num_radial_samples(usize)` | `3` | Radial samples in annulus |
| `set_annulus_margin(f32)` | `0.3` | Annulus width fraction |
| `set_min_samples(usize)` | `8` | Min gradient samples |
| `set_weight_ringness(f32)` | `0.7` | Ringness weight in score |
| `set_weight_coverage(f32)` | `0.3` | Coverage weight in score |
| `set_max_iterations(usize)` | `10` | Max refinement iterations |
| `set_convergence_tol(f32)` | `0.1` | Refinement convergence (px) |
| `set_polarity(str)` | `"both"` | `"bright"`, `"dark"`, `"both"` |
| `set_radius_hint(f32)` | `10.0` | Initial radius hypothesis |
| `set_min_score(f32)` | `0.0` | Minimum score threshold |
| `set_gradient_operator(str)` | `"sobel"` | `"sobel"` or `"scharr"` |

## Output formats

| Method | Return type | Stride | Fields |
|--------|------------|--------|--------|
| `detect_circles` | `Float32Array` | 4 | `x, y, radius, score` |
| `frst_response` | `Float32Array` | 1 | response value |
| `response_heatmap` | `Uint8Array` | 4 | R, G, B, A |
| `gradient_field` | `Float32Array` | 2 | gx, gy |

## Cleanup

Call `processor.free()` when done to release WASM memory.
