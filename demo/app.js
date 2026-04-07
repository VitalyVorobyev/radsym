import init, { RadSymProcessor } from '../crates/radsym-wasm/pkg/radsym_wasm.js';

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const $ = id => document.getElementById(id);

const cvOriginal  = $('cvOriginal');
const cvHeatmap   = $('cvHeatmap');
const cvGradient  = $('cvGradient');
const cvCircles   = $('cvCircles');
const runBtn      = $('runBtn');
const fileInput   = $('fileInput');
const statusEl    = $('status');
const timingEl    = $('timing');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let processor = null;
let currentPixels = null;  // Uint8Array RGBA
let currentW = 0;
let currentH = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setStatus(msg) { statusEl.textContent = msg; }

function parseRadii(str) {
  return new Uint32Array(
    str.split(',').map(s => parseInt(s.trim(), 10)).filter(n => n > 0)
  );
}

function drawRGBA(canvas, rgba, w, h) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  img.data.set(new Uint8ClampedArray(rgba.buffer || rgba));
  ctx.putImageData(img, 0, 0);
}

function drawImageOnCanvas(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
}

function getPixelsFromCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function loadImageFromURL(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image: ' + url));
    img.src = url;
  });
}

// ---------------------------------------------------------------------------
// Visualization: gradient magnitude
// ---------------------------------------------------------------------------

function renderGradientMagnitude(gradField, w, h) {
  const n = w * h;
  const mag = new Float32Array(n);
  let maxMag = 0;
  for (let i = 0; i < n; i++) {
    const gx = gradField[i * 2];
    const gy = gradField[i * 2 + 1];
    const m = Math.sqrt(gx * gx + gy * gy);
    mag[i] = m;
    if (m > maxMag) maxMag = m;
  }

  const rgba = new Uint8Array(n * 4);
  const scale = maxMag > 0 ? 255 / maxMag : 0;
  for (let i = 0; i < n; i++) {
    const v = Math.round(mag[i] * scale);
    rgba[i * 4]     = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

// ---------------------------------------------------------------------------
// Visualization: circle overlay
// ---------------------------------------------------------------------------

function drawCirclesOverlay(canvas, sourceCanvas, detections) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  // Draw original image as background
  ctx.drawImage(sourceCanvas, 0, 0);

  const count = detections.length / 4;
  if (count === 0) return;

  // Find score range for coloring
  let minScore = Infinity, maxScore = -Infinity;
  for (let i = 0; i < count; i++) {
    const s = detections[i * 4 + 3];
    if (s < minScore) minScore = s;
    if (s > maxScore) maxScore = s;
  }
  const scoreRange = maxScore - minScore || 1;

  ctx.lineWidth = 1.5;
  for (let i = 0; i < count; i++) {
    const x     = detections[i * 4];
    const y     = detections[i * 4 + 1];
    const r     = detections[i * 4 + 2];
    const score = detections[i * 4 + 3];

    // Color: green (low) -> yellow -> red (high)
    const t = (score - minScore) / scoreRange;
    const red   = Math.round(255 * Math.min(1, 2 * t));
    const green = Math.round(255 * Math.min(1, 2 * (1 - t)));

    ctx.strokeStyle = `rgba(${red}, ${green}, 40, 0.85)`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();

    // Small crosshair at center
    ctx.strokeStyle = `rgba(${red}, ${green}, 40, 0.6)`;
    ctx.beginPath();
    ctx.moveTo(x - 2, y); ctx.lineTo(x + 2, y);
    ctx.moveTo(x, y - 2); ctx.lineTo(x, y + 2);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Read config from UI
// ---------------------------------------------------------------------------

function configureProcessor(proc) {
  proc.set_radii(parseRadii($('radii').value));
  proc.set_alpha(parseFloat($('alpha').value));
  proc.set_gradient_threshold(parseFloat($('gradThreshold').value));
  proc.set_smoothing_factor(parseFloat($('smoothing').value));
  proc.set_polarity($('polarity').value);
  proc.set_gradient_operator($('gradOp').value);
  proc.set_radius_hint(parseFloat($('radiusHint').value));
  proc.set_min_score(parseFloat($('minScore').value));
  proc.set_nms_radius(parseInt($('nmsRadius').value, 10));
  proc.set_nms_threshold(parseFloat($('nmsThreshold').value));
  proc.set_max_detections(parseInt($('maxDetections').value, 10));
}

// ---------------------------------------------------------------------------
// Main processing
// ---------------------------------------------------------------------------

function run() {
  if (!currentPixels) return;

  runBtn.disabled = true;
  setStatus('Processing...');
  const timings = [];

  try {
    configureProcessor(processor);
    const w = currentW;
    const h = currentH;
    const pixels = currentPixels;
    const colormap = $('colormap').value;

    // 1. Gradient field
    let t0 = performance.now();
    const gradField = processor.gradient_field(pixels, w, h);
    timings.push(`Gradient: ${(performance.now() - t0).toFixed(1)} ms`);

    const gradRGBA = renderGradientMagnitude(gradField, w, h);
    drawRGBA(cvGradient, gradRGBA, w, h);

    // 2. FRST heatmap
    t0 = performance.now();
    const heatmapData = processor.response_heatmap(pixels, w, h, colormap);
    timings.push(`Heatmap: ${(performance.now() - t0).toFixed(1)} ms`);

    drawRGBA(cvHeatmap, heatmapData, w, h);

    // 3. Circle detection
    t0 = performance.now();
    const detections = processor.detect_circles(pixels, w, h);
    timings.push(`Detection: ${(performance.now() - t0).toFixed(1)} ms`);
    timings.push(`Circles found: ${detections.length / 4}`);

    drawCirclesOverlay(cvCircles, cvOriginal, detections);

    setStatus('Done');
  } catch (e) {
    setStatus('Error: ' + e);
    console.error(e);
  }

  timingEl.innerHTML = timings.join('<br>');
  runBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

function setImage(img) {
  drawImageOnCanvas(cvOriginal, img);
  const imageData = getPixelsFromCanvas(cvOriginal);
  currentPixels = new Uint8Array(imageData.data.buffer);
  currentW = img.width;
  currentH = img.height;

  // Clear other canvases
  for (const cv of [cvHeatmap, cvGradient, cvCircles]) {
    cv.width = img.width;
    cv.height = img.height;
  }
  timingEl.innerHTML = '';
  setStatus(`Image loaded: ${img.width} x ${img.height}`);
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  loadImageFromURL(url).then(img => {
    URL.revokeObjectURL(url);
    setImage(img);
  }).catch(e => setStatus('Error loading file: ' + e));
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function main() {
  try {
    await init();
    processor = new RadSymProcessor();
    setStatus('WASM initialized. Loading test image...');

    const img = await loadImageFromURL('../testdata/ringgrid.png');
    setImage(img);
    runBtn.disabled = false;
  } catch (e) {
    setStatus('Init error: ' + e);
    console.error(e);
  }
}

runBtn.addEventListener('click', run);
main();
