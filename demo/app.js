// radsym interactive demo — WASM driver + pixel-accurate overlay engine.
//
// Coordinate convention (see ADR-001 and gen_demo_samples.rs): the library
// reports centers where an integer coordinate is the PIXEL CENTER. On a canvas,
// `drawImage(img, 0, 0)` places that pixel's center at (x + 0.5, y + 0.5).
// Every positional overlay therefore maps image coord `c` to display coord
// `(c + 0.5) * displayScale` — the `+0.5` is the fix for the classic half-pixel
// overlay offset. Overlays are drawn on a separate, device-pixel-ratio-aware
// canvas so vectors stay crisp while the image stays pixel-sharp.

import init, { RadSymProcessor } from './pkg/radsym_wasm.js';

const DPR = Math.max(1, window.devicePixelRatio || 1);
const HEATMAP_COLORMAP = 'magma';

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const els = {};
[
  'runBtn', 'autoRun', 'status', 'timings', 'gallery', 'fileInput', 'galleryCaption',
  'algorithm', 'radii', 'polarity', 'minScore', 'gradientOperator', 'radiusHint',
  'alpha', 'gradientThreshold', 'smoothingFactor', 'nmsRadius', 'nmsThreshold',
  'maxDetections', 'numAngular', 'numRadial', 'annulusMargin', 'minSamples',
  'weightRingness', 'weightCoverage', 'maxIterations', 'convergenceTol', 'maxCenterDrift',
  'colorMode', 'zoom', 'showDetections', 'showProposals', 'showAnnulus', 'showLabels',
  'showGroundTruth', 'resetBtn', 'resultSub', 'chipCount', 'chipAlgo', 'viewportWrap',
  'viewport', 'cvImage', 'cvOverlay', 'inspector', 'inspectorSub', 'legendMode',
  'legendRamp', 'legendMax', 'legendItems', 'legendNote', 'cvOriginal', 'cvGradient',
  'cvHeatmap', 'cvProposals', 'cvProposalsOv', 'heatmapTitle',
].forEach((id) => (els[id] = $(id)));

// Parameter controls that require re-running detection when changed.
const DETECTION_PARAMS = [
  'algorithm', 'radii', 'polarity', 'minScore', 'gradientOperator', 'radiusHint',
  'alpha', 'gradientThreshold', 'smoothingFactor', 'nmsRadius', 'nmsThreshold',
  'maxDetections', 'numAngular', 'numRadial', 'annulusMargin', 'minSamples',
  'weightRingness', 'weightCoverage', 'maxIterations', 'convergenceTol', 'maxCenterDrift',
];
// View-only controls that just redraw overlays / legend.
const VIEW_PARAMS = ['colorMode', 'showDetections', 'showProposals', 'showAnnulus', 'showLabels', 'showGroundTruth'];

// Slider value-label formatters.
const VAL_FMT = {
  minScore: (v) => v.toFixed(2), radiusHint: (v) => v.toFixed(1), alpha: (v) => v.toFixed(1),
  gradientThreshold: (v) => v.toFixed(0), smoothingFactor: (v) => v.toFixed(1),
  nmsRadius: (v) => v.toFixed(0), nmsThreshold: (v) => v.toFixed(2), numAngular: (v) => v.toFixed(0),
  numRadial: (v) => v.toFixed(0), annulusMargin: (v) => v.toFixed(2), minSamples: (v) => v.toFixed(0),
  weightRingness: (v) => v.toFixed(2), weightCoverage: (v) => v.toFixed(2), maxIterations: (v) => v.toFixed(0),
  convergenceTol: (v) => v.toFixed(2), maxCenterDrift: (v) => v.toFixed(2), zoom: (v) => v.toFixed(1) + '×',
};

const DEFAULTS = {
  algorithm: 'frst', radii: '9,11,13,15,17', polarity: 'both', minScore: 0, gradientOperator: 'sobel',
  radiusHint: 10, alpha: 2, gradientThreshold: 0, smoothingFactor: 0.5, nmsRadius: 5, nmsThreshold: 0,
  maxDetections: 1000, numAngular: 64, numRadial: 9, annulusMargin: 0.3, minSamples: 8,
  weightRingness: 0.6, weightCoverage: 0.4, maxIterations: 10, convergenceTol: 0.1, maxCenterDrift: 0.5,
};

const ALGO_LABEL = { frst: 'FRST', frst_fused: 'FRST · fused', rsd: 'RSD', rsd_fused: 'RSD · fused' };
const STATUS_LABEL = ['Converged', 'Max iterations', 'Degenerate', 'Out of bounds'];
const STATUS_COLOR = ['#3ddc84', '#f2a93d', '#ff6b6b', '#8294b0'];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let processor = null;
let manifest = [];
let activeSample = null;          // manifest entry or null (upload)
let img = null;                   // { pixels: Uint8Array, w, h }
let results = { detections: [], proposals: [] };
let displayScale = 1;             // CSS px per image px, main viewport
let pinned = null;                // pinned inspector detection

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  try {
    await init();
    processor = new RadSymProcessor();
    manifest = await fetch('./samples.json').then((r) => r.json()).then((d) => d.samples);
    buildGallery();
    wireControls();
    const first = manifest.find((s) => s.id === 'concentric-rings') || manifest[0];
    await selectSample(first);
    els.runBtn.disabled = false;
    setStatus('Ready.');
  } catch (e) {
    setStatus('Failed to initialise: ' + e, true);
    console.error(e);
  }
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle('error', isError);
}

// ---------------------------------------------------------------------------
// Gallery & image loading
// ---------------------------------------------------------------------------

function buildGallery() {
  els.gallery.innerHTML = '';
  for (const s of manifest) {
    const b = document.createElement('button');
    b.className = 'thumb';
    b.dataset.id = s.id;
    b.innerHTML = `<img src="./${s.file}" alt="${s.label}" /><span class="cap">${s.label}</span>`;
    b.addEventListener('click', () => selectSample(s));
    els.gallery.appendChild(b);
  }
  const up = document.createElement('button');
  up.className = 'thumb upload';
  up.innerHTML = `<span class="plus">+</span><span>Upload</span>`;
  up.addEventListener('click', () => els.fileInput.click());
  els.gallery.appendChild(up);

  els.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    await loadFromURL(url, null);
    URL.revokeObjectURL(url);
    markActiveThumb(null);
    els.galleryCaption.textContent = `Uploaded: ${file.name}`;
    maybeRun(true);
  });
}

function markActiveThumb(id) {
  els.gallery.querySelectorAll('.thumb').forEach((t) => t.classList.toggle('active', t.dataset.id === id));
}

async function selectSample(sample) {
  activeSample = sample;
  markActiveThumb(sample.id);
  els.galleryCaption.textContent = sample.caption || '';
  applyParams(sample.params || {});
  // Ground-truth toggle availability.
  const hasGt = Array.isArray(sample.groundTruth) && sample.groundTruth.length > 0;
  els.showGroundTruth.disabled = !hasGt;
  els.showGroundTruth.parentElement.style.opacity = hasGt ? '1' : '0.45';
  await loadFromURL('./' + sample.file, sample);
  maybeRun(true);
}

function loadFromURL(url, sample) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const w = image.naturalWidth, h = image.naturalHeight;
      const off = document.createElement('canvas');
      off.width = w; off.height = h;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(image, 0, 0);
      const data = octx.getImageData(0, 0, w, h).data;
      img = { pixels: new Uint8Array(data), w, h, imageData: octx.getImageData(0, 0, w, h) };
      onImageLoaded();
      resolve();
    };
    image.onerror = reject;
    image.src = url;
  });
}

function onImageLoaded() {
  results = { detections: [], proposals: [] };
  pinned = null;
  // Match every stage frame to the image aspect ratio (avoids letterboxing so
  // overlay coordinates map cleanly).
  document.querySelectorAll('.stage-cell .frame').forEach((f) => (f.style.aspectRatio = `${img.w} / ${img.h}`));
  // Base image into every image-space canvas.
  putImage(els.cvImage);
  putImage(els.cvOriginal);
  putImage(els.cvProposals);
  clearCanvas(els.cvGradient);
  clearCanvas(els.cvHeatmap);
  sizeViewport();
  clearOverlay(els.cvOverlay);
  clearOverlay(els.cvProposalsOv);
  renderInspector(null);
}

function putImage(canvas) {
  canvas.width = img.w; canvas.height = img.h;
  canvas.getContext('2d').putImageData(img.imageData, 0, 0);
}
function clearCanvas(canvas) {
  canvas.width = img.w; canvas.height = img.h;
  const c = canvas.getContext('2d');
  c.clearRect(0, 0, img.w, img.h);
}

// ---------------------------------------------------------------------------
// Viewport sizing & overlay canvas setup (DPR-aware)
// ---------------------------------------------------------------------------

function sizeViewport() {
  if (!img) return;
  const zoom = parseFloat(els.zoom.value);
  const avail = els.viewportWrap.clientWidth - 2; // border
  const baseFit = avail / img.w;
  displayScale = baseFit * zoom;
  const dispW = img.w * displayScale;
  const dispH = img.h * displayScale;
  els.viewport.style.width = dispW + 'px';
  els.viewport.style.height = dispH + 'px';
}

// Prepare an overlay canvas of CSS size (cssW × cssH) with a DPR-scaled backing
// store, returning a 2D context whose user units are CSS pixels.
function prepOverlay(canvas, cssW, cssH) {
  canvas.width = Math.round(cssW * DPR);
  canvas.height = Math.round(cssH * DPR);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return ctx;
}
function clearOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Image → display transform (pixel-center → canvas origin). THE ½px fix.
const TX = (ix) => (ix + 0.5) * displayScale;
const TY = (iy) => (iy + 0.5) * displayScale;
const TR = (r) => r * displayScale;

// ---------------------------------------------------------------------------
// Detection run
// ---------------------------------------------------------------------------

let runScheduled = null;
function maybeRun(force = false) {
  if (!processor || !img) return;
  if (force || els.autoRun.checked) {
    clearTimeout(runScheduled);
    runScheduled = setTimeout(run, force ? 0 : 180);
  }
}

function configureProcessor() {
  const p = processor;
  const radii = els.radii.value.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0);
  if (radii.length) p.set_radii(new Uint32Array(radii));
  p.set_polarity(els.polarity.value);
  p.set_gradient_operator(els.gradientOperator.value);
  p.set_radius_hint(num('radiusHint'));
  p.set_alpha(num('alpha'));
  p.set_gradient_threshold(num('gradientThreshold'));
  p.set_smoothing_factor(num('smoothingFactor'));
  p.set_min_score(num('minScore'));
  p.set_nms_radius(int('nmsRadius'));
  p.set_nms_threshold(num('nmsThreshold'));
  p.set_max_detections(int('maxDetections'));
  p.set_num_angular_samples(int('numAngular'));
  p.set_num_radial_samples(int('numRadial'));
  p.set_annulus_margin(num('annulusMargin'));
  p.set_min_samples(int('minSamples'));
  p.set_weight_ringness(num('weightRingness'));
  p.set_weight_coverage(num('weightCoverage'));
  p.set_max_iterations(int('maxIterations'));
  p.set_convergence_tol(num('convergenceTol'));
  p.set_max_center_drift(num('maxCenterDrift'));
}
const num = (id) => parseFloat(els[id].value);
const int = (id) => parseInt(els[id].value, 10);

function run() {
  if (!processor || !img) return;
  const { pixels: px, w, h } = img;
  const algo = els.algorithm.value;
  try {
    configureProcessor();
    const t = {};
    let s = performance.now();
    const grad = processor.gradient_field(px, w, h); t.gradient = performance.now() - s;
    s = performance.now();
    const heat = processor.response_heatmap(px, w, h, algo, HEATMAP_COLORMAP); t.response = performance.now() - s;
    s = performance.now();
    const props = processor.extract_proposals(px, w, h, algo); t.proposals = performance.now() - s;
    s = performance.now();
    const det = processor.detect_circles_detailed_with(px, w, h, algo); t.detect = performance.now() - s;

    results.detections = parseDetections(det);
    results.proposals = parseProposals(props);
    pinned = null;

    renderGradient(grad);
    renderHeatmapBuf(heat);
    renderProposalsStage();
    renderOverlay();
    renderTimings(t);
    renderLegend();
    updateChips();
    renderInspector(null);
    setStatus(`Detected ${results.detections.length} circle${results.detections.length === 1 ? '' : 's'}.`);
  } catch (e) {
    setStatus('' + e, true);
    console.error(e);
  }
}

function parseDetections(arr) {
  const out = [];
  for (let i = 0; i + 7 < arr.length; i += 8) {
    out.push({
      x: arr[i], y: arr[i + 1], r: arr[i + 2], score: arr[i + 3],
      ringness: arr[i + 4], coverage: arr[i + 5], degenerate: arr[i + 6] > 0.5, status: arr[i + 7] | 0,
    });
  }
  return out;
}
function parseProposals(arr) {
  const out = [];
  for (let i = 0; i + 2 < arr.length; i += 3) out.push({ x: arr[i], y: arr[i + 1], score: arr[i + 2] });
  return out;
}

// ---------------------------------------------------------------------------
// Rendering — main overlay
// ---------------------------------------------------------------------------

function renderOverlay() {
  if (!img) return;
  const cssW = img.w * displayScale, cssH = img.h * displayScale;
  const ctx = prepOverlay(els.cvOverlay, cssW, cssH);
  const mode = els.colorMode.value;

  if (els.showProposals.checked) {
    ctx.strokeStyle = 'rgba(130,148,176,0.85)';
    ctx.lineWidth = 1;
    const sz = 3;
    for (const p of results.proposals) {
      const x = TX(p.x), y = TY(p.y);
      ctx.beginPath();
      ctx.moveTo(x - sz, y); ctx.lineTo(x + sz, y);
      ctx.moveTo(x, y - sz); ctx.lineTo(x, y + sz);
      ctx.stroke();
    }
  }

  if (els.showGroundTruth.checked && activeSample && Array.isArray(activeSample.groundTruth)) {
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 1.25;
    for (const g of activeSample.groundTruth) {
      const x = TX(g.x), y = TY(g.y);
      ctx.beginPath(); ctx.arc(x, y, TR(g.r), 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y);
      ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5);
      ctx.stroke();
      ctx.setLineDash([4, 3]);
    }
    ctx.setLineDash([]);
  }

  if (els.showDetections.checked) {
    for (const d of results.detections) drawDetection(ctx, d, mode);
  }
}

function drawDetection(ctx, d, mode) {
  const x = TX(d.x), y = TY(d.y), r = TR(d.r);
  const color = detectionColor(d, mode);

  if (els.showAnnulus.checked) {
    const m = num('annulusMargin');
    ctx.strokeStyle = 'rgba(51,198,227,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.arc(x, y, TR(d.r * (1 - m)), 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, TR(d.r * (1 + m)), 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();

  // center crosshair
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.moveTo(x - 4, y); ctx.lineTo(x + 4, y);
  ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
  ctx.stroke();

  if (els.showLabels.checked) {
    const label = d.score.toFixed(2);
    ctx.font = '600 11px ui-monospace, monospace';
    ctx.textAlign = 'center';
    const ty = y - r - 5;
    ctx.fillStyle = 'rgba(6,10,20,0.85)';
    ctx.fillText(label, x + 0.6, ty + 0.6);
    ctx.fillStyle = color;
    ctx.fillText(label, x, ty);
    ctx.textAlign = 'start';
  }
}

// ---------------------------------------------------------------------------
// Rendering — secondary stages
// ---------------------------------------------------------------------------

function renderGradient(grad) {
  const { w, h } = img;
  const n = w * h;
  let max = 1e-6;
  const mag = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const gx = grad[i * 2], gy = grad[i * 2 + 1];
    const m = Math.hypot(gx, gy);
    mag[i] = m; if (m > max) max = m;
  }
  const buf = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const v = Math.min(255, (mag[i] / max) * 255);
    buf[i * 4] = v; buf[i * 4 + 1] = v; buf[i * 4 + 2] = v; buf[i * 4 + 3] = 255;
  }
  els.cvGradient.width = w; els.cvGradient.height = h;
  els.cvGradient.getContext('2d').putImageData(new ImageData(buf, w, h), 0, 0);
}

function renderHeatmapBuf(heat) {
  const { w, h } = img;
  els.cvHeatmap.width = w; els.cvHeatmap.height = h;
  els.cvHeatmap.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(heat), w, h), 0, 0);
  els.heatmapTitle.textContent = `Response heatmap — ${ALGO_LABEL[els.algorithm.value]}`;
}

function renderProposalsStage() {
  putImage(els.cvProposals);
  const frame = els.cvProposalsOv.parentElement;
  const cssW = frame.clientWidth, cssH = frame.clientHeight;
  const ctx = prepOverlay(els.cvProposalsOv, cssW, cssH);
  const scale = cssW / img.w;
  ctx.strokeStyle = 'rgba(242,169,61,0.9)';
  ctx.lineWidth = 1;
  const sz = 2.5;
  for (const p of results.proposals) {
    const x = (p.x + 0.5) * scale, y = (p.y + 0.5) * scale;
    ctx.beginPath();
    ctx.moveTo(x - sz, y); ctx.lineTo(x + sz, y);
    ctx.moveTo(x, y - sz); ctx.lineTo(x, y + sz);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Color mapping
// ---------------------------------------------------------------------------

const RAMP = [
  [0.0, [91, 157, 255]],   // blue
  [0.5, [51, 198, 227]],   // teal
  [1.0, [61, 220, 132]],   // green
];
function rampColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i];
      const f = (t - t0) / (t1 - t0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return `rgb(${RAMP[RAMP.length - 1][1].join(',')})`;
}
function detectionColor(d, mode) {
  switch (mode) {
    case 'ringness': return rampColor(d.ringness);
    case 'coverage': return rampColor(d.coverage);
    case 'status': return STATUS_COLOR[d.status] || '#8294b0';
    case 'uniform': return '#f2a93d';
    default: return rampColor(d.score);
  }
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------

function renderLegend() {
  const mode = els.colorMode.value;
  const labels = { score: 'support score', ringness: 'ringness', coverage: 'angular coverage', status: 'refinement status', uniform: 'uniform' };
  els.legendMode.textContent = labels[mode];
  const ramp = els.legendRamp;
  const items = els.legendItems;
  items.innerHTML = '';

  if (mode === 'status') {
    ramp.style.background = 'transparent';
    ramp.style.border = 'none';
    els.legendMax.textContent = '';
    STATUS_LABEL.forEach((lab, i) => {
      const el = document.createElement('span');
      el.className = 'item';
      el.innerHTML = `<span class="swatch" style="background:${STATUS_COLOR[i]}"></span>${lab}`;
      items.appendChild(el);
    });
    els.legendNote.textContent = 'Circle color encodes how refinement terminated for each detection.';
  } else if (mode === 'uniform') {
    ramp.style.background = '#f2a93d';
    ramp.style.border = '1px solid var(--edge)';
    els.legendMax.textContent = '';
    els.legendNote.textContent = 'All detections drawn in the demo accent color; use a metric mode to compare quality.';
  } else {
    ramp.style.background = 'linear-gradient(90deg, rgb(91,157,255), rgb(51,198,227), rgb(61,220,132))';
    ramp.style.border = '1px solid var(--edge)';
    els.legendMax.textContent = '1.0';
    const notes = {
      score: 'Combined support score (ringness × weight + coverage × weight). Higher = stronger radial evidence.',
      ringness: 'How consistently the gradient points at the center around the annulus.',
      coverage: 'Fraction of the annulus with supporting gradient evidence.',
    };
    els.legendNote.textContent = notes[mode];
  }

  // Always show the fixed reference marks.
  const refs = [
    ['Proposal (pre-NMS)', 'background:rgba(130,148,176,0.9)'],
    ['Annulus sampled', 'background:rgba(51,198,227,0.6)'],
    ['Ground truth', 'background:#ffffff'],
  ];
  for (const [lab, style] of refs) {
    const el = document.createElement('span');
    el.className = 'item';
    el.innerHTML = `<span class="swatch" style="${style}"></span>${lab}`;
    items.appendChild(el);
  }
}

// ---------------------------------------------------------------------------
// Timings
// ---------------------------------------------------------------------------

const TIMING_COLORS = { gradient: '#f2a93d', response: '#5b9dff', proposals: '#33c6e3', detect: '#3ddc84' };
const TIMING_LABEL = { gradient: 'gradient', response: 'response', proposals: 'proposals', detect: 'score+refine' };
function renderTimings(t) {
  const max = Math.max(...Object.values(t), 0.01);
  els.timings.innerHTML = '';
  for (const k of ['gradient', 'response', 'proposals', 'detect']) {
    const row = document.createElement('div');
    row.className = 'timing-row';
    row.innerHTML = `<span>${TIMING_LABEL[k]}</span><span class="bar"><span style="width:${(t[k] / max) * 100}%;background:${TIMING_COLORS[k]}"></span></span><span class="ms">${t[k].toFixed(1)}ms</span>`;
    els.timings.appendChild(row);
  }
}

// ---------------------------------------------------------------------------
// Chips & inspector
// ---------------------------------------------------------------------------

function updateChips() {
  els.chipCount.innerHTML = `<b>${results.detections.length}</b> detection${results.detections.length === 1 ? '' : 's'}`;
  els.chipAlgo.textContent = ALGO_LABEL[els.algorithm.value];
  const algo = els.algorithm.value;
  els.resultSub.textContent = algo === 'frst'
    ? 'FRST proposals (per-proposal scale) → score → refine'
    : `${ALGO_LABEL[algo]} proposals (at radius hint) → score → refine`;
}

function renderInspector(d) {
  if (!d) {
    if (results.detections.length === 0) {
      els.inspector.innerHTML = '<div class="inspector-hint">Run a detection, then hover or click a circle to see its subpixel center, radius, and score breakdown.</div>';
      els.inspectorSub.textContent = 'hover a detection';
      return;
    }
    els.inspectorSub.textContent = pinned ? 'pinned — click empty space to release' : 'hover a detection';
    d = pinned;
    if (!d) {
      els.inspector.innerHTML = '<div class="inspector-hint">Hover a detected circle in the view above.</div>';
      return;
    }
  }
  const cell = (k, v, small) => `<div class="cell"><div class="k">${k}</div><div class="v${small ? ' small' : ''}">${v}</div></div>`;
  els.inspector.innerHTML =
    cell('center x', d.x.toFixed(2)) +
    cell('center y', d.y.toFixed(2)) +
    cell('radius', d.r.toFixed(2)) +
    cell('score', d.score.toFixed(3)) +
    cell('ringness', d.ringness.toFixed(3)) +
    cell('coverage', d.coverage.toFixed(3)) +
    cell('status', STATUS_LABEL[d.status] || '—', true);
}

function pickDetection(clientX, clientY) {
  const rect = els.cvOverlay.getBoundingClientRect();
  const ix = (clientX - rect.left) / displayScale - 0.5;
  const iy = (clientY - rect.top) / displayScale - 0.5;
  let best = null, bestD = Infinity;
  for (const d of results.detections) {
    const dist = Math.hypot(d.x - ix, d.y - iy);
    if (dist < Math.max(d.r + 4, 8) && dist < bestD) { bestD = dist; best = d; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Control wiring
// ---------------------------------------------------------------------------

function wireControls() {
  els.runBtn.addEventListener('click', () => run());
  els.resetBtn.addEventListener('click', () => { applyParams(DEFAULTS); maybeRun(true); });

  for (const id of DETECTION_PARAMS) {
    const ev = els[id].type === 'range' ? 'input' : 'change';
    els[id].addEventListener(ev, () => { syncVal(id); maybeRun(); });
  }
  for (const id of VIEW_PARAMS) {
    els[id].addEventListener('change', () => { renderOverlay(); renderLegend(); });
  }
  els.zoom.addEventListener('input', () => { syncVal('zoom'); sizeViewport(); renderOverlay(); });

  // Inspector interactions on the main overlay.
  els.cvOverlay.addEventListener('mousemove', (e) => {
    if (pinned) return;
    renderInspector(pickDetection(e.clientX, e.clientY));
  });
  els.cvOverlay.addEventListener('mouseleave', () => { if (!pinned) renderInspector(null); });
  els.cvOverlay.addEventListener('click', (e) => {
    const d = pickDetection(e.clientX, e.clientY);
    pinned = d && pinned !== d ? d : null;
    renderInspector(pinned);
  });

  let rz;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { sizeViewport(); renderOverlay(); renderProposalsStage(); }, 120);
  });
}

function syncVal(id) {
  const span = $(id + '-val');
  if (span && VAL_FMT[id]) span.textContent = VAL_FMT[id](parseFloat(els[id].value));
}

// Apply a params object (from a sample or defaults) to the controls.
function applyParams(p) {
  const setIf = (id, key) => { if (p[key] !== undefined && els[id]) els[id].value = p[key]; };
  setIf('algorithm', 'algorithm');
  if (p.radii !== undefined) els.radii.value = Array.isArray(p.radii) ? p.radii.join(',') : p.radii;
  setIf('polarity', 'polarity');
  setIf('minScore', 'minScore');
  setIf('gradientOperator', 'gradientOperator');
  setIf('radiusHint', 'radiusHint');
  setIf('alpha', 'alpha');
  setIf('gradientThreshold', 'gradientThreshold');
  setIf('smoothingFactor', 'smoothingFactor');
  setIf('nmsRadius', 'nmsRadius');
  setIf('nmsThreshold', 'nmsThreshold');
  setIf('maxDetections', 'maxDetections');
  setIf('numAngular', 'numAngular');
  setIf('numRadial', 'numRadial');
  setIf('annulusMargin', 'annulusMargin');
  setIf('minSamples', 'minSamples');
  setIf('weightRingness', 'weightRingness');
  setIf('weightCoverage', 'weightCoverage');
  setIf('maxIterations', 'maxIterations');
  setIf('convergenceTol', 'convergenceTol');
  setIf('maxCenterDrift', 'maxCenterDrift');
  // Refresh all value labels.
  Object.keys(VAL_FMT).forEach(syncVal);
}

// Debug hook for automated verification (harmless in normal use): exposes the
// live results, the image→display transform, and the active sample so a test
// can assert overlay alignment against ground truth.
window.__demo = {
  get ready() { return !!processor && !!img && results.detections.length >= 0; },
  get results() { return results; },
  get displayScale() { return displayScale; },
  get sample() { return activeSample; },
  tx: (ix) => TX(ix), ty: (iy) => TY(iy), tr: (r) => TR(r),
  run,
};

boot();
