import init, { RadSymProcessor } from './pkg/radsym_wasm.js';

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const $ = id => document.getElementById(id);

const cvOriginal   = $('cvOriginal');
const cvHeatmap    = $('cvHeatmap');
const cvGradient   = $('cvGradient');
const cvProposals  = $('cvProposals');
const cvCircles    = $('cvCircles');
const runBtn       = $('runBtn');
const fileInput    = $('fileInput');
const statusEl     = $('status');
const timingEl     = $('timing');
const detailBox    = $('detailBox');
const detailContent = $('detailContent');
const heatmapTitle = $('heatmapTitle');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let processor = null;
let currentPixels = null;  // Uint8Array RGBA
let currentW = 0;
let currentH = 0;
let lastDetections = [];   // parsed detection objects

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DET_STRIDE = 8;
const PROP_STRIDE = 3;
const ALGO_LABELS = {
  frst: 'FRST',
  frst_fused: 'FRST (fused)',
  rsd: 'RSD',
  rsd_fused: 'RSD (fused)',
};
const STATUS_LABELS = ['Converged', 'MaxIterations', 'Degenerate', 'OutOfBounds'];
const STATUS_COLORS = ['#2a2', '#da0', '#d33', '#999'];

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

/** Parse stride-8 Float32Array into detection objects. */
function parseDetections(flat) {
  const dets = [];
  const count = flat.length / DET_STRIDE;
  for (let i = 0; i < count; i++) {
    const b = i * DET_STRIDE;
    dets.push({
      x:        flat[b],
      y:        flat[b + 1],
      r:        flat[b + 2],
      total:    flat[b + 3],
      ringness: flat[b + 4],
      coverage: flat[b + 5],
      degenerate: flat[b + 6] === 1.0,
      status:   flat[b + 7],
    });
  }
  return dets;
}

/** Parse stride-3 Float32Array into proposal objects. */
function parseProposals(flat) {
  const props = [];
  const count = flat.length / PROP_STRIDE;
  for (let i = 0; i < count; i++) {
    const b = i * PROP_STRIDE;
    props.push({ x: flat[b], y: flat[b + 1], score: flat[b + 2] });
  }
  return props;
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
// Visualization: proposals overlay
// ---------------------------------------------------------------------------

function drawProposalsOverlay(canvas, sourceCanvas, proposals) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceCanvas, 0, 0);

  if (proposals.length === 0) return;

  let minS = Infinity, maxS = -Infinity;
  for (const p of proposals) {
    if (p.score < minS) minS = p.score;
    if (p.score > maxS) maxS = p.score;
  }
  const range = maxS - minS || 1;

  for (const p of proposals) {
    const t = (p.score - minS) / range;
    const r = Math.round(255 * Math.min(1, 2 * t));
    const g = Math.round(255 * Math.min(1, 2 * (1 - t)));
    ctx.strokeStyle = `rgba(${r}, ${g}, 40, 0.8)`;
    ctx.lineWidth = 1.5;

    // Cross marker
    const sz = 4;
    ctx.beginPath();
    ctx.moveTo(p.x - sz, p.y); ctx.lineTo(p.x + sz, p.y);
    ctx.moveTo(p.x, p.y - sz); ctx.lineTo(p.x, p.y + sz);
    ctx.stroke();
  }

  // Legend
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath();
  ctx.roundRect(w - 110, 8, 102, 20, 4);
  ctx.fill();
  ctx.font = '9px monospace';
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'left';
  ctx.fillText(`Proposals: ${proposals.length}`, w - 104, 22);
}

// ---------------------------------------------------------------------------
// Color helpers for detection overlay
// ---------------------------------------------------------------------------

function scoreColor(t) {
  const r = Math.round(255 * Math.min(1, 2 * t));
  const g = Math.round(255 * Math.min(1, 2 * (1 - t)));
  return `rgba(${r}, ${g}, 40, 0.85)`;
}

function ringnessColor(t) {
  const r = Math.round(40 + 215 * t);
  const g = Math.round(80 + 100 * t);
  const b = Math.round(220 - 180 * t);
  return `rgba(${r}, ${g}, ${b}, 0.85)`;
}

function coverageColor(t) {
  const r = Math.round(160 - 130 * t);
  const g = Math.round(60 + 195 * t);
  const b = Math.round(200 + 55 * t);
  return `rgba(${r}, ${g}, ${b}, 0.85)`;
}

function detColor(det, mode, minMax) {
  if (mode === 'status') {
    const idx = Math.min(Math.max(Math.round(det.status), 0), 3);
    return STATUS_COLORS[idx];
  }
  const field = mode === 'ringness' ? 'ringness'
    : mode === 'coverage' ? 'coverage' : 'total';
  const colorFn = mode === 'ringness' ? ringnessColor
    : mode === 'coverage' ? coverageColor : scoreColor;
  const range = minMax[field].max - minMax[field].min || 1;
  const t = (det[field] - minMax[field].min) / range;
  return colorFn(t);
}

function computeMinMax(dets) {
  const mm = {
    total:    { min: Infinity, max: -Infinity },
    ringness: { min: Infinity, max: -Infinity },
    coverage: { min: Infinity, max: -Infinity },
  };
  for (const d of dets) {
    for (const k of ['total', 'ringness', 'coverage']) {
      if (d[k] < mm[k].min) mm[k].min = d[k];
      if (d[k] > mm[k].max) mm[k].max = d[k];
    }
  }
  return mm;
}

// ---------------------------------------------------------------------------
// Visualization: circle overlay
// ---------------------------------------------------------------------------

function drawCirclesOverlay(canvas, sourceCanvas, dets, opts) {
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(sourceCanvas, 0, 0);

  if (dets.length === 0) return;
  const minMax = computeMinMax(dets);

  ctx.lineWidth = 1.5;
  for (const d of dets) {
    const color = detColor(d, opts.colorMode, minMax);
    ctx.strokeStyle = color;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(d.x - 3, d.y); ctx.lineTo(d.x + 3, d.y);
    ctx.moveTo(d.x, d.y - 3); ctx.lineTo(d.x, d.y + 3);
    ctx.stroke();
  }

  if (opts.showAnnulus) {
    const margin = opts.annulusMargin;
    ctx.lineWidth = 0.8;
    ctx.setLineDash([3, 3]);
    for (const d of dets) {
      ctx.strokeStyle = detColor(d, opts.colorMode, minMax).replace('0.85', '0.4');
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r * (1 - margin), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r * (1 + margin), 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  if (opts.showLabels) {
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    const field = opts.colorMode === 'ringness' ? 'ringness'
      : opts.colorMode === 'coverage' ? 'coverage' : 'total';
    for (const d of dets) {
      const val = opts.colorMode === 'status'
        ? STATUS_LABELS[Math.round(d.status)] || '?'
        : d[field].toFixed(2);
      const ty = d.y - d.r - 4;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillText(val, d.x + 0.5, ty + 0.5);
      ctx.fillStyle = '#fff';
      ctx.fillText(val, d.x, ty);
    }
  }

  if (opts.showStatus) {
    ctx.font = 'bold 8px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const d of dets) {
      const idx = Math.min(Math.max(Math.round(d.status), 0), 3);
      const label = ['C', 'M', 'D', 'O'][idx];
      const sx = d.x + d.r * 0.7;
      const sy = d.y + d.r * 0.7;
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = STATUS_COLORS[idx];
      ctx.fillText(label, sx, sy);
    }
    ctx.textBaseline = 'alphabetic';
  }

  if (opts.showLegend) {
    drawLegend(ctx, opts.colorMode, minMax, dets.length, w);
  }
}

function drawLegend(ctx, mode, minMax, count, canvasW) {
  const lw = 120, lh = mode === 'status' ? 72 : 48;
  const lx = canvasW - lw - 8, ly = 8;

  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.beginPath();
  ctx.roundRect(lx, ly, lw, lh, 4);
  ctx.fill();

  ctx.font = '9px monospace';
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'left';

  const modeLabel = mode === 'ringness' ? 'Ringness'
    : mode === 'coverage' ? 'Coverage'
    : mode === 'status' ? 'Status' : 'Score';
  ctx.fillText(`${modeLabel}  n=${count}`, lx + 6, ly + 13);

  if (mode === 'status') {
    for (let i = 0; i < 4; i++) {
      ctx.fillStyle = STATUS_COLORS[i];
      ctx.fillRect(lx + 6, ly + 20 + i * 12, 8, 8);
      ctx.fillStyle = '#fff';
      ctx.fillText(STATUS_LABELS[i], lx + 18, ly + 28 + i * 12);
    }
  } else {
    const barX = lx + 6, barY = ly + 20, barW = lw - 12, barH = 8;
    const colorFn = mode === 'ringness' ? ringnessColor
      : mode === 'coverage' ? coverageColor : scoreColor;
    for (let px = 0; px < barW; px++) {
      ctx.fillStyle = colorFn(px / barW);
      ctx.fillRect(barX + px, barY, 1, barH);
    }
    const field = mode === 'ringness' ? 'ringness'
      : mode === 'coverage' ? 'coverage' : 'total';
    ctx.fillStyle = '#fff';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(minMax[field].min.toFixed(2), barX, barY + barH + 10);
    ctx.textAlign = 'right';
    ctx.fillText(minMax[field].max.toFixed(2), barX + barW, barY + barH + 10);
    ctx.textAlign = 'left';
  }
}

// ---------------------------------------------------------------------------
// Click-to-inspect
// ---------------------------------------------------------------------------

cvCircles.addEventListener('click', (e) => {
  if (lastDetections.length === 0) return;

  const rect = cvCircles.getBoundingClientRect();
  const scaleX = cvCircles.width / rect.width;
  const scaleY = cvCircles.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  let best = null, bestDist = Infinity;
  for (const d of lastDetections) {
    const dx = d.x - mx, dy = d.y - my;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < d.r + 5 && dist < bestDist) {
      best = d;
      bestDist = dist;
    }
  }

  if (best) {
    const statusIdx = Math.min(Math.max(Math.round(best.status), 0), 3);
    detailContent.textContent = [
      `Center:     (${best.x.toFixed(2)}, ${best.y.toFixed(2)})`,
      `Radius:     ${best.r.toFixed(2)} px`,
      `Score:      ${best.total.toFixed(4)}`,
      `Ringness:   ${best.ringness.toFixed(4)}`,
      `Coverage:   ${best.coverage.toFixed(4)}`,
      `Degenerate: ${best.degenerate}`,
      `Status:     ${STATUS_LABELS[statusIdx]}`,
    ].join('\n');
    detailBox.style.display = 'block';
  }
});

// ---------------------------------------------------------------------------
// Read config from UI
// ---------------------------------------------------------------------------

function configureProcessor(proc) {
  // Algorithm config (shared by FRST and RSD)
  proc.set_radii(parseRadii($('radii').value));
  proc.set_alpha(parseFloat($('alpha').value));
  proc.set_gradient_threshold(parseFloat($('gradThreshold').value));
  proc.set_smoothing_factor(parseFloat($('smoothing').value));

  // Detection
  proc.set_polarity($('polarity').value);
  proc.set_gradient_operator($('gradOp').value);
  proc.set_radius_hint(parseFloat($('radiusHint').value));
  proc.set_min_score(parseFloat($('minScore').value));

  // NMS
  proc.set_nms_radius(parseInt($('nmsRadius').value, 10));
  proc.set_nms_threshold(parseFloat($('nmsThreshold').value));
  proc.set_max_detections(parseInt($('maxDetections').value, 10));

  // Support Scoring
  proc.set_num_angular_samples(parseInt($('numAngularSamples').value, 10));
  proc.set_num_radial_samples(parseInt($('numRadialSamples').value, 10));
  proc.set_annulus_margin(parseFloat($('annulusMargin').value));
  proc.set_min_samples(parseInt($('minSamples').value, 10));
  proc.set_weight_ringness(parseFloat($('weightRingness').value));
  proc.set_weight_coverage(parseFloat($('weightCoverage').value));

  // Refinement
  proc.set_max_iterations(parseInt($('maxIterations').value, 10));
  proc.set_convergence_tol(parseFloat($('convergenceTol').value));
  proc.set_max_center_drift(parseFloat($('maxCenterDrift').value));
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
    const algo = $('algorithm').value;
    const colormap = $('colormap').value;

    heatmapTitle.textContent = `${ALGO_LABELS[algo]} Heatmap`;

    // 1. Gradient magnitude
    let t0 = performance.now();
    const gradField = processor.gradient_field(pixels, w, h);
    timings.push(`Gradient: ${(performance.now() - t0).toFixed(1)} ms`);
    const gradRGBA = renderGradientMagnitude(gradField, w, h);
    drawRGBA(cvGradient, gradRGBA, w, h);

    // 2. Response heatmap
    t0 = performance.now();
    const heatmapData = processor.response_heatmap(pixels, w, h, algo, colormap);
    timings.push(`Heatmap (${ALGO_LABELS[algo]}): ${(performance.now() - t0).toFixed(1)} ms`);
    drawRGBA(cvHeatmap, heatmapData, w, h);

    // 3. Seed proposals
    t0 = performance.now();
    const rawProps = processor.extract_proposals(pixels, w, h, algo);
    const proposals = parseProposals(rawProps);
    timings.push(`Proposals: ${proposals.length} in ${(performance.now() - t0).toFixed(1)} ms`);
    drawProposalsOverlay(cvProposals, cvOriginal, proposals);

    // 4. Circle detection (full pipeline, always uses FRST internally)
    t0 = performance.now();
    const rawDets = processor.detect_circles_detailed(pixels, w, h);
    lastDetections = parseDetections(rawDets);
    timings.push(`Detection: ${lastDetections.length} circles in ${(performance.now() - t0).toFixed(1)} ms`);

    drawCirclesOverlay(cvCircles, cvOriginal, lastDetections, {
      colorMode: $('colorMode').value,
      showLabels: $('showLabels').checked,
      showAnnulus: $('showAnnulus').checked,
      showStatus: $('showStatus').checked,
      showLegend: $('showLegend').checked,
      annulusMargin: parseFloat($('annulusMargin').value),
    });

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

  for (const cv of [cvHeatmap, cvGradient, cvProposals, cvCircles]) {
    cv.width = img.width;
    cv.height = img.height;
  }
  lastDetections = [];
  detailBox.style.display = 'none';
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

    const img = await loadImageFromURL('./ringgrid.png');
    setImage(img);
    runBtn.disabled = false;
  } catch (e) {
    setStatus('Init error: ' + e);
    console.error(e);
  }
}

runBtn.addEventListener('click', run);
main();
