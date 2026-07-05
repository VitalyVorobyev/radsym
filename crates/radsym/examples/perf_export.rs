//! Performance export harness for the GitHub Pages performance page.
//!
//! Times the `detect_circles` pipeline broken down by stage, the FRST/RSD
//! voting backends across image sizes, and the per-call refinement cost, then
//! writes a `data.json` consumed by `.github/pages/performance/index.html`.
//!
//! The per-image stage breakdown mirrors the exact decomposition of
//! [`radsym::detect_circles`] (see `src/pipeline.rs::run_detection`): the
//! gradient is computed once and reused by voting, scoring, and refinement; the
//! FRST response feeds proposal extraction; every proposal is scored; only
//! non-degenerate proposals at or above `min_score` are refined. Timing the same
//! sequence of public stage functions keeps these numbers faithful to a real
//! `detect_circles` call — the page asserts the stage sum ≈ end-to-end.
//!
//! Reproduce (release build for meaningful timings; single-threaded and
//! deterministic — do NOT enable `rayon`):
//!
//! ```sh
//! cargo run --release -p radsym --example perf_export --features image-io,serde \
//!   -- .github/pages/performance/data.json
//! ```
//!
//! Pass `-` as the output path to write the JSON to stdout instead.

use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use serde::Serialize;

use radsym::diagnostics::{DiagnosticImage, overlay_circle, overlay_ellipse};
use radsym::{
    Circle, CircleDetection, DetectCirclesConfig, Ellipse, EllipseRefineConfig, FrstConfig,
    Homography, HomographyEllipseRefineConfig, OwnedImage, PixelCoord, Polarity,
    RadialCenterConfig, RsdConfig, compute_gradient, detect_circles, extract_proposals,
    frst_response, frst_response_fused, frst_response_scaled, load_grayscale,
    radial_center_refine_from_gradient, rectified_circle_to_image_ellipse, refine_circle,
    refine_ellipse, refine_ellipse_homography, rsd_response, rsd_response_fused, save_diagnostic,
    score_circle_support_detailed, sobel_gradient,
};

// The publishable surface-hole case reuses the exact pyramid + ellipse pipeline
// that `tests/surf_hole_regression.rs` asserts is correct, by including the same
// shared source file (single source of truth — no algorithm duplication). It
// renders its own pixels, so nothing private (e.g. `data/surf1.png`) is touched.
#[path = "../tests/support/surf_hole_synthetic.rs"]
mod surf_hole_synthetic;

// Ring-grid outer+inner ellipse detection, shared with `ringgrid_regression.rs`.
#[path = "../tests/support/ringgrid_detect.rs"]
mod ringgrid_detect;

/// Base repeat count reported in `meta.repeats`; sub-megapixel images use this.
const REPEATS: usize = 50;

// ---------------------------------------------------------------------------
// Synthetic image generators (the throughput disk matches benches/frst_bench.rs
// exactly, so page throughput numbers line up with `cargo bench`).
// ---------------------------------------------------------------------------

/// Bright disk with radius `size / 6` — matches `benches/frst_bench.rs`.
fn make_disk_image(size: usize) -> OwnedImage<u8> {
    make_fixed_disk(size, size as f32 / 6.0)
}

/// Bright disk with an explicit radius, centered in a `size`×`size` frame.
fn make_fixed_disk(size: usize, radius: f32) -> OwnedImage<u8> {
    let c = size as f32 / 2.0;
    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            if (dx * dx + dy * dy).sqrt() <= radius {
                data[y * size + x] = 255;
            }
        }
    }
    OwnedImage::from_vec(data, size, size).unwrap()
}

/// Rotated bright ellipse — matches `benches/refinement_bench.rs`.
fn make_ellipse_image(size: usize) -> OwnedImage<u8> {
    let c = size as f32 / 2.0;
    let semi_major = size as f32 / 6.0;
    let semi_minor = size as f32 / 7.5;
    let (sin_a, cos_a) = 0.35f32.sin_cos();
    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let lx = dx * cos_a + dy * sin_a;
            let ly = -dx * sin_a + dy * cos_a;
            if (lx / semi_major).powi(2) + (ly / semi_minor).powi(2) <= 1.0 {
                data[y * size + x] = 255;
            }
        }
    }
    OwnedImage::from_vec(data, size, size).unwrap()
}

/// Disk seen through a homography — matches `benches/refinement_bench.rs`.
fn make_projective_disk_image(
    size: usize,
    homography: &Homography,
    circle: Circle,
) -> OwnedImage<u8> {
    let mut data = vec![20u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let Some(rectified) =
                homography.map_image_to_rectified(PixelCoord::new(x as f32, y as f32))
            else {
                continue;
            };
            let dx = rectified.x - circle.center.x;
            let dy = rectified.y - circle.center.y;
            if (dx * dx + dy * dy).sqrt() <= circle.radius {
                data[y * size + x] = 255;
            }
        }
    }
    OwnedImage::from_vec(data, size, size).unwrap()
}

// ---------------------------------------------------------------------------
// Timing helpers.
// ---------------------------------------------------------------------------

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    }
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Warmup, then return the median wall-clock of `reps` calls, in milliseconds.
fn time_call_ms<T>(reps: usize, warmup: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..warmup {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

/// Per-pixel-count repeat/warmup budget — large images run fewer iterations to
/// keep the whole export to a few minutes while medians stay stable.
fn budget(pixels: usize) -> (usize, usize) {
    match pixels {
        0..=700_000 => (REPEATS, 5),
        700_001..=1_200_000 => (20, 4),
        _ => (8, 3),
    }
}

// ---------------------------------------------------------------------------
// Per-image stage breakdown — a faithful re-timing of run_detection().
// ---------------------------------------------------------------------------

struct StageTimes {
    gradient_ms: f64,
    voting_ms: f64,
    extract_ms: f64,
    score_ms: f64,
    refine_ms: f64,
    detections: Vec<CircleDetection>,
    end_to_end_ms: f64,
}

fn time_stages(image: &OwnedImage<u8>, cfg: &DetectCirclesConfig) -> StageTimes {
    let view = image.view();
    let (reps, warmup) = budget(image.width() * image.height());

    let (mut grad, mut vote, mut extract, mut score, mut refine) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for i in 0..(reps + warmup) {
        // Stage 1: gradient (once, reused below) — mirrors pipeline.rs:267.
        let t = Instant::now();
        let gradient = compute_gradient(black_box(&view), cfg.gradient_operator).unwrap();
        let g_ms = t.elapsed().as_secs_f64() * 1e3;

        // Stage 2: FRST voting. radii/polarity come from the top-level config
        // (their single source of truth). Times `frst_response_scaled`, not
        // plain `frst_response` — `run_detection` (pipeline.rs) calls the
        // scaled variant to get the per-pixel winning-radius map, which costs
        // extra `best`/`scale` buffer writes on top of the summed response.
        let frst_config = cfg
            .advanced
            .frst
            .to_frst_config(cfg.radii.clone(), cfg.polarity);
        let t = Instant::now();
        let scaled = frst_response_scaled(black_box(&gradient), black_box(&frst_config)).unwrap();
        let v_ms = t.elapsed().as_secs_f64() * 1e3;
        let response = scaled.response;

        // Stage 3: NMS proposal extraction — mirrors pipeline.rs:274.
        let t = Instant::now();
        let proposals = extract_proposals(black_box(&response), &cfg.advanced.nms, cfg.polarity);
        let e_ms = t.elapsed().as_secs_f64() * 1e3;

        // Stage 4: score EVERY proposal — mirrors pipeline.rs:279-281.
        let t = Instant::now();
        let mut scored = Vec::with_capacity(proposals.len());
        for proposal in &proposals {
            let circle = Circle::new(proposal.seed.position, cfg.radius_hint);
            let breakdown = score_circle_support_detailed(
                black_box(&gradient),
                black_box(&circle),
                &cfg.advanced.scoring,
            );
            scored.push((circle, breakdown));
        }
        let s_ms = t.elapsed().as_secs_f64() * 1e3;

        // Stage 5: refine only accepted proposals (non-degenerate AND
        // >= min_score) — mirrors pipeline.rs:283-300.
        let t = Instant::now();
        for (circle, breakdown) in &scored {
            if breakdown.is_degenerate || breakdown.total < cfg.min_score {
                continue;
            }
            let _ = black_box(refine_circle(
                black_box(&gradient),
                black_box(circle),
                &cfg.advanced.refinement,
            ));
        }
        let r_ms = t.elapsed().as_secs_f64() * 1e3;

        black_box(&response);
        if i >= warmup {
            grad.push(g_ms);
            vote.push(v_ms);
            extract.push(e_ms);
            score.push(s_ms);
            refine.push(r_ms);
        }
    }

    let detections = detect_circles(&view, cfg).unwrap();
    let end_to_end_ms = time_call_ms(reps, warmup, || detect_circles(&view, cfg).unwrap());

    StageTimes {
        gradient_ms: r3(median(grad)),
        voting_ms: r3(median(vote)),
        extract_ms: r3(median(extract)),
        score_ms: r3(median(score)),
        refine_ms: r3(median(refine)),
        detections,
        end_to_end_ms: r3(end_to_end_ms),
    }
}

// ---------------------------------------------------------------------------
// JSON schema (consumed by .github/pages/performance/index.html).
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Meta {
    cpu: String,
    rustc: String,
    git_sha: String,
    git_dirty: bool,
    generated: String,
    repeats: usize,
    warmup: usize,
    threads: String,
    source: String,
}

#[derive(Serialize)]
struct ImageRow {
    label: String,
    kind: String,
    file: String,
    img: Option<String>,
    width: usize,
    height: usize,
    polarity: String,
    radii: Vec<u32>,
    detections: usize,
    gradient_ms: f64,
    voting_ms: f64,
    extract_ms: f64,
    score_ms: f64,
    refine_ms: f64,
    end_to_end_ms: f64,
    note: Option<String>,
}

#[derive(Serialize)]
struct ThroughputRow {
    algo: String,
    variant: String,
    size: usize,
    ms: f64,
    melems_per_s: f64,
}

#[derive(Serialize)]
struct Throughput {
    desc: String,
    rows: Vec<ThroughputRow>,
}

#[derive(Serialize)]
struct RefineCall {
    name: String,
    us: f64,
    note: String,
}

#[derive(Serialize)]
struct Refinement {
    desc: String,
    calls: Vec<RefineCall>,
}

#[derive(Serialize)]
struct Frame {
    file: String,
    note: Option<String>,
    ms: f64,
}

#[derive(Serialize)]
struct EndToEnd {
    desc: String,
    frames: Vec<Frame>,
}

#[derive(Serialize)]
struct Root {
    meta: Meta,
    images: Vec<ImageRow>,
    throughput: Throughput,
    refinement: Refinement,
    end_to_end: EndToEnd,
}

// ---------------------------------------------------------------------------
// Meta capture (no extra dependencies — shells out to git/rustc, reads sysctl).
// ---------------------------------------------------------------------------

fn cmd(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn cpu_model() -> String {
    if cfg!(target_os = "macos") {
        cmd("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
    }
    .unwrap_or_else(|| "unknown CPU".to_string())
}

fn meta() -> Meta {
    let generated = cmd("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{}", d.as_secs()))
            .unwrap_or_default()
    });
    Meta {
        cpu: cpu_model(),
        rustc: cmd("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_string()),
        git_sha: cmd("git", &["rev-parse", "--short", "HEAD"])
            .unwrap_or_else(|| "unknown".to_string()),
        git_dirty: cmd("git", &["status", "--porcelain"])
            .map(|s| !s.is_empty())
            .unwrap_or(false),
        generated,
        repeats: REPEATS,
        warmup: 5,
        threads: "single".to_string(),
        source: "crates/radsym/examples/perf_export.rs".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Per-case detection configs (kept consistent with examples/configs/*).
// ---------------------------------------------------------------------------

fn synthetic_config() -> DetectCirclesConfig {
    // Single centered radius-12 disk. Three fixes make the overlay land exactly
    // on the disk (verified against ground truth: center (128,128), radius 12):
    //   * radii bracket the true radius tightly (the wide [8..16] band let
    //     smoothing bias the winning FRST radius down to ~8);
    //   * low `smoothing_factor` so per-radius blur (σ = factor·r) does not
    //     penalise the larger, correct radius and shrink the detection;
    //   * `patch_radius = 20 > disk radius` so radial-center refinement locks
    //     onto the true center (the default patch=12 drifts to ~(126,126) and
    //     reports OutOfBounds).
    // `max_detections = 1` keeps only the single real disk.
    let mut cfg = DetectCirclesConfig::for_radii([10, 11, 12, 13, 14])
        .polarity(Polarity::Bright)
        .radius_hint(12.0)
        .min_score(0.15);
    cfg.advanced.frst.gradient_threshold = 1.0;
    cfg.advanced.frst.smoothing_factor = 0.15;
    cfg.advanced.nms.threshold = 0.01;
    cfg.advanced.nms.max_detections = 1;
    cfg.advanced.refinement.advanced.radial_center.patch_radius = 20;
    cfg
}

/// Ring-grid outer mean radius (px) and inner/outer radius ratio, measured from
/// the labelled features in `testdata/ringgrid_features.json`. Used as scale
/// priors for the RSD + ellipse detection (see [`ringgrid_detect`]).
const RINGGRID_OUTER_HINT: f32 = 22.4;
const RINGGRID_INNER_RATIO: f32 = 0.47;

// ---------------------------------------------------------------------------
// Sections.
// ---------------------------------------------------------------------------

/// How a case is timed and detected.
///
/// Most cases run the plain `detect_circles` pipeline. The surface-hole case
/// runs the pyramid + ellipse pipeline (`surf_hole_synthetic::detect_case_image`)
/// instead, because the feature is a large tilted ellipse detected on a
/// downsampled working image.
enum Timing {
    /// Time `detect_circles` stage-by-stage with this config; overlay circles.
    /// Boxed: `DetectCirclesConfig` is far larger than the other variants.
    Detect(Box<DetectCirclesConfig>),
    /// Multi-ellipse surface-hole pipeline (pyramid + ellipse) at this level;
    /// overlay every detected ellipse.
    Surf { level: u8 },
    /// Ring-grid RSD + ellipse pipeline; overlay each ring's outer and inner
    /// ellipse.
    Ringgrid {
        outer_hint: f32,
        inner_ratio: f32,
        max_rings: usize,
    },
}

struct Case {
    label: String,
    kind: String,
    file: String,
    slug: &'static str,
    /// Whether to publish a preview PNG for this case. `false` for images
    /// whose source pixels are private and must not be uploaded to the
    /// public GitHub Pages site (e.g. `data/surf1.png`).
    preview: bool,
    image: OwnedImage<u8>,
    timing: Timing,
    note: Option<String>,
}

/// Max preview width (px); wider images are downsampled to keep the
/// committed PNGs and page weight small.
const MAX_PREVIEW_WIDTH: usize = 1024;

/// Overlay color for detected circles — matches the page's `--rose` CSS
/// variable (already used for the "refine" stage-bar segment).
const OVERLAY_COLOR: [u8; 4] = [255, 107, 107, 255];

/// Secondary overlay color (light blue) for inner ellipses, so a ring's outer
/// (rose) and inner (blue) contours are distinguishable.
const INNER_COLOR: [u8; 4] = [120, 200, 255, 255];

/// Nearest-neighbor downsample to at most `max_w` px wide, preserving aspect
/// ratio. No-op (scale 1.0) if already narrower. Returns (data, w, h, scale).
fn downsample_gray(data: &[u8], w: usize, h: usize, max_w: usize) -> (Vec<u8>, usize, usize, f32) {
    if w <= max_w {
        return (data.to_vec(), w, h, 1.0);
    }
    let scale = max_w as f32 / w as f32;
    let new_w = max_w;
    let new_h = ((h as f32) * scale).round().max(1.0) as usize;
    let mut out = vec![0u8; new_w * new_h];
    for y in 0..new_h {
        let sy = (((y as f32 + 0.5) / scale) as usize).min(h - 1);
        for x in 0..new_w {
            let sx = (((x as f32 + 0.5) / scale) as usize).min(w - 1);
            out[y * new_w + x] = data[sy * w + sx];
        }
    }
    (out, new_w, new_h, scale)
}

/// Render a grayscale-with-detection-circles preview PNG (downsampled if
/// wide) and save it under `img_dir`. Returns the JSON-relative path.
fn save_preview(
    image: &OwnedImage<u8>,
    detections: &[CircleDetection],
    img_dir: &Path,
    slug: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let (w0, h0) = (image.width(), image.height());
    let (gray, w, h, scale) = downsample_gray(image.data(), w0, h0, MAX_PREVIEW_WIDTH);

    let mut preview = DiagnosticImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = gray[y * w + x];
            preview.set_pixel(x, y, [v, v, v, 255]);
        }
    }

    for det in detections {
        let c = &det.hypothesis;
        let scaled = Circle::new(
            PixelCoord::new(c.center.x * scale, c.center.y * scale),
            c.radius * scale,
        );
        overlay_circle(&mut preview, &scaled, OVERLAY_COLOR);
    }

    std::fs::create_dir_all(img_dir)?;
    let path = img_dir.join(format!("{slug}.png"));
    save_diagnostic(&preview, &path)?;
    Ok(format!("img/{slug}.png"))
}

/// Render a grayscale preview PNG (downsampled if wide) with a list of coloured
/// ellipses drawn on top, and save it under `img_dir`. Each ellipse is scaled by
/// the same preview downscale factor as the pixels, then drawn with its colour.
fn save_preview_ellipses(
    image: &OwnedImage<u8>,
    ellipses: &[(Ellipse, [u8; 4])],
    img_dir: &Path,
    slug: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let (w0, h0) = (image.width(), image.height());
    let (gray, w, h, scale) = downsample_gray(image.data(), w0, h0, MAX_PREVIEW_WIDTH);

    let mut preview = DiagnosticImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = gray[y * w + x];
            preview.set_pixel(x, y, [v, v, v, 255]);
        }
    }

    for (ellipse, color) in ellipses {
        let scaled = Ellipse::new(
            PixelCoord::new(ellipse.center.x * scale, ellipse.center.y * scale),
            ellipse.semi_major * scale,
            ellipse.semi_minor * scale,
            ellipse.angle,
        );
        overlay_ellipse(&mut preview, &scaled, *color);
    }

    std::fs::create_dir_all(img_dir)?;
    let path = img_dir.join(format!("{slug}.png"));
    save_diagnostic(&preview, &path)?;
    Ok(format!("img/{slug}.png"))
}

/// Stage timings plus the coloured ellipses to overlay, for the ellipse-based
/// cases (surface hole, ring grid). Mirrors [`StageTimes`] but carries ellipses.
struct EllipseTiming {
    gradient_ms: f64,
    voting_ms: f64,
    extract_ms: f64,
    score_ms: f64,
    refine_ms: f64,
    end_to_end_ms: f64,
    ellipses: Vec<(Ellipse, [u8; 4])>,
    detections: usize,
    radii: Vec<u32>,
    polarity: String,
}

/// Median of `reps` samples of each stage `Duration`, in milliseconds.
fn median_stage_ms(samples: &[std::time::Duration]) -> f64 {
    r3(median(
        samples.iter().map(|d| d.as_secs_f64() * 1e3).collect(),
    ))
}

/// Time the multi-ellipse surface-hole pipeline and collect every detected
/// ellipse (target hole + distractors) for the overlay. The one-time pyramid
/// downsample folds into the `gradient` stage.
fn time_surf(
    image: &OwnedImage<u8>,
    level: u8,
) -> Result<EllipseTiming, Box<dyn std::error::Error>> {
    let (reps, warmup) = budget(image.width() * image.height());
    let (mut grad, mut vote, mut extract, mut score, mut refine) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..(reps + warmup) {
        let d = surf_hole_synthetic::time_multi_once(image, level)?;
        if i >= warmup {
            grad.push(d.gradient);
            vote.push(d.voting);
            extract.push(d.extract);
            score.push(d.score);
            refine.push(d.refine);
        }
    }

    let detection = surf_hole_synthetic::detect_all_ellipses(image, level)?;
    let end_to_end_ms = time_call_ms(reps, warmup, || {
        surf_hole_synthetic::detect_all_ellipses(image, level).unwrap()
    });
    let ellipses = detection
        .ellipses
        .iter()
        .map(|e| (e.image_ellipse, OVERLAY_COLOR))
        .collect();

    Ok(EllipseTiming {
        gradient_ms: median_stage_ms(&grad),
        voting_ms: median_stage_ms(&vote),
        extract_ms: median_stage_ms(&extract),
        score_ms: median_stage_ms(&score),
        refine_ms: median_stage_ms(&refine),
        end_to_end_ms: r3(end_to_end_ms),
        detections: detection.ellipses.len(),
        radii: detection.radii,
        polarity: "Bright".to_string(),
        ellipses,
    })
}

/// Time the ring-grid outer+inner ellipse pipeline and collect the outer (rose)
/// and inner (blue) ellipses for the overlay.
fn time_ringgrid(
    image: &OwnedImage<u8>,
    outer_hint: f32,
    inner_ratio: f32,
    max_rings: usize,
) -> Result<EllipseTiming, Box<dyn std::error::Error>> {
    let (reps, warmup) = budget(image.width() * image.height());
    let (mut grad, mut vote, mut extract, mut score, mut refine) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..(reps + warmup) {
        let d = ringgrid_detect::time_rings_once(image, outer_hint, inner_ratio, max_rings);
        if i >= warmup {
            grad.push(d.gradient);
            vote.push(d.voting);
            extract.push(d.extract);
            score.push(d.score);
            refine.push(d.refine);
        }
    }

    let rings = ringgrid_detect::detect_rings_in_image(image, outer_hint, inner_ratio, max_rings);
    let end_to_end_ms = time_call_ms(reps, warmup, || {
        ringgrid_detect::detect_rings_in_image(image, outer_hint, inner_ratio, max_rings)
    });
    let mut ellipses = Vec::with_capacity(rings.len() * 2);
    for ring in &rings {
        ellipses.push((ring.outer, OVERLAY_COLOR));
        if let Some(inner) = ring.inner {
            ellipses.push((inner, INNER_COLOR));
        }
    }

    Ok(EllipseTiming {
        gradient_ms: median_stage_ms(&grad),
        voting_ms: median_stage_ms(&vote),
        extract_ms: median_stage_ms(&extract),
        score_ms: median_stage_ms(&score),
        refine_ms: median_stage_ms(&refine),
        end_to_end_ms: r3(end_to_end_ms),
        detections: rings.len(),
        radii: ringgrid_detect::build_radius_band(outer_hint, 0.8, 1.16, 5),
        polarity: "Dark".to_string(),
        ellipses,
    })
}

/// Build an `ImageRow` from an ellipse-based timing result, drawing its ellipses.
fn ellipse_row(
    case: &Case,
    t: EllipseTiming,
    img_dir: Option<&Path>,
) -> Result<ImageRow, Box<dyn std::error::Error>> {
    let img = if case.preview {
        img_dir
            .map(|dir| save_preview_ellipses(&case.image, &t.ellipses, dir, case.slug))
            .transpose()?
    } else {
        None
    };
    Ok(ImageRow {
        label: case.label.clone(),
        kind: case.kind.clone(),
        file: case.file.clone(),
        img,
        width: case.image.width(),
        height: case.image.height(),
        polarity: t.polarity,
        radii: t.radii,
        detections: t.detections,
        gradient_ms: t.gradient_ms,
        voting_ms: t.voting_ms,
        extract_ms: t.extract_ms,
        score_ms: t.score_ms,
        refine_ms: t.refine_ms,
        end_to_end_ms: t.end_to_end_ms,
        note: case.note.clone(),
    })
}

fn image_rows(
    cases: &[Case],
    img_dir: Option<&Path>,
) -> Result<Vec<ImageRow>, Box<dyn std::error::Error>> {
    cases
        .iter()
        .map(|case| {
            eprintln!(
                "  stage timing: {} ({}x{})",
                case.label,
                case.image.width(),
                case.image.height()
            );
            match &case.timing {
                Timing::Detect(cfg) => {
                    let t = time_stages(&case.image, cfg);
                    let img = if case.preview {
                        img_dir
                            .map(|dir| save_preview(&case.image, &t.detections, dir, case.slug))
                            .transpose()?
                    } else {
                        None
                    };
                    Ok(ImageRow {
                        label: case.label.clone(),
                        kind: case.kind.clone(),
                        file: case.file.clone(),
                        img,
                        width: case.image.width(),
                        height: case.image.height(),
                        polarity: format!("{:?}", cfg.polarity),
                        radii: cfg.radii.clone(),
                        detections: t.detections.len(),
                        gradient_ms: t.gradient_ms,
                        voting_ms: t.voting_ms,
                        extract_ms: t.extract_ms,
                        score_ms: t.score_ms,
                        refine_ms: t.refine_ms,
                        end_to_end_ms: t.end_to_end_ms,
                        note: case.note.clone(),
                    })
                }
                Timing::Surf { level } => {
                    ellipse_row(case, time_surf(&case.image, *level)?, img_dir)
                }
                Timing::Ringgrid {
                    outer_hint,
                    inner_ratio,
                    max_rings,
                } => ellipse_row(
                    case,
                    time_ringgrid(&case.image, *outer_hint, *inner_ratio, *max_rings)?,
                    img_dir,
                ),
            }
        })
        .collect()
}

fn throughput_rows() -> Throughput {
    let mut frst_cfg = FrstConfig::default();
    frst_cfg.radii = vec![5, 7, 9, 11, 13];
    let mut rsd_cfg = RsdConfig::default();
    rsd_cfg.radii = vec![5, 7, 9, 11, 13];

    let mut rows = Vec::new();
    for &size in &[256usize, 512, 1024] {
        let image = make_disk_image(size);
        let gradient = sobel_gradient(&image.view()).unwrap();
        let (reps, warmup) = budget(size * size);
        let elems = (size * size) as f64;

        let mut push = |algo: &str, variant: &str, ms: f64| {
            let secs = ms / 1e3;
            rows.push(ThroughputRow {
                algo: algo.to_string(),
                variant: variant.to_string(),
                size,
                ms: r3(ms),
                melems_per_s: r1(elems / secs / 1e6),
            });
        };

        // black_box the inputs so the optimizer cannot prove the calls are
        // loop-invariant and hoist them out of the timing loop.
        eprintln!("  throughput: size {size}");
        push(
            "frst",
            "unfused",
            time_call_ms(reps, warmup, || {
                frst_response(black_box(&gradient), black_box(&frst_cfg)).unwrap()
            }),
        );
        push(
            "frst",
            "fused",
            time_call_ms(reps, warmup, || {
                frst_response_fused(black_box(&gradient), black_box(&frst_cfg)).unwrap()
            }),
        );
        push(
            "rsd",
            "unfused",
            time_call_ms(reps, warmup, || {
                rsd_response(black_box(&gradient), black_box(&rsd_cfg)).unwrap()
            }),
        );
        push(
            "rsd",
            "fused",
            time_call_ms(reps, warmup, || {
                rsd_response_fused(black_box(&gradient), black_box(&rsd_cfg)).unwrap()
            }),
        );
    }

    Throughput {
        desc: "Voting-backend throughput on synthetic disks (radii [5,7,9,11,13], \
               gradient precomputed, single-thread). FRST keeps the orientation+alpha \
               term; RSD and the fused variants drop work for speed. detect_circles uses \
               FRST-unfused."
            .to_string(),
        rows,
    }
}

fn refinement_rows() -> Refinement {
    let reps = 200;
    let warmup = 20;

    // Small disk (radius 10) so its ring sits inside the radial-center patch
    // (radius 12) and center refinement engages gradient instead of fast-failing
    // in the flat interior of a large disk. Ellipse/homography use the larger
    // edge-search targets from benches/refinement_bench.rs.
    let disk = make_fixed_disk(256, 10.0);
    let disk_grad = sobel_gradient(&disk.view()).unwrap();
    let seed = PixelCoord::new(129.0, 127.0);
    let circle = Circle::new(seed, 10.0);
    let rc_cfg = RadialCenterConfig::default();
    let circle_cfg = radsym::CircleRefineConfig::default();

    let ellipse_img = make_ellipse_image(256);
    let ellipse_grad = sobel_gradient(&ellipse_img.view()).unwrap();
    let ellipse = Ellipse::new(PixelCoord::new(131.0, 126.0), 256.0 / 6.5, 256.0 / 6.5, 0.0);
    let ellipse_cfg = EllipseRefineConfig::default();

    let homography = Homography::new([
        [1.12, 0.05, 18.0],
        [0.03, 1.0, 12.0],
        [0.0008, -0.0005, 1.0],
    ])
    .unwrap();
    let rect_circle = Circle::new(PixelCoord::new(128.0, 120.0), 34.0);
    let proj = make_projective_disk_image(256, &homography, rect_circle);
    let proj_grad = sobel_gradient(&proj.view()).unwrap();
    let proj_ellipse = rectified_circle_to_image_ellipse(&homography, &rect_circle).unwrap();
    let initial = Ellipse::new(
        PixelCoord::new(proj_ellipse.center.x + 3.0, proj_ellipse.center.y - 2.0),
        proj_ellipse.semi_major * 0.95,
        proj_ellipse.semi_minor * 1.05,
        proj_ellipse.angle + 0.05,
    );
    let homog_cfg = HomographyEllipseRefineConfig::default();

    eprintln!("  refinement micro-costs");
    let calls = vec![
        RefineCall {
            name: "radial_center".to_string(),
            us: r2(1e3
                * time_call_ms(reps, warmup, || {
                    radial_center_refine_from_gradient(
                        black_box(&disk_grad),
                        black_box(seed),
                        black_box(&rc_cfg),
                    )
                })),
            note: "single-pass Parthasarathy weighted least squares".to_string(),
        },
        RefineCall {
            name: "circle".to_string(),
            us: r2(1e3
                * time_call_ms(reps, warmup, || {
                    refine_circle(
                        black_box(&disk_grad),
                        black_box(&circle),
                        black_box(&circle_cfg),
                    )
                    .unwrap()
                })),
            note: "iterative: radial-center + annulus radius re-estimate".to_string(),
        },
        RefineCall {
            name: "ellipse".to_string(),
            us: r2(1e3
                * time_call_ms(reps, warmup, || {
                    refine_ellipse(
                        black_box(&ellipse_grad),
                        black_box(&ellipse),
                        black_box(&ellipse_cfg),
                    )
                    .unwrap()
                })),
            note: "iterative edge search + guarded Gauss-Newton".to_string(),
        },
        RefineCall {
            name: "homography".to_string(),
            us: r2(1e3
                * time_call_ms(reps, warmup, || {
                    refine_ellipse_homography(
                        black_box(&proj_grad),
                        black_box(&initial),
                        black_box(&homography),
                        black_box(&homog_cfg),
                    )
                })),
            note: "homography-aware ellipse refine in rectified space".to_string(),
        },
    ];

    Refinement {
        desc: "Per-call refinement cost (p50, microseconds). Each consumes a precomputed \
               gradient field; radial_center and circle run on a radius-10 disk, ellipse \
               and homography on larger edge-search targets."
            .to_string(),
        calls,
    }
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Force deterministic single-threaded timing *if* this example is built with
    // the `rayon` feature (`--features rayon`), in which case FRST parallelizes
    // multi-radius voting over the global rayon pool. With the default feature
    // set this is a harmless no-op: radsym's `rayon` feature is `dep:rayon`-gated
    // and is NOT pulled in by the `criterion` dev-dependency, so the sequential
    // `.iter()` path is what runs and what the page documents.
    // SAFETY: set at the very top of main, before any work or threads start.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", "1") };

    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = format!("{manifest}/../..");
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{root}/.github/pages/performance/data.json"));

    eprintln!("perf_export: timing stages (this takes a few minutes)...");

    let big_note = |reps: usize| Some(format!("Large image: {reps} timed reps (see methodology)."));

    let surf_case = &surf_hole_synthetic::CASES[0];
    let surf_image = surf_hole_synthetic::render_case(surf_case);

    let cases = vec![
        Case {
            label: "Synthetic disk 256".to_string(),
            kind: "synthetic disk".to_string(),
            file: "synthetic:disk-256".to_string(),
            slug: "disk-256",
            preview: true,
            image: make_fixed_disk(256, 12.0),
            timing: Timing::Detect(Box::new(synthetic_config())),
            note: Some("Radii bracket the radius-12 disk, Sobel + FRST. Voting dominates; the gradient is computed once and reused by scoring and refinement.".to_string()),
        },
        Case {
            label: "Synthetic disk 512".to_string(),
            kind: "synthetic disk".to_string(),
            file: "synthetic:disk-512".to_string(),
            slug: "disk-512",
            preview: true,
            image: make_fixed_disk(512, 12.0),
            timing: Timing::Detect(Box::new(synthetic_config())),
            note: Some("Same radii as the 256 case: cost grows with the response-map area, not the radii.".to_string()),
        },
        Case {
            label: "Synthetic disk 1024".to_string(),
            kind: "synthetic disk".to_string(),
            file: "synthetic:disk-1024".to_string(),
            slug: "disk-1024",
            preview: true,
            image: make_fixed_disk(1024, 12.0),
            timing: Timing::Detect(Box::new(synthetic_config())),
            note: big_note(budget(1024 * 1024).0),
        },
        Case {
            label: "Ring grid".to_string(),
            kind: "ring grid".to_string(),
            file: "testdata/ringgrid.png".to_string(),
            slug: "ringgrid",
            preview: true,
            image: load_grayscale(format!("{root}/testdata/ringgrid.png"))?,
            timing: Timing::Ringgrid {
                outer_hint: RINGGRID_OUTER_HINT,
                inner_ratio: RINGGRID_INNER_RATIO,
                max_rings: 130,
            },
            note: Some("Each ring's outer (rose) and inner (blue) ellipse via RSD centre voting + local ellipse refinement. Recovers all 63 labelled outer centres to sub-pixel accuracy (mean centre error ~0.5px, axis error ~0.4px).".to_string()),
        },
        Case {
            label: "Surface hole (synthetic)".to_string(),
            kind: "surface hole".to_string(),
            file: "synthetic:surf-hole".to_string(),
            slug: "surf-hole",
            preview: true,
            image: surf_image,
            timing: Timing::Surf {
                level: surf_hole_synthetic::DEFAULT_PYRAMID_LEVEL,
            },
            note: Some(format!(
                "Generated stand-in for the private surf dataset (its pixels are \
                 not published). All three elliptical features (the large tilted \
                 hole plus two smaller distractors) detected via the pyramid \
                 (level {}) + multi-scale ellipse pipeline; correctness is \
                 CI-asserted by tests/perf_export_surf.rs. The one-time downsample \
                 is folded into the gradient bar, so detection runs on the small \
                 working image. {}",
                surf_hole_synthetic::DEFAULT_PYRAMID_LEVEL,
                big_note(budget(2048 * 1536).0).unwrap_or_default(),
            )),
        },
    ];

    let img_dir = (out != "-").then(|| Path::new(&out).with_file_name("img"));
    let images = image_rows(&cases, img_dir.as_deref())?;
    let throughput = throughput_rows();
    let refinement = refinement_rows();

    let end_to_end = EndToEnd {
        desc: "Whole detect_circles() per image (p50). The sum of an image's stage bars \
               equals this within a few percent (the residual is diagnostics bookkeeping)."
            .to_string(),
        frames: images
            .iter()
            .map(|row| Frame {
                file: row.file.clone(),
                note: Some(format!("{}, radii {:?}", row.polarity, row.radii)),
                ms: row.end_to_end_ms,
            })
            .collect(),
    };

    let root_doc = Root {
        meta: meta(),
        images,
        throughput,
        refinement,
        end_to_end,
    };

    let json = serde_json::to_string_pretty(&root_doc)?;
    if out == "-" {
        println!("{json}");
    } else {
        std::fs::write(&out, format!("{json}\n"))?;
        eprintln!("perf_export: wrote {out}");
    }
    Ok(())
}
