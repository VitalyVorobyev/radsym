use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use radsym::core::gradient::{GradientField, sobel_gradient, thin_gradient};
use radsym::core::image_view::OwnedImage;
use radsym::propose::rsd::{RsdConfig, rsd_response, rsd_response_fused};

// Realistic radii band (also used to derive median radius for smoothing).
const RADII: [u32; 5] = [8, 10, 12, 14, 16];

fn median_radius(radii: &[u32]) -> f32 {
    let mut v = radii.to_vec();
    v.sort_unstable();
    v[v.len() / 2] as f32
}

fn make_disk_image(size: usize) -> OwnedImage<u8> {
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let r = size as f32 / 6.0;
    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= r {
                data[y * size + x] = 255;
            }
        }
    }
    OwnedImage::from_vec(data, size, size).unwrap()
}

/// Separable 3x3 box blur, `passes` times, to widen sharp edges into the
/// multi-pixel bands that real (blurred) imagery produces — the condition under
/// which edge-thinning has anything to remove.
fn box_blur_u8(data: &mut [u8], w: usize, h: usize, passes: usize) {
    for _ in 0..passes {
        let mut tmp = vec![0u8; w * h];
        // horizontal
        for y in 0..h {
            for x in 0..w {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                let s = data[y * w + x0] as u32 + data[y * w + x] as u32 + data[y * w + x1] as u32;
                tmp[y * w + x] = (s / 3) as u8;
            }
        }
        // vertical
        for y in 0..h {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            for x in 0..w {
                let s = tmp[y0 * w + x] as u32 + tmp[y * w + x] as u32 + tmp[y1 * w + x] as u32;
                data[y * w + x] = (s / 3) as u8;
            }
        }
    }
}

/// A grid of bright ring markers (annuli) on a dark background, blurred so edges
/// span 3–5 px — a stand-in for ringgrid's real detection input.
fn make_ring_grid(size: usize, blur_passes: usize) -> OwnedImage<u8> {
    let mut data = vec![0u8; size * size];
    let spacing = size / 8; // ~7x7 markers
    let outer = 14.0f32;
    let inner = 10.0f32;
    let mut cy = spacing / 2;
    while cy < size {
        let mut cx = spacing / 2;
        while cx < size {
            let y0 = cy.saturating_sub(outer as usize + 1);
            let y1 = (cy + outer as usize + 1).min(size);
            let x0 = cx.saturating_sub(outer as usize + 1);
            let x1 = (cx + outer as usize + 1).min(size);
            for y in y0..y1 {
                for x in x0..x1 {
                    let dx = x as f32 - cx as f32;
                    let dy = y as f32 - cy as f32;
                    let d = (dx * dx + dy * dy).sqrt();
                    if d >= inner && d <= outer {
                        data[y * size + x] = 255;
                    }
                }
            }
            cx += spacing;
        }
        cy += spacing;
    }
    box_blur_u8(&mut data, size, size, blur_passes);
    OwnedImage::from_vec(data, size, size).unwrap()
}

fn strong_edge_count(field: &GradientField, threshold: f32) -> usize {
    let t2 = threshold * threshold;
    let gx = field.gx();
    let gy = field.gy();
    let w = field.width();
    let h = field.height();
    let mut n = 0;
    for y in 0..h {
        for x in 0..w {
            let a = *gx.get(x, y).unwrap();
            let b = *gy.get(x, y).unwrap();
            if a * a + b * b >= t2 {
                n += 1;
            }
        }
    }
    n
}

fn max_magnitude(field: &GradientField) -> f32 {
    let mut m = 0.0f32;
    for y in 0..field.height() {
        for x in 0..field.width() {
            m = m.max(field.magnitude(x, y).unwrap());
        }
    }
    m
}

/// Print fixture statistics once (strong-edge reduction is the *mechanism*
/// behind any speedup, measured directly rather than assumed).
fn report_fixture_stats() {
    eprintln!("\n=== FIXTURE STATS (ring-grid, 2 blur passes) ===");
    eprintln!(
        "{:<8} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "size", "max|g|", "edges@0", "edges@thr", "thin@thr", "reduction"
    );
    for &size in &[512usize, 1024] {
        let img = make_ring_grid(size, 2);
        let grad = sobel_gradient(&img.view()).unwrap();
        let thin = thin_gradient(&grad).unwrap();
        let maxm = max_magnitude(&grad);
        let thr = 0.15 * maxm; // "strong edge" = 15% of peak magnitude
        let e0 = strong_edge_count(&grad, 0.0);
        let e_thr = strong_edge_count(&grad, thr);
        let t_thr = strong_edge_count(&thin, thr);
        let reduction = 100.0 * (1.0 - t_thr as f32 / e_thr.max(1) as f32);
        eprintln!("{size:<8} {maxm:>10.1} {e0:>12} {e_thr:>12} {t_thr:>12} {reduction:>9.1}%");
    }
    eprintln!("(threshold used in benches = 0.15 * max|g|)\n");
}

fn bench_thin_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("thin_gradient");
    for &size in &[256usize, 512, 1024] {
        let img = make_ring_grid(size, 2);
        let grad = sobel_gradient(&img.view()).unwrap();
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &grad, |b, g| {
            b.iter(|| thin_gradient(g).unwrap());
        });
    }
    group.finish();
}

/// The headline comparison: baseline voting vs (thin + voting), swept over
/// gradient_threshold ∈ {0, positive}. At threshold 0 thinning cannot skip the
/// zeroed pixels, so ~no speedup; the win appears only with a positive
/// threshold. Thinning cost is folded into the "thinned" closure.
fn bench_rsd_thinning_effect(c: &mut Criterion) {
    report_fixture_stats();

    let mut group = c.benchmark_group("rsd_thinning_effect");
    for &size in &[512usize, 1024] {
        let img = make_ring_grid(size, 2);
        let grad = sobel_gradient(&img.view()).unwrap();
        let thr_pos = 0.15 * max_magnitude(&grad);

        for &(label, thr) in &[("thr0", 0.0f32), ("thrPos", thr_pos)] {
            let mut config = RsdConfig::default();
            config.radii = RADII.to_vec();
            config.gradient_threshold = thr;

            group.throughput(Throughput::Elements((size * size) as u64));

            group.bench_function(BenchmarkId::new(format!("baseline/{label}"), size), |b| {
                b.iter(|| rsd_response_fused(&grad, &config).unwrap());
            });
            group.bench_function(BenchmarkId::new(format!("thinned/{label}"), size), |b| {
                b.iter(|| {
                    let t = thin_gradient(&grad).unwrap();
                    rsd_response_fused(&t, &config).unwrap()
                });
            });
        }
    }
    group.finish();
}

/// Per-stage attribution using only the public API. Voting-only is obtained by
/// choosing smoothing_factor so sigma ≤ 0.5 (blur skipped); the blur cost is the
/// delta against the default smoothing_factor.
fn bench_rsd_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("rsd_stages");
    let med = median_radius(&RADII);
    let sf_novote = 0.4 / med; // sigma = 0.4 → blur skipped

    for &size in &[512usize, 1024] {
        let img = make_ring_grid(size, 2);
        let grad = sobel_gradient(&img.view()).unwrap();
        let thr_pos = 0.15 * max_magnitude(&grad);
        group.throughput(Throughput::Elements((size * size) as u64));

        // Stage 1: gradient computation.
        group.bench_with_input(BenchmarkId::new("gradient", size), &img, |b, im| {
            b.iter(|| sobel_gradient(&im.view()).unwrap());
        });
        // Stage 2: thinning.
        group.bench_with_input(BenchmarkId::new("thin", size), &grad, |b, g| {
            b.iter(|| thin_gradient(g).unwrap());
        });
        // Stage 3: voting only (blur skipped), positive threshold.
        let mut cfg_novote = RsdConfig::default();
        cfg_novote.radii = RADII.to_vec();
        cfg_novote.gradient_threshold = thr_pos;
        cfg_novote.smoothing_factor = sf_novote;
        group.bench_with_input(BenchmarkId::new("vote_only", size), &grad, |b, g| {
            b.iter(|| rsd_response_fused(g, &cfg_novote).unwrap());
        });
        // Stage 3+4: voting + blur (default smoothing).
        let mut cfg_blur = RsdConfig::default();
        cfg_blur.radii = RADII.to_vec();
        cfg_blur.gradient_threshold = thr_pos;
        group.bench_with_input(BenchmarkId::new("vote_blur", size), &grad, |b, g| {
            b.iter(|| rsd_response_fused(g, &cfg_blur).unwrap());
        });
    }
    group.finish();
}

/// Isolate thinning's effect on the *voting* stage alone (blur skipped, positive
/// threshold), so the voting-portion speedup isn't masked by the blur.
fn bench_vote_isolated(c: &mut Criterion) {
    let mut group = c.benchmark_group("rsd_vote_isolated");
    let med = median_radius(&RADII);
    let sf_novote = 0.4 / med; // sigma = 0.4 → blur skipped

    for &size in &[512usize, 1024] {
        let img = make_ring_grid(size, 2);
        let grad = sobel_gradient(&img.view()).unwrap();
        let thr_pos = 0.15 * max_magnitude(&grad);

        let mut cfg = RsdConfig::default();
        cfg.radii = RADII.to_vec();
        cfg.gradient_threshold = thr_pos;
        cfg.smoothing_factor = sf_novote;

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::new("baseline", size), &grad, |b, g| {
            b.iter(|| rsd_response_fused(g, &cfg).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("thinned", size), &grad, |b, g| {
            b.iter(|| {
                let t = thin_gradient(g).unwrap();
                rsd_response_fused(&t, &cfg).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_rsd(c: &mut Criterion) {
    let mut config = RsdConfig::default();
    config.radii = RADII.to_vec();

    let mut group = c.benchmark_group("rsd");
    for &size in &[256, 512, 1024] {
        let image = make_disk_image(size);
        let gradient = sobel_gradient(&image.view()).unwrap();
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &gradient, |b, grad| {
            b.iter(|| rsd_response(grad, &config).unwrap());
        });
    }
    group.finish();
}

fn bench_rsd_fused(c: &mut Criterion) {
    let mut config = RsdConfig::default();
    config.radii = RADII.to_vec();

    let mut group = c.benchmark_group("rsd_fused");
    for &size in &[256, 512, 1024] {
        let image = make_disk_image(size);
        let gradient = sobel_gradient(&image.view()).unwrap();
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &gradient, |b, grad| {
            b.iter(|| rsd_response_fused(grad, &config).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_thin_gradient,
    bench_rsd_thinning_effect,
    bench_rsd_stages,
    bench_vote_isolated,
    bench_rsd,
    bench_rsd_fused
);
criterion_main!(benches);
