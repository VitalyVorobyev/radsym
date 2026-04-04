use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use radsym::core::gradient::sobel_gradient;
use radsym::core::image_view::OwnedImage;
use radsym::propose::rsd::{rsd_response, rsd_response_fused, RsdConfig};

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

fn bench_rsd(c: &mut Criterion) {
    let config = RsdConfig {
        radii: vec![5, 7, 9, 11, 13],
        ..RsdConfig::default()
    };

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
    let config = RsdConfig {
        radii: vec![5, 7, 9, 11, 13],
        ..RsdConfig::default()
    };

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

criterion_group!(benches, bench_rsd, bench_rsd_fused);
criterion_main!(benches);
