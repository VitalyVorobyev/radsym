use criterion::{criterion_group, criterion_main, Criterion};

use radsym::core::gradient::sobel_gradient;
use radsym::core::image_view::OwnedImage;
use radsym::support::score::score_circle_support;
use radsym::{Circle, PixelCoord, ScoringConfig};

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

fn bench_score_circle(c: &mut Criterion) {
    let size = 256;
    let image = make_disk_image(size);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let center = PixelCoord::new(size as f32 / 2.0, size as f32 / 2.0);
    let circle = Circle::new(center, size as f32 / 6.0);
    let config = ScoringConfig::default();

    c.bench_function("score_circle_256", |b| {
        b.iter(|| score_circle_support(&gradient, &circle, &config));
    });
}

criterion_group!(benches, bench_score_circle);
criterion_main!(benches);
