use criterion::{criterion_group, criterion_main, Criterion};

use radsym::core::gradient::sobel_gradient;
use radsym::core::image_view::OwnedImage;
use radsym::{
    frst_response_homography, radial_center_refine_from_gradient, refine_circle, refine_ellipse,
    refine_ellipse_homography, Circle, CircleRefineConfig, Ellipse, EllipseRefineConfig,
    FrstConfig, Homography, HomographyEllipseRefineConfig, PixelCoord, RadialCenterConfig,
    RectifiedGrid,
};

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

fn make_ellipse_image(size: usize) -> OwnedImage<u8> {
    let cx = size as f32 / 2.0;
    let cy = size as f32 / 2.0;
    let semi_major = size as f32 / 6.0;
    let semi_minor = size as f32 / 7.5;
    let angle = 0.35f32;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let lx = dx * cos_a + dy * sin_a;
            let ly = -dx * sin_a + dy * cos_a;
            if (lx / semi_major).powi(2) + (ly / semi_minor).powi(2) <= 1.0 {
                data[y * size + x] = 255;
            }
        }
    }
    OwnedImage::from_vec(data, size, size).unwrap()
}

fn make_concentric_distractor_ellipse_image(size: usize) -> OwnedImage<u8> {
    let center = PixelCoord::new(size as f32 / 2.0, size as f32 / 2.0 - 2.0);
    let outer_a = size as f32 / 6.0;
    let outer_b = size as f32 / 8.5;
    let inner_a = outer_a * 0.58;
    let inner_b = outer_b * 0.58;
    let angle = 0.38f32;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let mut data = vec![24.0f32; size * size];

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center.x;
            let dy = y as f32 - center.y;
            let lx = dx * cos_a + dy * sin_a;
            let ly = -dx * sin_a + dy * cos_a;
            let outer_r = (lx / outer_a).powi(2) + (ly / outer_b).powi(2);
            let inner_r = (lx / inner_a).powi(2) + (ly / inner_b).powi(2);
            let mut value = if outer_r <= 1.0 { 220.0 } else { 22.0 };
            if inner_r <= 1.0 {
                let theta = ly.atan2(lx);
                if (-2.25..=0.85).contains(&theta) {
                    value = 2.0;
                }
            }
            value += 8.0 * ((0.14 * x as f32).sin() + (0.09 * y as f32).cos());
            data[y * size + x] = value.clamp(0.0, 255.0);
        }
    }

    OwnedImage::from_vec(
        data.iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
        size,
        size,
    )
    .unwrap()
}

fn make_projective_disk_image(
    size: usize,
    homography: &Homography,
    circle: Circle,
) -> OwnedImage<u8> {
    let mut data = vec![20u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let image_point = PixelCoord::new(x as f32, y as f32);
            let Some(rectified) = homography.map_image_to_rectified(image_point) else {
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

fn bench_radial_center(c: &mut Criterion) {
    let size = 256;
    let image = make_disk_image(size);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let seed = PixelCoord::new(size as f32 / 2.0 + 2.0, size as f32 / 2.0 - 1.0);
    let config = RadialCenterConfig::default();

    c.bench_function("radial_center_256", |b| {
        b.iter(|| radial_center_refine_from_gradient(&gradient, seed, &config));
    });
}

fn bench_refine_circle(c: &mut Criterion) {
    let size = 256;
    let image = make_disk_image(size);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let center = PixelCoord::new(size as f32 / 2.0 + 2.0, size as f32 / 2.0 - 1.0);
    let circle = Circle::new(center, size as f32 / 6.0);
    let config = CircleRefineConfig::default();

    c.bench_function("refine_circle_256", |b| {
        b.iter(|| refine_circle(&gradient, &circle, &config));
    });
}

fn bench_refine_ellipse(c: &mut Criterion) {
    let size = 256;
    let image = make_ellipse_image(size);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let ellipse = Ellipse::new(
        PixelCoord::new(size as f32 / 2.0 + 3.0, size as f32 / 2.0 - 2.0),
        size as f32 / 6.5,
        size as f32 / 6.5,
        0.0,
    );
    let config = EllipseRefineConfig::default();

    c.bench_function("refine_ellipse_256", |b| {
        b.iter(|| refine_ellipse(&gradient, &ellipse, &config));
    });
}

fn bench_refine_ellipse_concentric_distractor(c: &mut Criterion) {
    let size = 256;
    let image = make_concentric_distractor_ellipse_image(size);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let ellipse = Ellipse::new(
        PixelCoord::new(size as f32 / 2.0 + 3.0, size as f32 / 2.0 - 3.0),
        size as f32 / 6.2,
        size as f32 / 6.2,
        0.0,
    );
    let config = EllipseRefineConfig::default();

    c.bench_function("refine_ellipse_256_concentric_distractor", |b| {
        b.iter(|| refine_ellipse(&gradient, &ellipse, &config));
    });
}

fn bench_refine_ellipse_homography(c: &mut Criterion) {
    let size = 256;
    let homography = Homography::new([
        [1.12, 0.05, 18.0],
        [0.03, 1.0, 12.0],
        [0.0008, -0.0005, 1.0],
    ])
    .unwrap();
    let rectified_circle = Circle::new(PixelCoord::new(128.0, 120.0), 34.0);
    let image = make_projective_disk_image(size, &homography, rectified_circle);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let ellipse =
        radsym::rectified_circle_to_image_ellipse(&homography, &rectified_circle).unwrap();
    let initial = Ellipse::new(
        PixelCoord::new(ellipse.center.x + 3.0, ellipse.center.y - 2.0),
        ellipse.semi_major * 0.95,
        ellipse.semi_minor * 1.05,
        ellipse.angle + 0.05,
    );
    let config = HomographyEllipseRefineConfig::default();

    c.bench_function("refine_ellipse_homography_256", |b| {
        b.iter(|| refine_ellipse_homography(&gradient, &initial, &homography, &config));
    });
}

fn bench_frst_homography(c: &mut Criterion) {
    let size = 256;
    let homography = Homography::new([
        [1.08, 0.04, 22.0],
        [0.02, 0.96, 10.0],
        [0.0007, -0.0004, 1.0],
    ])
    .unwrap();
    let rectified_circle = Circle::new(PixelCoord::new(128.0, 124.0), 30.0);
    let image = make_projective_disk_image(size, &homography, rectified_circle);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let grid = RectifiedGrid::new(size, size).unwrap();
    let config = FrstConfig {
        radii: vec![28, 30, 32],
        ..FrstConfig::default()
    };

    c.bench_function("frst_homography_256", |b| {
        b.iter(|| frst_response_homography(&gradient, &homography, grid, &config));
    });
}

criterion_group!(
    benches,
    bench_radial_center,
    bench_refine_circle,
    bench_refine_ellipse,
    bench_refine_ellipse_concentric_distractor,
    bench_refine_ellipse_homography,
    bench_frst_homography
);
criterion_main!(benches);
