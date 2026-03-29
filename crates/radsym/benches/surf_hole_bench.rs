use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use radsym::PyramidWorkspace;

#[path = "../tests/support/surf_hole_synthetic.rs"]
mod surf_hole_synthetic;

use surf_hole_synthetic::{detect_case_image, render_case, CASES, DEFAULT_PYRAMID_LEVEL};

fn bench_surf_hole(c: &mut Criterion) {
    let image = render_case(&CASES[0]);
    let view = image.view();
    let mut workspace = PyramidWorkspace::with_capacity(DEFAULT_PYRAMID_LEVEL.saturating_add(1));
    let mut group = c.benchmark_group("surf_hole");

    group.bench_function("pyramid_level_owned_synthetic_2048x1536_l3", |b| {
        b.iter(|| {
            radsym::pyramid_level_owned(black_box(&view), black_box(DEFAULT_PYRAMID_LEVEL)).unwrap()
        })
    });

    group.bench_function("pyramid_workspace_level_synthetic_2048x1536_l3", |b| {
        b.iter(|| {
            let level = workspace
                .level(black_box(view), black_box(DEFAULT_PYRAMID_LEVEL))
                .unwrap();
            black_box(level.image().width())
        })
    });

    group.bench_function("surf_hole_composed_synthetic_2048x1536_l3", |b| {
        b.iter(|| detect_case_image(black_box(&image), black_box(DEFAULT_PYRAMID_LEVEL)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_surf_hole);
criterion_main!(benches);
