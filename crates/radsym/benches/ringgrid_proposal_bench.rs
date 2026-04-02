#[cfg(feature = "image-io")]
use std::hint::black_box;
#[cfg(feature = "image-io")]
use std::path::PathBuf;

#[cfg(feature = "image-io")]
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "image-io")]
use radsym::core::gradient::sobel_gradient;
#[cfg(feature = "image-io")]
use radsym::core::nms::{non_maximum_suppression, NmsConfig};
#[cfg(feature = "image-io")]
use radsym::propose::extract::{extract_proposals, ResponseMap};
#[cfg(feature = "image-io")]
use radsym::propose::rsd::{rsd_response, rsd_response_single, RsdConfig};
#[cfg(feature = "image-io")]
use radsym::{load_grayscale, Polarity};

#[cfg(feature = "image-io")]
struct BenchFixture {
    gradient: radsym::core::gradient::GradientField,
    response: ResponseMap,
    single_vote_only: RsdConfig,
    single_smoothed: RsdConfig,
    multi_vote_only: RsdConfig,
    multi_radius: RsdConfig,
    nms: NmsConfig,
}

#[cfg(feature = "image-io")]
fn ringgrid_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join("ringgrid.png")
}

#[cfg(feature = "image-io")]
fn build_fixture() -> BenchFixture {
    let image = load_grayscale(ringgrid_path()).unwrap();
    let gradient = sobel_gradient(&image.view()).unwrap();

    let radii = vec![18, 20, 22, 24, 26];
    let single_vote_only = RsdConfig {
        radii: vec![22],
        gradient_threshold: 2.0,
        polarity: Polarity::Dark,
        smoothing_factor: 0.0,
    };
    let single_smoothed = RsdConfig {
        radii: vec![22],
        gradient_threshold: 2.0,
        polarity: Polarity::Dark,
        smoothing_factor: 0.5,
    };
    let multi_vote_only = RsdConfig {
        radii: vec![18, 20, 22, 24, 26],
        gradient_threshold: 2.0,
        polarity: Polarity::Dark,
        smoothing_factor: 0.0,
    };
    let multi_radius = RsdConfig {
        radii,
        gradient_threshold: 2.0,
        polarity: Polarity::Dark,
        smoothing_factor: 0.5,
    };

    let response = rsd_response(&gradient, &multi_radius).unwrap();
    let nms = NmsConfig {
        radius: 13,
        threshold: 0.01,
        max_detections: 256,
    };

    BenchFixture {
        gradient,
        response,
        single_vote_only,
        single_smoothed,
        multi_vote_only,
        multi_radius,
        nms,
    }
}

#[cfg(feature = "image-io")]
fn bench_ringgrid_proposals(c: &mut Criterion) {
    let fixture = build_fixture();
    let mut group = c.benchmark_group("ringgrid_proposals");

    group.bench_function("rsd_single_vote_only_ringgrid_720x540_r22", |b| {
        b.iter(|| {
            rsd_response_single(
                black_box(&fixture.gradient),
                black_box(22),
                black_box(&fixture.single_vote_only),
            )
            .unwrap()
        })
    });

    group.bench_function("rsd_single_smoothed_ringgrid_720x540_r22", |b| {
        b.iter(|| {
            rsd_response_single(
                black_box(&fixture.gradient),
                black_box(22),
                black_box(&fixture.single_smoothed),
            )
            .unwrap()
        })
    });

    group.bench_function("rsd_multi_ringgrid_720x540_r18_26_n5", |b| {
        b.iter(|| {
            rsd_response(
                black_box(&fixture.gradient),
                black_box(&fixture.multi_radius),
            )
            .unwrap()
        })
    });

    group.bench_function("rsd_multi_vote_only_ringgrid_720x540_r18_26_n5", |b| {
        b.iter(|| {
            rsd_response(
                black_box(&fixture.gradient),
                black_box(&fixture.multi_vote_only),
            )
            .unwrap()
        })
    });

    group.bench_function("nms_ringgrid_720x540_r13", |b| {
        b.iter(|| {
            non_maximum_suppression(black_box(&fixture.response.view()), black_box(&fixture.nms))
        })
    });

    group.bench_function("extract_proposals_ringgrid_720x540_r13", |b| {
        b.iter(|| {
            extract_proposals(
                black_box(&fixture.response),
                black_box(&fixture.nms),
                black_box(Polarity::Dark),
            )
        })
    });

    group.finish();
}

#[cfg(feature = "image-io")]
criterion_group!(benches, bench_ringgrid_proposals);
#[cfg(feature = "image-io")]
criterion_main!(benches);

#[cfg(not(feature = "image-io"))]
fn main() {}
