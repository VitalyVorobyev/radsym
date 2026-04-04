#![cfg(all(feature = "image-io", feature = "serde"))]

use std::path::PathBuf;

use radsym::core::gradient::sobel_gradient;
use radsym::core::nms::NmsConfig;
use radsym::propose::rsd::RsdConfig;
use radsym::{
    Ellipse, EllipseRefineConfig, OwnedImage, PixelCoord, Polarity, Proposal, extract_proposals,
    refine_ellipse, rsd_response, suppress_proposals_by_distance,
};

#[derive(serde::Deserialize)]
struct RinggridFixture {
    detected_markers: Vec<MarkerFixture>,
}

#[derive(serde::Deserialize)]
struct MarkerFixture {
    center: [f32; 2],
    ellipse_outer: EllipseFixture,
    ellipse_inner: Option<EllipseFixture>,
}

#[derive(serde::Deserialize)]
struct EllipseFixture {
    a: f32,
    b: f32,
    angle: f32,
}

fn testdata_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata")
        .join(name)
}

fn fixture() -> RinggridFixture {
    let path = testdata_path("ringgrid_features.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn raster_bounds(fixture: &RinggridFixture) -> (usize, usize) {
    let mut width = 0.0f32;
    let mut height = 0.0f32;

    for marker in &fixture.detected_markers {
        let outer_extent = marker.ellipse_outer.a.max(marker.ellipse_outer.b) + 12.0;
        width = width.max(marker.center[0] + outer_extent);
        height = height.max(marker.center[1] + outer_extent);
    }

    (
        (width.ceil() as usize).max(2),
        (height.ceil() as usize).max(2),
    )
}

fn fill_ellipse(
    data: &mut [f32],
    width: usize,
    height: usize,
    center: PixelCoord,
    ellipse: &EllipseFixture,
    value: f32,
) {
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();
    let extent = ellipse.a.max(ellipse.b) + 2.5;
    let x0 = (center.x - extent).floor().max(0.0) as usize;
    let y0 = (center.y - extent).floor().max(0.0) as usize;
    let x1 = (center.x + extent).ceil().min((width - 1) as f32) as usize;
    let y1 = (center.y + extent).ceil().min((height - 1) as f32) as usize;

    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 - center.x;
            let dy = y as f32 - center.y;
            let lx = dx * cos_a + dy * sin_a;
            let ly = -dx * sin_a + dy * cos_a;
            let level = (lx / ellipse.a).powi(2) + (ly / ellipse.b).powi(2);
            if level <= 1.0 {
                data[y * width + x] = value;
            }
        }
    }
}

fn blur3x3_inplace(data: &mut [f32], width: usize, height: usize, passes: usize) {
    let mut tmp = vec![0.0f32; data.len()];

    for _ in 0..passes {
        for y in 0..height {
            for x in 0..width {
                let left = x.saturating_sub(1);
                let right = (x + 1).min(width - 1);
                tmp[y * width + x] =
                    (data[y * width + left] + data[y * width + x] + data[y * width + right]) / 3.0;
            }
        }

        for y in 0..height {
            let up = y.saturating_sub(1);
            let down = (y + 1).min(height - 1);
            for x in 0..width {
                data[y * width + x] =
                    (tmp[up * width + x] + tmp[y * width + x] + tmp[down * width + x]) / 3.0;
            }
        }
    }
}

fn ringgrid_image(fixture: &RinggridFixture) -> OwnedImage<u8> {
    let (width, height) = raster_bounds(fixture);
    let mut data = vec![232.0f32; width * height];

    for marker in &fixture.detected_markers {
        let center = PixelCoord::new(marker.center[0], marker.center[1]);
        fill_ellipse(
            &mut data,
            width,
            height,
            center,
            &marker.ellipse_outer,
            26.0,
        );
        if let Some(inner) = marker.ellipse_inner.as_ref() {
            fill_ellipse(&mut data, width, height, center, inner, 232.0);
        }
    }

    blur3x3_inplace(&mut data, width, height, 2);

    let data = data
        .into_iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    OwnedImage::from_vec(data, width, height).unwrap()
}

fn build_radius_band(
    base_radius: f32,
    start_scale: f32,
    stop_scale: f32,
    steps: usize,
) -> Vec<u32> {
    let start = (base_radius * start_scale).round().max(4.0) as u32;
    let stop = (base_radius * stop_scale).round().max(start as f32 + 1.0) as u32;
    if steps <= 1 || start == stop {
        return vec![start];
    }

    let mut radii = (0..steps)
        .map(|index| {
            let t = index as f32 / (steps - 1) as f32;
            (start as f32 + t * (stop - start) as f32).round() as u32
        })
        .collect::<Vec<_>>();
    radii.sort_unstable();
    radii.dedup();
    radii
}

fn gt_centers(fixture: &RinggridFixture) -> Vec<PixelCoord> {
    fixture
        .detected_markers
        .iter()
        .map(|marker| PixelCoord::new(marker.center[0], marker.center[1]))
        .collect()
}

fn outer_radius_hint(fixture: &RinggridFixture) -> f32 {
    fixture
        .detected_markers
        .iter()
        .map(|marker| 0.5 * (marker.ellipse_outer.a + marker.ellipse_outer.b))
        .sum::<f32>()
        / fixture.detected_markers.len() as f32
}

fn inner_ratio_hint(fixture: &RinggridFixture) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for marker in &fixture.detected_markers {
        let Some(inner) = marker.ellipse_inner.as_ref() else {
            continue;
        };
        let outer_mean = 0.5 * (marker.ellipse_outer.a + marker.ellipse_outer.b);
        let inner_mean = 0.5 * (inner.a + inner.b);
        sum += inner_mean / outer_mean;
        count += 1;
    }
    sum / count.max(1) as f32
}

fn match_indices(
    reference: &[PixelCoord],
    candidates: &[PixelCoord],
    gate: f32,
) -> Vec<(usize, usize)> {
    let mut taken = vec![false; candidates.len()];
    let mut matched = Vec::new();

    for (gt_index, gt) in reference.iter().enumerate() {
        let mut best_index = None;
        let mut best_distance = gate;

        for (candidate_index, candidate) in candidates.iter().enumerate() {
            if taken[candidate_index] {
                continue;
            }

            let dx = gt.x - candidate.x;
            let dy = gt.y - candidate.y;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance <= best_distance {
                best_distance = distance;
                best_index = Some(candidate_index);
            }
        }

        if let Some(candidate_index) = best_index {
            taken[candidate_index] = true;
            matched.push((gt_index, candidate_index));
        }
    }

    matched
}

fn detect_outer_rsd_candidates(
    fixture: &RinggridFixture,
    gradient: &radsym::core::gradient::GradientField,
) -> Vec<Proposal> {
    let outer_hint = outer_radius_hint(fixture);
    let response = rsd_response(
        gradient,
        &RsdConfig {
            radii: build_radius_band(outer_hint, 0.8, 1.16, 5),
            gradient_threshold: 2.0,
            polarity: Polarity::Dark,
            smoothing_factor: 0.5,
        },
    )
    .unwrap();
    let proposals = extract_proposals(
        &response,
        &NmsConfig {
            radius: (0.55 * outer_hint).round().max(6.0) as usize,
            threshold: 0.01,
            max_detections: 256,
        },
        Polarity::Dark,
    );

    suppress_proposals_by_distance(
        &proposals,
        1.25 * outer_hint,
        fixture.detected_markers.len() * 2,
    )
}

#[test]
fn ringgrid_outer_rsd_candidates_recall_all_ground_truth_centers() {
    let fixture = fixture();
    let image = ringgrid_image(&fixture);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let gt_centers = gt_centers(&fixture);

    let proposals = detect_outer_rsd_candidates(&fixture, &gradient);
    let proposal_centers = proposals
        .iter()
        .map(|proposal| proposal.seed.position)
        .collect::<Vec<_>>();
    let matched = match_indices(&gt_centers, &proposal_centers, 10.0);

    let mean_center_error = matched
        .iter()
        .map(|(gt_index, candidate_index)| {
            let gt = gt_centers[*gt_index];
            let candidate = proposal_centers[*candidate_index];
            let dx = gt.x - candidate.x;
            let dy = gt.y - candidate.y;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f32>()
        / matched.len().max(1) as f32;
    let max_center_error = matched
        .iter()
        .map(|(gt_index, candidate_index)| {
            let gt = gt_centers[*gt_index];
            let candidate = proposal_centers[*candidate_index];
            let dx = gt.x - candidate.x;
            let dy = gt.y - candidate.y;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f32::max);

    // The JSON fixture is partial ground truth: all annotated markers must be
    // recoverable, but additional valid seeds on the same hex lattice are
    // allowed and desirable.
    assert_eq!(matched.len(), fixture.detected_markers.len());
    assert!(proposals.len() >= fixture.detected_markers.len());
    assert!(mean_center_error < 1.2);
    assert!(max_center_error < 2.5);
}

#[test]
fn ringgrid_local_ellipse_refinement_recovers_outer_and_inner_geometry() {
    let fixture = fixture();
    let image = ringgrid_image(&fixture);
    let gradient = sobel_gradient(&image.view()).unwrap();
    let gt_centers = gt_centers(&fixture);
    let outer_hint = outer_radius_hint(&fixture);
    let inner_ratio = inner_ratio_hint(&fixture);

    let proposals = detect_outer_rsd_candidates(&fixture, &gradient);
    let outer_config = EllipseRefineConfig {
        max_iterations: 5,
        convergence_tol: 0.05,
        annulus_margin: 0.12,
        ray_count: 96,
        radial_search_inner: 0.60,
        radial_search_outer: 1.45,
        normal_search_half_width: 6.0,
        min_inlier_coverage: 0.60,
        max_center_shift_fraction: 0.40,
        max_axis_ratio: 1.80,
        ..EllipseRefineConfig::default()
    };
    let inner_config = EllipseRefineConfig {
        max_iterations: 5,
        convergence_tol: 0.05,
        annulus_margin: 0.10,
        ray_count: 96,
        radial_search_inner: 0.75,
        radial_search_outer: 1.20,
        normal_search_half_width: 4.0,
        min_inlier_coverage: 0.55,
        max_center_shift_fraction: 0.25,
        max_axis_ratio: 1.80,
        ..EllipseRefineConfig::default()
    };

    let refined_outer = proposals
        .iter()
        .map(|proposal| {
            refine_ellipse(
                &gradient,
                &Ellipse::new(proposal.seed.position, outer_hint, outer_hint, 0.0),
                &outer_config,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let refined_outer_centers = refined_outer
        .iter()
        .map(|result| result.hypothesis.center)
        .collect::<Vec<_>>();
    let matched_outer = match_indices(&gt_centers, &refined_outer_centers, 10.0);

    let mean_outer_center_error = matched_outer
        .iter()
        .map(|(gt_index, candidate_index)| {
            let gt = gt_centers[*gt_index];
            let predicted = refined_outer[*candidate_index].hypothesis.center;
            let dx = gt.x - predicted.x;
            let dy = gt.y - predicted.y;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f32>()
        / matched_outer.len().max(1) as f32;
    let mean_outer_axis_error = matched_outer
        .iter()
        .map(|(gt_index, candidate_index)| {
            let gt = &fixture.detected_markers[*gt_index].ellipse_outer;
            let predicted = &refined_outer[*candidate_index].hypothesis;
            0.5 * ((predicted.semi_major - gt.a).abs() + (predicted.semi_minor - gt.b).abs())
        })
        .sum::<f32>()
        / matched_outer.len().max(1) as f32;

    let matched_inner = matched_outer
        .iter()
        .filter_map(|(gt_index, candidate_index)| {
            let gt_inner = fixture.detected_markers[*gt_index].ellipse_inner.as_ref()?;
            let gt_center = fixture.detected_markers[*gt_index].center;
            let outer = refined_outer[*candidate_index].hypothesis;
            let inner_seed = Ellipse::new(
                outer.center,
                outer.semi_major * inner_ratio,
                outer.semi_minor * inner_ratio,
                outer.angle,
            );
            let refined = refine_ellipse(&gradient, &inner_seed, &inner_config).ok()?;
            Some((gt_center, gt_inner, refined.hypothesis))
        })
        .collect::<Vec<_>>();
    let mean_inner_center_error = matched_inner
        .iter()
        .map(|(gt_center, _, predicted)| {
            let dx = gt_center[0] - predicted.center.x;
            let dy = gt_center[1] - predicted.center.y;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f32>()
        / matched_inner.len().max(1) as f32;
    let mean_inner_axis_error = matched_inner
        .iter()
        .map(|(_, gt, predicted)| {
            0.5 * ((predicted.semi_major - gt.a).abs() + (predicted.semi_minor - gt.b).abs())
        })
        .sum::<f32>()
        / matched_inner.len().max(1) as f32;
    let expected_inner = fixture
        .detected_markers
        .iter()
        .filter(|marker| marker.ellipse_inner.is_some())
        .count();

    assert_eq!(matched_outer.len(), fixture.detected_markers.len());
    assert!(mean_outer_center_error < 0.8);
    assert!(mean_outer_axis_error < 0.6);

    assert_eq!(matched_inner.len(), expected_inner);
    assert!(mean_inner_center_error < 1.0);
    assert!(mean_inner_axis_error < 0.75);
}
