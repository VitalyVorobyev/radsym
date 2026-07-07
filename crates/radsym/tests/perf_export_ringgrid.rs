//! Locks the ring-grid ellipses the performance page overlays. The page runs
//! `ringgrid_detect::detect_rings_in_image` on the real `testdata/ringgrid.png`
//! and draws each ring's outer (+inner) ellipse, so this test proves that
//! pipeline recovers every labelled ring center to sub-pixel accuracy on the
//! real image (the `ringgrid_regression.rs` suite only exercises a synthetic
//! raster).
#![cfg(all(feature = "image-io", feature = "serde"))]

use radsym::{PixelCoord, load_grayscale};

#[path = "support/ringgrid_detect.rs"]
mod ringgrid_detect;

fn root(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

struct Gt {
    center: PixelCoord,
    outer_a: f32,
    outer_b: f32,
}

fn ground_truth() -> Vec<Gt> {
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root("testdata/ringgrid_features.json")).unwrap(),
    )
    .unwrap();
    json["detected_markers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let c = m["center"].as_array().unwrap();
            let o = &m["ellipse_outer"];
            Gt {
                center: PixelCoord::new(
                    c[0].as_f64().unwrap() as f32,
                    c[1].as_f64().unwrap() as f32,
                ),
                outer_a: o["a"].as_f64().unwrap() as f32,
                outer_b: o["b"].as_f64().unwrap() as f32,
            }
        })
        .collect()
}

#[test]
fn perf_page_ringgrid_recovers_all_outer_ellipses() {
    let image = load_grayscale(root("testdata/ringgrid.png")).unwrap();
    let gt = ground_truth();
    let outer_hint = gt
        .iter()
        .map(|g| 0.5 * (g.outer_a + g.outer_b))
        .sum::<f32>()
        / gt.len() as f32;

    // Same parameters the perf page uses (see perf_export.rs RINGGRID_* consts).
    let rings = ringgrid_detect::detect_rings_in_image(&image, outer_hint, 0.47, gt.len() * 2);
    assert!(
        rings.len() >= gt.len(),
        "expected >= {} rings, got {}",
        gt.len(),
        rings.len()
    );

    // Match each labelled center to the nearest detected outer ellipse (<= 6px).
    let mut taken = vec![false; rings.len()];
    let mut matched = 0usize;
    let mut center_err_sum = 0.0f32;
    let mut axis_err_sum = 0.0f32;
    for g in &gt {
        let mut best = None;
        let mut best_d = 6.0f32;
        for (i, r) in rings.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let d = (r.outer.center.x - g.center.x).hypot(r.outer.center.y - g.center.y);
            if d < best_d {
                best_d = d;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            taken[i] = true;
            matched += 1;
            center_err_sum += best_d;
            let o = &rings[i].outer;
            axis_err_sum +=
                0.5 * ((o.semi_major - g.outer_a).abs() + (o.semi_minor - g.outer_b).abs());
        }
    }

    let mean_center_err = center_err_sum / matched.max(1) as f32;
    let mean_axis_err = axis_err_sum / matched.max(1) as f32;

    assert_eq!(matched, gt.len(), "not all labelled rings recovered");
    assert!(
        mean_center_err < 1.5,
        "outer center error too large: {mean_center_err:.2}px"
    );
    assert!(
        mean_axis_err < 1.0,
        "outer axis error too large: {mean_axis_err:.2}px"
    );
}
