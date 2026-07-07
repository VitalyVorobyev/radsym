//! Locks the surface-hole ellipses that the performance page
//! (`examples/perf_export.rs`) overlays. The page renders `CASES[0]` via
//! `render_case` + `detect_all_ellipses` and draws every returned
//! `image_ellipse`, so this test proves those ellipses are finite and that all
//! three ground-truth features (the target hole plus two distractors) are
//! recovered accurately.

#[path = "support/surf_hole_synthetic.rs"]
mod surf_hole_synthetic;

use radsym::PixelCoord;

use surf_hole_synthetic::{CASES, detect_all_ellipses, render_case};

#[test]
fn perf_page_surf_overlay_detects_all_three_ellipses() {
    let case = CASES[0]; // the exact case perf_export publishes as "surf-hole".
    let image = render_case(&case);
    let detection =
        detect_all_ellipses(&image, surf_hole_synthetic::DEFAULT_PYRAMID_LEVEL).unwrap();

    // Every drawn ellipse must be finite with positive axes.
    for e in &detection.ellipses {
        let g = e.image_ellipse;
        for (name, v) in [
            ("center.x", g.center.x),
            ("center.y", g.center.y),
            ("semi_major", g.semi_major),
            ("semi_minor", g.semi_minor),
            ("angle", g.angle),
        ] {
            assert!(v.is_finite(), "drawn ellipse {name} not finite: {v}");
        }
        assert!(
            g.semi_major > 0.0 && g.semi_minor > 0.0,
            "drawn ellipse has non-positive axes: ({}, {})",
            g.semi_major,
            g.semi_minor,
        );
    }

    // Ground-truth features: target + distractors.
    let mut gts: Vec<(PixelCoord, f32, f32)> = vec![(
        case.target.center,
        case.target.semi_major,
        case.target.semi_minor,
    )];
    for d in case.distractors {
        gts.push((d.center, d.semi_major, d.semi_minor));
    }

    for (center, semi_major, semi_minor) in gts {
        let best = detection
            .ellipses
            .iter()
            .map(|e| {
                let g = e.image_ellipse;
                let d = (g.center.x - center.x).hypot(g.center.y - center.y);
                (d, g)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("at least one detected ellipse");

        assert!(
            best.0 <= 15.0,
            "GT ellipse ({:.0},{:.0}) not detected: nearest center is {:.1}px away",
            center.x,
            center.y,
            best.0,
        );
        // Recovered axes should be within ~20% of the labelled semi-axes.
        let mean_gt = 0.5 * (semi_major + semi_minor);
        let mean_pred = 0.5 * (best.1.semi_major + best.1.semi_minor);
        assert!(
            (mean_pred - mean_gt).abs() <= 0.2 * mean_gt,
            "GT ellipse ({:.0},{:.0}) axis mismatch: mean {:.1} vs {:.1}",
            center.x,
            center.y,
            mean_pred,
            mean_gt,
        );
    }
}
