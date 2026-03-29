#[path = "support/surf_hole_synthetic.rs"]
mod surf_hole_synthetic;

use radsym::Ellipse;

use surf_hole_synthetic::{
    detect_case_image, ellipse_iou, normalize_ellipse, render_case, CASES, DEFAULT_PYRAMID_LEVEL,
};

#[test]
fn surf_hole_composition_matches_synthetic_regression_cases() {
    for case in CASES {
        let image = render_case(&case);
        let detection = detect_case_image(&image, DEFAULT_PYRAMID_LEVEL).unwrap();
        let pred = normalize_ellipse(detection.best.image_ellipse);
        let target = normalize_ellipse(Ellipse::new(
            case.target.center,
            case.target.semi_major,
            case.target.semi_minor,
            case.target.angle,
        ));

        let center_error = (pred.center.x - target.center.x).hypot(pred.center.y - target.center.y);
        let iou = ellipse_iou(pred, target);

        assert!(
            center_error <= 14.0,
            "{} center error too large: {:.2}px; predicted center=({:.2}, {:.2}) target=({:.2}, {:.2})",
            case.name,
            center_error,
            pred.center.x,
            pred.center.y,
            target.center.x,
            target.center.y,
        );
        assert!(
            iou >= 0.78,
            "{} IoU too low: {:.3}; predicted axes=({:.2}, {:.2}) angle={:.3} target axes=({:.2}, {:.2}) angle={:.3}",
            case.name,
            iou,
            pred.semi_major,
            pred.semi_minor,
            pred.angle,
            target.semi_major,
            target.semi_minor,
            target.angle,
        );
    }
}
