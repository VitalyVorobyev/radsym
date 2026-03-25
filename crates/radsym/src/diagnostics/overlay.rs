//! Shape overlay drawing for diagnostic images.
//!
//! Provides functions to draw geometric primitives (circles, ellipses,
//! markers) onto [`DiagnosticImage`] buffers.

use crate::core::geometry::{Circle, Ellipse};
use crate::core::scalar::Scalar;
use crate::propose::seed::Proposal;

use super::heatmap::DiagnosticImage;

/// Draw a circle outline onto a diagnostic image.
pub fn overlay_circle(img: &mut DiagnosticImage, circle: &Circle, color: [u8; 4]) {
    let n_points = ((2.0 * std::f32::consts::PI * circle.radius) as usize).max(32);
    for i in 0..n_points {
        let theta = 2.0 * std::f32::consts::PI * i as Scalar / n_points as Scalar;
        let x = (circle.center.x + circle.radius * theta.cos()).round() as i32;
        let y = (circle.center.y + circle.radius * theta.sin()).round() as i32;
        if x >= 0 && (x as usize) < img.width() && y >= 0 && (y as usize) < img.height() {
            img.set_pixel(x as usize, y as usize, color);
        }
    }
}

/// Draw an ellipse outline onto a diagnostic image.
pub fn overlay_ellipse(img: &mut DiagnosticImage, ellipse: &Ellipse, color: [u8; 4]) {
    let max_axis = ellipse.semi_major.max(ellipse.semi_minor);
    let n_points = ((2.0 * std::f32::consts::PI * max_axis) as usize).max(64);
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();

    for i in 0..n_points {
        let theta = 2.0 * std::f32::consts::PI * i as Scalar / n_points as Scalar;
        let lx = ellipse.semi_major * theta.cos();
        let ly = ellipse.semi_minor * theta.sin();

        let x = (ellipse.center.x + lx * cos_a - ly * sin_a).round() as i32;
        let y = (ellipse.center.y + lx * sin_a + ly * cos_a).round() as i32;

        if x >= 0 && (x as usize) < img.width() && y >= 0 && (y as usize) < img.height() {
            img.set_pixel(x as usize, y as usize, color);
        }
    }
}

/// Draw a cross marker at a position.
pub fn overlay_marker(
    img: &mut DiagnosticImage,
    x: Scalar,
    y: Scalar,
    size: usize,
    color: [u8; 4],
) {
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let s = size as i32;

    for dx in -s..=s {
        let px = cx + dx;
        if px >= 0 && (px as usize) < img.width() && cy >= 0 && (cy as usize) < img.height() {
            img.set_pixel(px as usize, cy as usize, color);
        }
    }
    for dy in -s..=s {
        let py = cy + dy;
        if cx >= 0 && (cx as usize) < img.width() && py >= 0 && (py as usize) < img.height() {
            img.set_pixel(cx as usize, py as usize, color);
        }
    }
}

/// Draw markers at each proposal location.
pub fn overlay_proposals(
    img: &mut DiagnosticImage,
    proposals: &[Proposal],
    color: [u8; 4],
    marker_size: usize,
) {
    for p in proposals {
        overlay_marker(
            img,
            p.seed.position.x,
            p.seed.position.y,
            marker_size,
            color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::coords::PixelCoord;

    #[test]
    fn overlay_circle_draws_pixels() {
        let mut img = DiagnosticImage::new(80, 80);
        let circle = Circle::new(PixelCoord::new(40.0, 40.0), 15.0);
        overlay_circle(&mut img, &circle, [255, 0, 0, 255]);

        // Check that some pixels near the circle are non-zero
        let px = img.get_pixel(55, 40); // right edge
        assert_eq!(px, [255, 0, 0, 255]);
    }

    #[test]
    fn overlay_marker_draws_cross() {
        let mut img = DiagnosticImage::new(20, 20);
        overlay_marker(&mut img, 10.0, 10.0, 3, [0, 255, 0, 255]);

        assert_eq!(img.get_pixel(10, 10), [0, 255, 0, 255]);
        assert_eq!(img.get_pixel(13, 10), [0, 255, 0, 255]);
        assert_eq!(img.get_pixel(10, 13), [0, 255, 0, 255]);
    }

    #[test]
    fn overlay_ellipse_draws_pixels() {
        let mut img = DiagnosticImage::new(80, 80);
        let ellipse = Ellipse::new(PixelCoord::new(40.0, 40.0), 20.0, 10.0, 0.0);
        overlay_ellipse(&mut img, &ellipse, [0, 0, 255, 255]);

        // Check right edge of semi-major axis
        let px = img.get_pixel(60, 40);
        assert_eq!(px, [0, 0, 255, 255]);
    }
}
