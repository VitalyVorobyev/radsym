//! Radial and normal profile extraction.
//!
//! Profiles provide 1D cross-sections through the image around a hypothesis
//! center, useful for visualizing and characterizing ring-like structures.

use crate::core::coords::PixelCoord;
use crate::core::geometry::Ellipse;
use crate::core::image_view::ImageView;
use crate::core::scalar::Scalar;

/// A radial profile: azimuthally-averaged intensity as a function of radius.
#[derive(Debug, Clone)]
pub struct RadialProfile {
    /// Sampled intensity values (azimuthal average at each radius).
    pub values: Vec<Scalar>,
    /// Corresponding radii.
    pub radii: Vec<Scalar>,
    /// Center used for the profile.
    pub center: PixelCoord,
}

/// Compute an azimuthally-averaged radial profile.
///
/// For each radius step, samples at `num_angles` evenly spaced angles and
/// averages the bilinearly interpolated intensity.
pub fn compute_radial_profile(
    image: &ImageView<'_, f32>,
    center: PixelCoord,
    max_radius: Scalar,
    num_radial_steps: usize,
    num_angles: usize,
) -> RadialProfile {
    let mut values = Vec::with_capacity(num_radial_steps);
    let mut radii = Vec::with_capacity(num_radial_steps);

    for ri in 0..num_radial_steps {
        let r = max_radius * (ri as Scalar + 0.5) / num_radial_steps as Scalar;
        let mut sum = 0.0f32;
        let mut count = 0u32;

        for ai in 0..num_angles {
            let theta = 2.0 * std::f32::consts::PI * ai as Scalar / num_angles as Scalar;
            let sx = center.x + r * theta.cos();
            let sy = center.y + r * theta.sin();

            if let Some(val) = image.sample(sx, sy) {
                sum += val;
                count += 1;
            }
        }

        let avg = if count > 0 {
            sum / count as Scalar
        } else {
            0.0
        };
        values.push(avg);
        radii.push(r);
    }

    RadialProfile {
        values,
        radii,
        center,
    }
}

/// Compute a 1D profile along the normal direction of an ellipse at a given angle.
///
/// Samples along the outward normal from the ellipse center, at the specified
/// parametric angle on the ellipse. Useful for inspecting edge transitions.
pub fn compute_normal_profile(
    image: &ImageView<'_, f32>,
    ellipse: &Ellipse,
    parametric_angle: Scalar,
    profile_half_length: Scalar,
    num_steps: usize,
) -> Vec<Scalar> {
    let cos_a = ellipse.angle.cos();
    let sin_a = ellipse.angle.sin();

    // Point on ellipse in local frame
    let ex = ellipse.semi_major * parametric_angle.cos();
    let ey = ellipse.semi_minor * parametric_angle.sin();

    // Normal direction in local frame (perpendicular to tangent)
    // Tangent: (-semi_major * sin(t), semi_minor * cos(t))
    // Outward normal: (semi_minor * cos(t), semi_major * sin(t)) (unnormalized)
    let nx_local = ellipse.semi_minor * parametric_angle.cos();
    let ny_local = ellipse.semi_major * parametric_angle.sin();
    let n_mag = (nx_local * nx_local + ny_local * ny_local).sqrt();
    if n_mag < 1e-8 {
        return vec![0.0; num_steps];
    }
    let nx_local = nx_local / n_mag;
    let ny_local = ny_local / n_mag;

    // Rotate to image frame
    let base_x = ellipse.center.x + ex * cos_a - ey * sin_a;
    let base_y = ellipse.center.y + ex * sin_a + ey * cos_a;
    let dir_x = nx_local * cos_a - ny_local * sin_a;
    let dir_y = nx_local * sin_a + ny_local * cos_a;

    let mut profile = Vec::with_capacity(num_steps);
    for si in 0..num_steps {
        let t = -profile_half_length
            + 2.0 * profile_half_length * si as Scalar / (num_steps - 1).max(1) as Scalar;
        let sx = base_x + t * dir_x;
        let sy = base_y + t * dir_y;
        let val = image.sample(sx, sy).unwrap_or(0.0);
        profile.push(val);
    }

    profile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::geometry::Ellipse;
    use crate::core::image_view::OwnedImage;

    #[test]
    fn radial_profile_of_ring() {
        let size = 80;
        let cx = 40.0f32;
        let cy = 40.0f32;
        let r_inner = 14.0f32;
        let r_outer = 18.0f32;

        let mut data = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                if r >= r_inner && r <= r_outer {
                    data[y * size + x] = 1.0;
                }
            }
        }

        let img = OwnedImage::from_vec(data, size, size).unwrap();
        let profile = compute_radial_profile(&img.view(), PixelCoord::new(cx, cy), 30.0, 30, 64);

        assert_eq!(profile.values.len(), 30);
        assert_eq!(profile.radii.len(), 30);

        // Find the peak: should be around r=16 (middle of ring)
        let (peak_idx, _) = profile
            .values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let peak_r = profile.radii[peak_idx];
        let mid_r = (r_inner + r_outer) / 2.0;
        assert!(
            (peak_r - mid_r).abs() < 3.0,
            "peak at r={peak_r}, expected near {mid_r}"
        );
    }

    #[test]
    fn radial_profile_of_uniform() {
        let size = 40;
        let data = vec![0.5f32; size * size];
        let img = OwnedImage::from_vec(data, size, size).unwrap();
        let profile =
            compute_radial_profile(&img.view(), PixelCoord::new(20.0, 20.0), 10.0, 10, 32);

        // All values should be ~0.5
        for &v in &profile.values {
            assert!((v - 0.5).abs() < 0.01, "expected ~0.5, got {v}");
        }
    }

    #[test]
    fn normal_profile_length() {
        let size = 40;
        let data = vec![1.0f32; size * size];
        let img = OwnedImage::from_vec(data, size, size).unwrap();
        let ellipse = Ellipse::new(PixelCoord::new(20.0, 20.0), 10.0, 10.0, 0.0);
        let profile = compute_normal_profile(&img.view(), &ellipse, 0.0, 5.0, 11);
        assert_eq!(profile.len(), 11);
    }
}
