//! Gaussian blur for response maps.
//!
//! For small sigma (≤ 2.0) uses direct separable convolution.
//! For larger sigma uses a 3-pass stacked box blur approximation,
//! which is O(1) per pixel regardless of sigma.
//!
//! Reference for stacked box blur:
//! - Wells, W.M. (1986). *Efficient Synthesis of Gaussian Filters by
//!   Cascaded Uniform Filters.* IEEE TPAMI 8(2).
//! - W3C Filter Effects Module Level 1, §12.2.

use super::image_view::OwnedImage;
use super::scalar::Scalar;

/// Gaussian blur on an `OwnedImage<f32>`, in-place.
///
/// Uses a 1D Gaussian kernel with `radius = ceil(3 * sigma)` and mirror-clamp
/// boundary handling for small sigma. For sigma > 2.0, switches to a 3-pass
/// stacked box blur approximation (O(1) per pixel).
///
/// No-op if `sigma <= 0.5` (kernel radius would be zero).
pub(crate) fn gaussian_blur_inplace(image: &mut OwnedImage<Scalar>, sigma: Scalar) {
    if sigma <= 0.5 {
        return;
    }
    if sigma <= 2.0 {
        direct_gaussian_blur_inplace(image, sigma);
    } else {
        stacked_box_blur_inplace(image, sigma);
    }
}

/// Direct separable Gaussian convolution (original implementation).
///
/// Cost: O(w * h * ceil(3*sigma)) per pass.
fn direct_gaussian_blur_inplace(image: &mut OwnedImage<Scalar>, sigma: Scalar) {
    let w = image.width();
    let h = image.height();

    let krad = (3.0 * sigma).ceil() as usize;
    if krad == 0 {
        return;
    }
    let ksize = 2 * krad + 1;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f32; ksize];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for (i, k) in kernel.iter_mut().enumerate() {
        let d = i as Scalar - krad as Scalar;
        *k = (-d * d / s2).exp();
        sum += *k;
    }
    for k in &mut kernel {
        *k /= sum;
    }

    // Horizontal pass
    let mut buf = vec![0.0f32; w * h];
    let data = image.data();
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - krad as i32).clamp(0, w as i32 - 1) as usize;
                acc += data[y * w + sx] * kv;
            }
            buf[y * w + x] = acc;
        }
    }

    // Vertical pass
    let out = image.data_mut();
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - krad as i32).clamp(0, h as i32 - 1) as usize;
                acc += buf[sy * w + x] * kv;
            }
            out[y * w + x] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Stacked box blur: 3-pass O(1)-per-pixel Gaussian approximation
// ---------------------------------------------------------------------------

/// Compute box radii for a 3-pass stacked box blur approximating a Gaussian
/// with standard deviation `sigma`.
///
/// Returns three box half-widths (radii). A box of radius `r` has width `2r+1`.
fn box_radii_for_sigma(sigma: Scalar) -> [usize; 3] {
    // Ideal box width: w_ideal = sqrt(12 * sigma^2 / N + 1), N = 3 passes
    let w_ideal = (12.0 * sigma * sigma / 3.0 + 1.0).sqrt();
    let wl_raw = w_ideal.floor() as usize;
    let wl = if wl_raw % 2 == 0 { wl_raw - 1 } else { wl_raw }; // largest odd <= w_ideal
    let wu = wl + 2;

    // How many passes use wl vs wu:
    // variance of N box blurs = N * (w^2 - 1) / 12
    // We need: m * (wl^2-1)/12 + (3-m) * (wu^2-1)/12 = sigma^2
    // Solving for m:
    let target_var = 12.0 * sigma * sigma;
    let wl2 = (wl * wl) as Scalar;
    let wu2 = (wu * wu) as Scalar;
    let m_ideal = (3.0 * wu2 - 3.0 - target_var) / (wu2 - wl2);
    let m = m_ideal.round().clamp(0.0, 3.0) as usize;

    let rl = wl / 2;
    let ru = wu / 2;
    match m {
        0 => [ru, ru, ru],
        1 => [rl, ru, ru],
        2 => [rl, rl, ru],
        _ => [rl, rl, rl],
    }
}

/// 3-pass stacked box blur approximating Gaussian with standard deviation `sigma`.
///
/// Each box blur pass is O(1) per pixel using running sums, making the total
/// cost independent of sigma. Mirror-clamp boundary handling.
fn stacked_box_blur_inplace(image: &mut OwnedImage<Scalar>, sigma: Scalar) {
    let w = image.width();
    let h = image.height();
    let mut buf = vec![0.0f32; w * h];

    let radii = box_radii_for_sigma(sigma);
    for &r in &radii {
        if r == 0 {
            continue;
        }
        box_blur_horizontal(image.data(), &mut buf, w, h, r);
        box_blur_vertical(&buf, image.data_mut(), w, h, r);
    }
}

/// Horizontal box blur pass using running sums. O(1) per pixel.
///
/// Reads from `src`, writes to `dst`. Box radius `r` gives window width `2r+1`.
/// Uses mirror-clamp boundary handling.
fn box_blur_horizontal(src: &[Scalar], dst: &mut [Scalar], w: usize, h: usize, r: usize) {
    let diameter = 2 * r + 1;
    let inv = 1.0 / diameter as Scalar;
    let w_i32 = w as i32;

    for y in 0..h {
        let row = y * w;

        // Initialize running sum with the first window (centered at x=0)
        let mut sum = 0.0f32;
        for i in 0..diameter {
            let sx = (i as i32 - r as i32).clamp(0, w_i32 - 1) as usize;
            sum += src[row + sx];
        }
        dst[row] = sum * inv;

        // Slide the window across the row
        for x in 1..w {
            // Add entering pixel (right edge of new window)
            let enter = (x as i32 + r as i32).clamp(0, w_i32 - 1) as usize;
            // Remove leaving pixel (left edge of old window)
            let leave = (x as i32 - r as i32 - 1).clamp(0, w_i32 - 1) as usize;
            sum += src[row + enter] - src[row + leave];
            dst[row + x] = sum * inv;
        }
    }
}

/// Vertical box blur pass using running sums. O(1) per pixel.
///
/// Reads from `src`, writes to `dst`. Box radius `r` gives window width `2r+1`.
/// Uses mirror-clamp boundary handling.
fn box_blur_vertical(src: &[Scalar], dst: &mut [Scalar], w: usize, h: usize, r: usize) {
    let diameter = 2 * r + 1;
    let inv = 1.0 / diameter as Scalar;
    let h_i32 = h as i32;

    for x in 0..w {
        // Initialize running sum with the first window (centered at y=0)
        let mut sum = 0.0f32;
        for i in 0..diameter {
            let sy = (i as i32 - r as i32).clamp(0, h_i32 - 1) as usize;
            sum += src[sy * w + x];
        }
        dst[x] = sum * inv;

        // Slide the window down the column
        for y in 1..h {
            let enter = (y as i32 + r as i32).clamp(0, h_i32 - 1) as usize;
            let leave = (y as i32 - r as i32 - 1).clamp(0, h_i32 - 1) as usize;
            sum += src[enter * w + x] - src[leave * w + x];
            dst[y * w + x] = sum * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image_view::OwnedImage;

    /// Stacked box blur should approximate a Gaussian within tolerance.
    ///
    /// Applies both methods to a centered impulse and checks that the
    /// peak-normalized maximum absolute difference is small.
    #[test]
    fn box_blur_approximates_gaussian() {
        let size = 64;
        let sigma = 10.0;

        // Create impulse image
        let mut data = vec![0.0f32; size * size];
        data[size / 2 * size + size / 2] = 1.0;

        let mut img_direct = OwnedImage::from_vec(data.clone(), size, size).unwrap();
        let mut img_box = OwnedImage::from_vec(data, size, size).unwrap();

        direct_gaussian_blur_inplace(&mut img_direct, sigma);
        stacked_box_blur_inplace(&mut img_box, sigma);

        let d = img_direct.data();
        let b = img_box.data();

        let peak = d.iter().copied().fold(0.0f32, Scalar::max);
        assert!(peak > 0.0, "direct Gaussian peak should be positive");

        let max_err = d
            .iter()
            .zip(b.iter())
            .map(|(&dv, &bv)| (dv - bv).abs())
            .fold(0.0f32, Scalar::max);

        let relative_err = max_err / peak;
        assert!(
            relative_err < 0.10,
            "box blur should approximate Gaussian within 10%, got {:.1}%",
            relative_err * 100.0
        );
    }

    /// Box radii computation produces reasonable values.
    #[test]
    fn box_radii_sanity() {
        // For large sigma, all radii should be > 0
        let radii = box_radii_for_sigma(10.0);
        for &r in &radii {
            assert!(r > 0, "radius should be positive for sigma=10");
        }

        // Variance of 3 box blurs should approximate sigma^2
        let sigma = 10.0;
        let radii = box_radii_for_sigma(sigma);
        let total_var: f32 = radii
            .iter()
            .map(|&r| {
                let w = (2 * r + 1) as f32;
                (w * w - 1.0) / 12.0
            })
            .sum();
        let target_var = sigma * sigma;
        let var_err = (total_var - target_var).abs() / target_var;
        assert!(
            var_err < 0.1,
            "variance should approximate sigma^2 within 10%, got err={:.1}%",
            var_err * 100.0
        );
    }

    /// Blur preserves total energy (sum of pixels).
    #[test]
    fn blur_preserves_energy() {
        let size = 32;
        let mut data = vec![0.0f32; size * size];
        // Place some energy away from boundaries
        data[size / 2 * size + size / 2] = 100.0;
        data[size / 3 * size + size / 3] = 50.0;

        let sum_before: f32 = data.iter().sum();
        let mut img = OwnedImage::from_vec(data, size, size).unwrap();
        gaussian_blur_inplace(&mut img, 5.0);
        let sum_after: f32 = img.data().iter().sum();

        let energy_err = (sum_after - sum_before).abs() / sum_before;
        assert!(
            energy_err < 0.01,
            "blur should preserve energy within 1%, got err={:.2}%",
            energy_err * 100.0
        );
    }

    /// Dispatch: small sigma uses direct, large sigma uses box blur.
    /// Both should produce non-zero output for an impulse input.
    #[test]
    fn dispatch_both_paths() {
        for &sigma in &[1.0, 5.0, 15.0] {
            let size = 32;
            let mut data = vec![0.0f32; size * size];
            data[size / 2 * size + size / 2] = 1.0;
            let mut img = OwnedImage::from_vec(data, size, size).unwrap();
            gaussian_blur_inplace(&mut img, sigma);
            let peak = img.data().iter().copied().fold(0.0f32, Scalar::max);
            assert!(peak > 0.0, "sigma={sigma}: peak should be positive");
        }
    }
}
