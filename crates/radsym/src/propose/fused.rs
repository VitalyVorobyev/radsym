//! Shared fused multi-radius voting pass used by both FRST and RSD fused variants.

use crate::core::error::Result;
use crate::core::gradient::GradientField;
use crate::core::image_view::OwnedImage;
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;

/// Compute the median of a non-empty slice of radii.
pub(crate) fn compute_median_radius(radii: &[u32]) -> Scalar {
    let mut sorted = radii.to_vec();
    sorted.sort_unstable();
    let len = sorted.len();
    if len % 2 == 1 {
        sorted[len / 2] as Scalar
    } else {
        (sorted[len / 2 - 1] as Scalar + sorted[len / 2] as Scalar) * 0.5
    }
}

/// Fused multi-radius magnitude-only voting pass.
///
/// Iterates over all pixels once, voting at each configured radius into a
/// single shared accumulator. Applies one Gaussian blur at the end with
/// `sigma = smoothing_factor * median(radii)`.
pub(crate) fn fused_voting_pass(
    gradient: &GradientField,
    radii: &[u32],
    gradient_threshold: Scalar,
    polarity: Polarity,
    smoothing_factor: Scalar,
) -> Result<OwnedImage<Scalar>> {
    let w = gradient.width();
    let h = gradient.height();

    let mut accumulator = OwnedImage::<Scalar>::zeros(w, h)?;
    let acc_data = accumulator.data_mut();
    let gx_data = gradient.gx.data();
    let gy_data = gradient.gy.data();

    let thresh_sq = gradient_threshold * gradient_threshold;
    let vote_pos = polarity.votes_positive();
    let vote_neg = polarity.votes_negative();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let gx = gx_data[idx];
            let gy = gy_data[idx];
            let mag_sq = gx * gx + gy * gy;
            if mag_sq < thresh_sq {
                continue;
            }

            let mag = mag_sq.sqrt();
            let inv_mag = mag.recip();
            let dx = gx * inv_mag;
            let dy = gy * inv_mag;

            for &radius in radii {
                let n = radius as Scalar;
                let offset_x = (dx * n).round() as isize;
                let offset_y = (dy * n).round() as isize;

                if vote_pos {
                    let px = x as isize + offset_x;
                    let py = y as isize + offset_y;
                    if px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h {
                        acc_data[py as usize * w + px as usize] += mag;
                    }
                }

                if vote_neg {
                    let px = x as isize - offset_x;
                    let py = y as isize - offset_y;
                    if px >= 0 && (px as usize) < w && py >= 0 && (py as usize) < h {
                        acc_data[py as usize * w + px as usize] += mag;
                    }
                }
            }
        }
    }

    let median_radius = compute_median_radius(radii);
    let sigma = smoothing_factor * median_radius;
    if sigma > 0.5 {
        crate::core::blur::gaussian_blur_inplace(&mut accumulator, sigma);
    }

    Ok(accumulator)
}
