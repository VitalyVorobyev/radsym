//! Separable Gaussian blur for response maps.

use super::image_view::OwnedImage;
use super::scalar::Scalar;

/// Separable Gaussian blur on an `OwnedImage<f32>`, in-place.
///
/// Uses a 1D Gaussian kernel with `radius = ceil(3 * sigma)` and mirror-clamp
/// boundary handling. No-op if `sigma <= 0.5` (kernel radius would be zero).
pub(crate) fn gaussian_blur_inplace(image: &mut OwnedImage<Scalar>, sigma: Scalar) {
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
