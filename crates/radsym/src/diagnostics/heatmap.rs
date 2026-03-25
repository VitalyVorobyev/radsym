//! Response map heatmap generation.
//!
//! Converts floating-point response maps into RGBA color images for
//! visualization, using a configurable colormap.

use crate::core::image_view::OwnedImage;
use crate::core::scalar::Scalar;

/// An RGBA diagnostic image (4 bytes per pixel: R, G, B, A).
#[derive(Debug, Clone)]
pub struct DiagnosticImage {
    /// RGBA pixel data, row-major.
    pub(crate) data: Vec<u8>,
    /// Image width in pixels.
    width: usize,
    /// Image height in pixels.
    height: usize,
}

impl DiagnosticImage {
    /// Create a new diagnostic image filled with transparent black.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0u8; width * height * 4],
            width,
            height,
        }
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The raw RGBA pixel data.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume the image, returning the raw RGBA pixel data.
    #[inline]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Set a pixel at (x, y) to the given RGBA color.
    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, rgba: [u8; 4]) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 4;
            self.data[idx..idx + 4].copy_from_slice(&rgba);
        }
    }

    /// Get the RGBA color at (x, y).
    #[inline]
    pub fn get_pixel(&self, x: usize, y: usize) -> [u8; 4] {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 4;
            [
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
                self.data[idx + 3],
            ]
        } else {
            [0, 0, 0, 0]
        }
    }
}

/// Available colormaps for heatmap rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Colormap {
    /// Blue → cyan → green → yellow → red (jet-like).
    Jet,
    /// Black → red → yellow → white (hot-like).
    Hot,
    /// Black → blue → magenta → white (magma-like).
    Magma,
}

/// Map a normalized value `t` in `[0, 1]` to an RGBA color using the given colormap.
fn colormap_value(t: Scalar, cmap: Colormap) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = match cmap {
        Colormap::Jet => {
            let r = (1.5 - (t - 0.75).abs() * 4.0).clamp(0.0, 1.0);
            let g = (1.5 - (t - 0.5).abs() * 4.0).clamp(0.0, 1.0);
            let b = (1.5 - (t - 0.25).abs() * 4.0).clamp(0.0, 1.0);
            (r, g, b)
        }
        Colormap::Hot => {
            let r = (t * 3.0).clamp(0.0, 1.0);
            let g = ((t - 0.333) * 3.0).clamp(0.0, 1.0);
            let b = ((t - 0.666) * 3.0).clamp(0.0, 1.0);
            (r, g, b)
        }
        Colormap::Magma => {
            let r = (t * 2.0).clamp(0.0, 1.0);
            let g = ((t - 0.5) * 2.0).clamp(0.0, 1.0);
            let b = (t * 1.5).clamp(0.0, 1.0);
            (r, g, b)
        }
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
}

/// Render a response map as a heatmap image.
///
/// Maps response values to colors using the specified colormap. Values
/// are normalized to `[0, 1]` based on the response range.
pub fn response_heatmap(response: &OwnedImage<Scalar>, colormap: Colormap) -> DiagnosticImage {
    let w = response.width();
    let h = response.height();
    let data = response.data();

    let max = data.iter().copied().fold(0.0f32, Scalar::max);
    let min = data.iter().copied().fold(Scalar::INFINITY, Scalar::min);
    let range = (max - min).max(1e-8);

    let mut img = DiagnosticImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let val = data[y * w + x];
            let t = (val - min) / range;
            img.set_pixel(x, y, colormap_value(t, colormap));
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_image_set_get() {
        let mut img = DiagnosticImage::new(4, 4);
        img.set_pixel(1, 2, [255, 0, 128, 255]);
        assert_eq!(img.get_pixel(1, 2), [255, 0, 128, 255]);
        assert_eq!(img.get_pixel(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn heatmap_dimensions() {
        let response = OwnedImage::from_vec(vec![0.0f32; 16], 4, 4).unwrap();
        let hm = response_heatmap(&response, Colormap::Jet);
        assert_eq!(hm.width(), 4);
        assert_eq!(hm.height(), 4);
        assert_eq!(hm.data().len(), 4 * 4 * 4);
    }

    #[test]
    fn heatmap_gradient_produces_varying_colors() {
        let data: Vec<Scalar> = (0..100).map(|i| i as Scalar / 99.0).collect();
        let response = OwnedImage::from_vec(data, 10, 10).unwrap();
        let hm = response_heatmap(&response, Colormap::Hot);

        let lo = hm.get_pixel(0, 0);
        let hi = hm.get_pixel(9, 9);
        // Low value should be darker than high value
        let lo_sum: u32 = lo[0] as u32 + lo[1] as u32 + lo[2] as u32;
        let hi_sum: u32 = hi[0] as u32 + hi[1] as u32 + hi[2] as u32;
        assert!(hi_sum > lo_sum, "high value should be brighter");
    }
}
