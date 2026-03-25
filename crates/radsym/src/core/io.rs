//! Image I/O utilities (feature-gated: `image-io`).
//!
//! Provides convenience functions for loading grayscale images from disk.

use std::path::Path;

use super::error::{RadSymError, Result};
use super::image_view::OwnedImage;

/// Load a grayscale image from a file path.
///
/// Supports any format that the [`image`] crate can decode (PNG, JPEG, etc.).
/// The image is converted to 8-bit grayscale (`Luma8`) before wrapping in
/// an [`OwnedImage<u8>`].
///
/// # Errors
///
/// Returns [`RadSymError::ImageIo`] if the file cannot be read or decoded.
pub fn load_grayscale(path: impl AsRef<Path>) -> Result<OwnedImage<u8>> {
    let path = path.as_ref();
    let img = image::open(path).map_err(|e| RadSymError::ImageIo {
        reason: format!("{path}: {e}", path = path.display()),
    })?;
    let gray = img.into_luma8();
    let (w, h) = gray.dimensions();
    let data = gray.into_raw();
    OwnedImage::from_vec(data, w as usize, h as usize)
}
