//! Image export utilities (feature-gated: `image-io`).
//!
//! Provides functions for saving diagnostic and grayscale images to disk.

use std::path::Path;

use crate::core::error::{RadSymError, Result};
use crate::core::image_view::OwnedImage;
use crate::core::scalar::Scalar;

use super::heatmap::DiagnosticImage;

/// Save a [`DiagnosticImage`] (RGBA) to a file.
///
/// The output format is inferred from the file extension (`.png`, `.jpg`, etc.).
///
/// # Errors
///
/// Returns [`RadSymError::ImageIo`] if the file cannot be written.
pub fn save_diagnostic(img: &DiagnosticImage, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    image::save_buffer(
        path,
        img.data(),
        img.width() as u32,
        img.height() as u32,
        image::ColorType::Rgba8,
    )
    .map_err(|e| RadSymError::ImageIo {
        reason: format!("{path}: {e}", path = path.display()),
    })
}

/// Save an [`OwnedImage<u8>`] (grayscale) to a file.
///
/// The output format is inferred from the file extension.
///
/// # Errors
///
/// Returns [`RadSymError::ImageIo`] if the file cannot be written.
pub fn save_grayscale(img: &OwnedImage<u8>, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    image::save_buffer(
        path,
        img.data(),
        img.width() as u32,
        img.height() as u32,
        image::ColorType::L8,
    )
    .map_err(|e| RadSymError::ImageIo {
        reason: format!("{path}: {e}", path = path.display()),
    })
}

/// Save an [`OwnedImage<Scalar>`] (float response map) to a file.
///
/// Values are normalized to 0–255 range before saving as 8-bit grayscale.
///
/// # Errors
///
/// Returns [`RadSymError::ImageIo`] if the file cannot be written.
pub fn save_response_map(img: &OwnedImage<Scalar>, path: impl AsRef<Path>) -> Result<()> {
    let data = img.data();
    let max = data.iter().copied().fold(0.0f32, Scalar::max);
    let scale = if max > 0.0 { 255.0 / max } else { 0.0 };
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| (v * scale).clamp(0.0, 255.0) as u8)
        .collect();
    let out = OwnedImage::from_vec(bytes, img.width(), img.height())?;
    save_grayscale(&out, path)
}
