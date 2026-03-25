//! Visualization and diagnostic export.
//!
//! This module provides tools for generating heatmaps, overlays, and data
//! exports to inspect and debug algorithm behavior.
//!
//! ## Components
//!
//! - [`heatmap`] — response map to RGBA colormap rendering
//! - [`overlay`] — draw circles, ellipses, markers onto diagnostic images

#[cfg(feature = "image-io")]
pub mod export;
pub mod heatmap;
pub mod overlay;
