//! Visualization and diagnostic export.
//!
//! This module provides tools for generating heatmaps, overlays, and data
//! exports to inspect and debug algorithm behavior.
//!
//! ## Components
//!
//! - [`heatmap`] — response map to RGBA colormap rendering
//! - [`overlay`] — draw circles, ellipses, markers onto diagnostic images
//! - [`detection`] — diagnostic evidence from the circle-detection pipeline

pub mod detection;
#[cfg(feature = "image-io")]
pub mod export;
pub mod heatmap;
pub mod overlay;

pub use crate::support::score::SupportScoreBreakdown;
pub use detection::{CircleDetectionDiagnostics, RejectedProposal, RejectionReason};
pub use heatmap::{Colormap, DiagnosticImage, response_heatmap};
pub use overlay::{overlay_circle, overlay_ellipse, overlay_marker, overlay_proposals};
