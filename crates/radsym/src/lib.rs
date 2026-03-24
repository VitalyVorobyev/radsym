//! # radsym
//!
//! Radial symmetry detection library: center proposal generation, local
//! circular and elliptical support analysis, support scoring, and local
//! image-space refinement.
//!
//! ## Modules
//!
//! - [`core`] — fundamental types, image views, geometry, gradient, NMS
//! - [`propose`] — center-proposal generation (FRST, RSD)
//! - [`support`] — local support extraction and scoring
//! - [`refine`] — local hypothesis refinement
//! - [`diagnostics`] — visualization and export
//!
//! ## Feature flags
//!
//! - `rayon` — parallel execution for multi-radius proposals
//! - `image-io` — load images via the `image` crate
//! - `tracing` — structured logging
//! - `affine` — experimental affine-aware extensions
//! - `serde` — serialization support

pub mod core;
pub mod diagnostics;
pub mod propose;
pub mod refine;
pub mod support;

#[cfg(feature = "affine")]
pub mod affine;

// Re-export the most commonly used types at crate root.
pub use crate::core::coords::PixelCoord;
pub use crate::core::error::{RadSymError, Result};
pub use crate::core::geometry::{Annulus, Circle, Ellipse};
pub use crate::core::image_view::{ImageView, OwnedImage};
pub use crate::core::polarity::Polarity;
