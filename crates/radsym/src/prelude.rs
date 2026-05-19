//! Convenience re-exports for common workflows.
//!
//! ```rust
//! use radsym::prelude::*;
//! ```
//!
//! This imports the types and functions needed for the standard
//! propose-score-refine pipeline without requiring individual imports.
//!
//! ## Smoke test: one-call detection via prelude
//!
//! ```rust
//! use radsym::prelude::*;
//!
//! // Create a 64x64 synthetic bright disk image.
//! let size = 64usize;
//! let mut data = vec![0u8; size * size];
//! for y in 0..size {
//!     for x in 0..size {
//!         let dx = x as f32 - 32.0;
//!         let dy = y as f32 - 32.0;
//!         if (dx * dx + dy * dy).sqrt() <= 10.0 {
//!             data[y * size + x] = 255;
//!         }
//!     }
//! }
//! let image = ImageView::from_slice(&data, size, size).unwrap();
//!
//! // Detect circles with the one-call pipeline.
//! let config = DetectCirclesConfig::for_radii([9, 10, 11]).polarity(Polarity::Bright);
//! let detections = detect_circles(&image, &config).unwrap();
//! assert!(!detections.is_empty(), "should detect the synthetic disk");
//!
//! // The best detection's refined center sits near (32, 32).
//! let best = &detections[0];
//! let c = best.hypothesis.center;
//! let err = ((c.x - 32.0).powi(2) + (c.y - 32.0).powi(2)).sqrt();
//! assert!(err < 5.0, "refined center should be near (32, 32)");
//! assert!(best.score.total > 0.0, "support score should be positive");
//! ```
//!
//! The prelude also re-exports the individual stage entry points
//! ([`sobel_gradient`], [`frst_response`], [`extract_proposals`],
//! [`score_circle_support`], [`refine_circle`]) for the composable workflow.

// Core types
pub use crate::core::coords::PixelCoord;
pub use crate::core::error::{RadSymError, Result};
pub use crate::core::geometry::{Annulus, Circle, Ellipse};
pub use crate::core::gradient::{
    GradientField, GradientOperator, compute_gradient, scharr_gradient, sobel_gradient,
};
pub use crate::core::image_view::{ImageView, OwnedImage};
pub use crate::core::nms::NmsConfig;
pub use crate::core::polarity::Polarity;

// Proposal generation
pub use crate::propose::extract::{ResponseMap, extract_proposals};
pub use crate::propose::frst::{FrstConfig, frst_response, frst_response_fused};
pub use crate::propose::rsd::{RsdConfig, rsd_response, rsd_response_fused};
pub use crate::propose::seed::Proposal;

// Support scoring
pub use crate::support::score::{
    ScoringConfig, SupportScore, score_circle_support, score_ellipse_support,
};

// Refinement
pub use crate::refine::circle::{CircleRefineConfig, refine_circle};
pub use crate::refine::ellipse::{EllipseRefineConfig, refine_ellipse};
pub use crate::refine::result::{RefinementResult, RefinementStatus};

// Pipeline
pub use crate::pipeline::{CircleDetection, DetectCirclesConfig, Detection, detect_circles};
