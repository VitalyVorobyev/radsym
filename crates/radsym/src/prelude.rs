//! Convenience re-exports for common workflows.
//!
//! ```rust
//! use radsym::prelude::*;
//! ```
//!
//! This imports the types and functions needed for the standard
//! propose-score-refine pipeline without requiring individual imports.
//!
//! ## Smoke test: detect-score-refine via prelude
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
//! // Propose
//! let gradient = sobel_gradient(&image).unwrap();
//! let frst_cfg = FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() };
//! let response = frst_response(&gradient, &frst_cfg).unwrap();
//! let nms = NmsConfig { radius: 5, threshold: 0.0, max_detections: 5 };
//! let proposals = extract_proposals(&response, &nms, Polarity::Bright);
//! assert!(!proposals.is_empty(), "should find at least one proposal");
//!
//! // Score
//! let best = &proposals[0];
//! let circle = Circle::new(best.seed.position, 10.0);
//! let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
//! assert!(score.total > 0.0, "support score should be positive");
//!
//! // Refine
//! let result = refine_circle(&gradient, &circle, &CircleRefineConfig::default()).unwrap();
//! let c = result.hypothesis.center;
//! let err = ((c.x - 32.0).powi(2) + (c.y - 32.0).powi(2)).sqrt();
//! assert!(err < 5.0, "refined center should be near (32, 32)");
//! ```

// Core types
pub use crate::core::coords::PixelCoord;
pub use crate::core::error::{RadSymError, Result};
pub use crate::core::geometry::{Annulus, Circle, Ellipse};
pub use crate::core::gradient::{
    compute_gradient, scharr_gradient, sobel_gradient, GradientField, GradientOperator,
};
pub use crate::core::image_view::{ImageView, OwnedImage};
pub use crate::core::nms::NmsConfig;
pub use crate::core::polarity::Polarity;

// Proposal generation
pub use crate::propose::extract::{extract_proposals, ResponseMap};
pub use crate::propose::frst::{frst_response, multiradius_response, FrstConfig};
pub use crate::propose::rsd::{rsd_response, rsd_response_fused, RsdConfig};
pub use crate::propose::seed::Proposal;

// Support scoring
pub use crate::support::score::{
    score_circle_support, score_ellipse_support, ScoringConfig, SupportScore,
};

// Refinement
pub use crate::refine::circle::{refine_circle, CircleRefineConfig};
pub use crate::refine::ellipse::{refine_ellipse, EllipseRefineConfig};
pub use crate::refine::result::{RefinementResult, RefinementStatus};

// Pipeline
pub use crate::pipeline::{detect_circles, DetectCirclesConfig, Detection};
