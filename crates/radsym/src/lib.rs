//! # radsym
//!
//! Radial symmetry detection library: center proposal generation, local
//! circular and elliptical support analysis, support scoring, and local
//! image-space refinement. CPU-first, deterministic, composable tools.
//!
//! ## Quick start
//!
//! The one-call [`detect_circles`] runs the whole propose-score-refine
//! pipeline. Configure it with the [`DetectCirclesConfig::for_radii`] builder:
//!
//! ```rust
//! use radsym::{detect_circles, DetectCirclesConfig, ImageView, Polarity};
//!
//! // 1. Load or create an image (here: synthetic bright disk)
//! let size = 64;
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
//! // 2. Detect circles in a single call.
//! let config = DetectCirclesConfig::for_radii([9, 10, 11])
//!     .polarity(Polarity::Bright)
//!     .radius_hint(10.0)
//!     .min_score(0.2);
//! let detections = detect_circles(&image, &config).unwrap();
//!
//! // 3. Inspect the results (sorted by descending support score).
//! for detection in &detections {
//!     println!(
//!         "center=({:.1}, {:.1}) r={:.1} score={:.3}",
//!         detection.hypothesis.center.x,
//!         detection.hypothesis.center.y,
//!         detection.hypothesis.radius,
//!         detection.score.total,
//!     );
//! }
//! assert!(!detections.is_empty());
//! ```
//!
//! Power users can still drive the propose-score-refine stages directly with
//! [`compute_gradient`], [`frst_response`], [`extract_proposals`],
//! [`score_circle_support`], and [`refine_circle`]. To inspect why proposals
//! were accepted or rejected, use [`detect_circles_with_diagnostics`].
//!
//! ## Modules
//!
//! - [`core`] — fundamental types, image views, geometry, gradient, NMS,
//!   homography, circle fitting
//! - [`propose`] — center-proposal generation (FRST, RSD, homography-aware)
//! - [`support`] — local support extraction and scoring
//! - [`refine`] — local hypothesis refinement (Parthasarathy radial center,
//!   iterative circle/ellipse, homography-aware ellipse)
//! - [`diagnostics`] — visualization: heatmaps, overlays
//!
//! ## Feature flags
//!
//! - `rayon` — parallel execution for multi-radius proposals
//! - `image-io` — load images via the `image` crate
//! - `tracing` — structured logging
//! - `affine` — experimental affine-aware extensions (GFRS)
//! - `serde` — serialization support

pub mod core;
pub mod diagnostics;
pub mod pipeline;
pub mod prelude;
pub mod propose;
pub mod refine;
pub mod support;

#[cfg(feature = "affine")]
pub mod affine;

// Re-export the most commonly used types at crate root. The root facade is
// intentionally small: it carries the result/config/stage contract for the
// common detect-score-refine workflow. Low-level helpers, diagnostics, pyramid
// tooling, and algorithm scaffolding are reached through their module paths.

// Core types
pub use crate::core::coords::PixelCoord;
pub use crate::core::error::{RadSymError, Result};
pub use crate::core::geometry::{Annulus, Circle, Ellipse};
pub use crate::core::gradient::{
    GradientField, GradientOperator, compute_gradient, compute_gradient_f32, scharr_gradient,
    scharr_gradient_f32, sobel_gradient, sobel_gradient_f32,
};
pub use crate::core::homography::{Homography, RectifiedGrid, rectified_circle_to_image_ellipse};
pub use crate::core::image_view::{ImageView, OwnedImage};
pub use crate::core::nms::NmsConfig;
pub use crate::core::polarity::Polarity;

// Proposal generation
pub use crate::propose::extract::{ResponseMap, extract_proposals, suppress_proposals_by_distance};
pub use crate::propose::frst::{FrstConfig, frst_response, frst_response_fused};
pub use crate::propose::homography::{
    HomographyProposal, HomographyRerankAdvanced, HomographyRerankConfig, RectifiedResponseMap,
    RerankedProposal, extract_rectified_proposals, frst_response_homography,
    rerank_proposals_homography,
};
pub use crate::propose::remap::{remap_proposal_to_image, remap_proposals_to_image};
pub use crate::propose::rsd::{RsdConfig, rsd_response, rsd_response_fused};
pub use crate::propose::seed::{Proposal, ProposalSource, SeedPoint};

// Support scoring
pub use crate::support::annulus::AnnulusSamplingConfig;
pub use crate::support::score::{
    ScoringConfig, SupportScore, score_circle_support, score_ellipse_support,
    score_rectified_circle_support,
};

// Refinement
pub use crate::refine::circle::{CircleRefineAdvanced, CircleRefineConfig, refine_circle};
pub use crate::refine::ellipse::{EllipseRefineAdvanced, EllipseRefineConfig, refine_ellipse};
pub use crate::refine::homography::{
    HomographyEllipseRefineAdvanced, HomographyEllipseRefineConfig, HomographyRefinementResult,
    refine_ellipse_homography,
};
pub use crate::refine::radial_center::{
    RadialCenterConfig, radial_center_refine, radial_center_refine_from_gradient,
};
pub use crate::refine::result::{RefinementResult, RefinementStatus};

// Pipeline
pub use crate::pipeline::{
    CircleDetection, DetectCirclesAdvanced, DetectCirclesConfig, Detection, detect_circles,
    detect_circles_with_diagnostics,
};

// I/O (feature-gated)
#[cfg(feature = "image-io")]
pub use crate::core::io::load_grayscale;
#[cfg(feature = "image-io")]
pub use crate::diagnostics::export::{save_diagnostic, save_grayscale, save_response_map};
