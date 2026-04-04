//! # radsym
//!
//! Radial symmetry detection library: center proposal generation, local
//! circular and elliptical support analysis, support scoring, and local
//! image-space refinement. CPU-first, deterministic, composable tools.
//!
//! ## Quick start
//!
//! ```rust
//! use radsym::{
//!     ImageView, FrstConfig, Circle, Polarity, ScoringConfig, NmsConfig,
//!     sobel_gradient, frst_response, extract_proposals, score_circle_support,
//! };
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
//! // 2. Compute gradient and FRST response
//! let gradient = sobel_gradient(&image).unwrap();
//! let config = FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() };
//! let response = frst_response(&gradient, &config).unwrap();
//!
//! // 3. Extract proposals via NMS
//! let nms = NmsConfig { radius: 5, threshold: 0.0, max_detections: 5 };
//! let proposals = extract_proposals(&response, &nms, Polarity::Bright);
//!
//! // 4. Score the best proposal
//! if let Some(best) = proposals.first() {
//!     let circle = Circle::new(best.seed.position, 10.0);
//!     let score = score_circle_support(&gradient, &circle, &ScoringConfig::default());
//!     assert!(score.total > 0.0);
//! }
//! ```
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

// Re-export the most commonly used types at crate root.

// Core types
pub use crate::core::circle_fit::{fit_circle, fit_circle_weighted};
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
pub use crate::core::pyramid::{
    OwnedPyramidLevel, PyramidLevelView, PyramidWorkspace, pyramid_level_owned,
};

// Proposal generation
pub use crate::propose::extract::{ResponseMap, extract_proposals, suppress_proposals_by_distance};
pub use crate::propose::frst::{
    FrstConfig, frst_response, frst_response_single, multiradius_response,
};
pub use crate::propose::homography::{
    HomographyProposal, HomographyRerankConfig, RectifiedResponseMap, RerankedProposal,
    extract_rectified_proposals, frst_response_homography, rerank_proposals_homography,
};
pub use crate::propose::remap::{remap_proposal_to_image, remap_proposals_to_image};
pub use crate::propose::rsd::{RsdConfig, rsd_response, rsd_response_fused};
pub use crate::propose::seed::{Proposal, ProposalSource, SeedPoint};

// Support scoring
pub use crate::support::annulus::AnnulusSamplingConfig;
pub use crate::support::evidence::SupportEvidence;
pub use crate::support::hypothesis::{
    AnnulusHypothesis, CircleHypothesis, ConcentricPairHypothesis, EllipseHypothesis,
};
pub use crate::support::score::{
    ScoringConfig, SupportScore, score_circle_support, score_ellipse_support,
    score_rectified_circle_support,
};

// Refinement
pub use crate::refine::circle::{CircleRefineConfig, refine_circle};
pub use crate::refine::ellipse::{EllipseRefineConfig, refine_ellipse};
pub use crate::refine::homography::{
    HomographyEllipseRefineConfig, HomographyRefinementResult, refine_ellipse_homography,
};
pub use crate::refine::radial_center::{
    RadialCenterConfig, radial_center_refine, radial_center_refine_from_gradient,
};
pub use crate::refine::result::{RefinementResult, RefinementStatus};

// Pipeline
pub use crate::pipeline::{DetectCirclesConfig, Detection, detect_circles};

// Diagnostics
pub use crate::diagnostics::heatmap::{Colormap, DiagnosticImage, response_heatmap};
pub use crate::diagnostics::overlay::{overlay_circle, overlay_ellipse};

// I/O (feature-gated)
#[cfg(feature = "image-io")]
pub use crate::core::io::load_grayscale;
#[cfg(feature = "image-io")]
pub use crate::diagnostics::export::{save_diagnostic, save_grayscale, save_response_map};
