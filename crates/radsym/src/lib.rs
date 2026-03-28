//! # radsym
//!
//! Radial symmetry detection library: center proposal generation, local
//! circular and elliptical support analysis, support scoring, and local
//! image-space refinement. CPU-first, deterministic, composable tools.
//!
//! ## Quick start
//!
//! ```rust
//! use radsym::{ImageView, FrstConfig, Circle, PixelCoord, Polarity, ScoringConfig};
//! use radsym::core::gradient::sobel_gradient;
//! use radsym::propose::extract::{extract_proposals, ResponseMap};
//! use radsym::propose::seed::ProposalSource;
//! use radsym::core::nms::NmsConfig;
//! use radsym::support::score::score_circle_support;
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
//! let response = radsym::frst_response(&gradient, &config).unwrap();
//!
//! // 3. Extract proposals via NMS
//! let response_map = ResponseMap::new(response, ProposalSource::Frst);
//! let nms = NmsConfig { radius: 5, threshold: 0.0, max_detections: 5 };
//! let proposals = extract_proposals(&response_map, &nms, Polarity::Bright);
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
pub mod propose;
pub mod refine;
pub mod support;

#[cfg(feature = "affine")]
pub mod affine;

// Re-export the most commonly used types at crate root.
pub use crate::core::circle_fit::{fit_circle, fit_circle_weighted};
pub use crate::core::coords::PixelCoord;
pub use crate::core::error::{RadSymError, Result};
pub use crate::core::geometry::{Annulus, Circle, Ellipse};
pub use crate::core::homography::{rectified_circle_to_image_ellipse, Homography, RectifiedGrid};
pub use crate::core::image_view::{ImageView, OwnedImage};
pub use crate::core::polarity::Polarity;
pub use crate::propose::extract::{extract_proposals, suppress_proposals_by_distance, ResponseMap};
pub use crate::propose::frst::{frst_response, frst_response_single, FrstConfig};
pub use crate::propose::homography::{
    extract_rectified_proposals, frst_response_homography, rerank_proposals_homography,
    HomographyProposal, HomographyRerankConfig, RectifiedResponseMap, RerankedProposal,
};
pub use crate::propose::rsd::{rsd_response, RsdConfig};
pub use crate::propose::seed::{Proposal, ProposalSource, SeedPoint};
pub use crate::support::evidence::SupportEvidence;
pub use crate::support::hypothesis::{
    AnnulusHypothesis, CircleHypothesis, ConcentricPairHypothesis, EllipseHypothesis,
};
pub use crate::support::score::{score_rectified_circle_support, ScoringConfig, SupportScore};

pub use crate::refine::circle::{refine_circle, CircleRefineConfig};
pub use crate::refine::ellipse::{refine_ellipse, EllipseRefineConfig};
pub use crate::refine::homography::{
    refine_ellipse_homography, HomographyEllipseRefineConfig, HomographyRefinementResult,
};
pub use crate::refine::radial_center::{
    radial_center_refine, radial_center_refine_from_gradient, RadialCenterConfig,
};
pub use crate::refine::result::{RefinementResult, RefinementStatus};

#[cfg(feature = "image-io")]
pub use crate::core::io::load_grayscale;
#[cfg(feature = "image-io")]
pub use crate::diagnostics::export::{save_diagnostic, save_grayscale, save_response_map};
