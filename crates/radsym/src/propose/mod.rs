//! Center-proposal generation.
//!
//! This module implements gradient-based center voting algorithms that produce
//! candidate locations (seeds/proposals) for radially symmetric structures.
//!
//! ## Algorithms
//!
//! - [`frst`] — Fast Radial Symmetry Transform (Loy & Zelinsky 2002/2003)
//! - [`rsd`] — Radial Symmetry Detector, fast magnitude-only variant
//!   (Barnes, Zelinsky, Fletcher 2008)
//!
//! ## Usage
//!
//! 1. Compute gradient field from image: [`crate::core::gradient::sobel_gradient`]
//! 2. Compute response map: [`frst::frst_response`] or [`rsd::rsd_response`]
//! 3. Extract proposals: [`extract::extract_proposals`]

pub mod extract;
pub mod frst;
pub mod homography;
pub mod rsd;
pub mod seed;
