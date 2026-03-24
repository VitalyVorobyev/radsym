//! Center-proposal generation.
//!
//! This module implements gradient-based center voting algorithms that produce
//! candidate locations (seeds/proposals) for radially symmetric structures.
//!
//! ## Algorithms
//!
//! - [`frst`] — Fast Radial Symmetry Transform (Loy & Zelinsky 2002/2003)
//!
//! ## Usage
//!
//! 1. Compute gradient field from image: [`crate::core::gradient::sobel_gradient`]
//! 2. Compute FRST response map: [`frst::frst_response`]
//! 3. Extract proposals: [`extract::extract_proposals`]

pub mod extract;
pub mod frst;
pub mod seed;
