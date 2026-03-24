//! Experimental affine/projective-aware extensions.
//!
//! This module provides affine-aware proposal generation for detecting
//! radially symmetric structures under perspective distortion. Based on
//! GFRS (Ni, Singh, Bahlmann, CVPR 2012).
//!
//! Requires the `affine` feature flag.
//!
//! ## Overview
//!
//! - [`transform`] — 2×2 affine map type and parameter sampling
//! - [`propose`] — GFRS-style voting with affine-warped gradient offsets

pub mod propose;
pub mod transform;
