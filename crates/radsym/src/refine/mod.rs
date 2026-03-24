//! Local image-space hypothesis refinement.
//!
//! This module provides algorithms for refining seed locations and geometric
//! hypotheses (circles, ellipses) to subpixel accuracy using local image
//! evidence.
//!
//! ## Algorithms
//!
//! - **Radial center** ([`radial_center`]): Non-iterative subpixel center
//!   refinement via weighted least-squares intersection of gradient lines.
//!   Based on Parthasarathy, Nature Methods 2012.
//! - **Circle refinement** ([`circle`]): Iterative refinement of center and
//!   radius using annulus sampling and radial center sub-steps.
//! - **Ellipse refinement** ([`ellipse`]): Iterative refinement of center,
//!   semi-axes, and orientation using elliptical annulus sampling.

pub mod circle;
pub mod ellipse;
pub mod radial_center;
pub mod result;
