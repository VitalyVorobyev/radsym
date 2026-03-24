//! Local support extraction and scoring.
//!
//! This module provides tools for sampling local image evidence around a
//! hypothesis (circle, ellipse, annulus) and scoring how strongly that
//! evidence supports the hypothesis.
//!
//! ## Workflow
//!
//! 1. Create a hypothesis: [`hypothesis::CircleHypothesis`] or [`hypothesis::EllipseHypothesis`]
//! 2. Sample evidence: [`annulus::sample_annulus`] or [`annulus::sample_elliptical_annulus`]
//! 3. Score support: [`score::score_circle_support`] or [`score::score_ellipse_support`]
//! 4. Inspect profiles: [`profile::compute_radial_profile`]
//! 5. Check coverage: [`coverage::angular_coverage`]

pub mod annulus;
pub mod coverage;
pub mod evidence;
pub mod hypothesis;
pub mod profile;
pub mod score;
