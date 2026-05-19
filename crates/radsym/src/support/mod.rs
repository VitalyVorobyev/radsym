//! Local support extraction and scoring.
//!
//! This module scores how strongly local image evidence supports a geometric
//! hypothesis. The public entry points are [`score::score_circle_support`] and
//! [`score::score_ellipse_support`]: they sample gradient evidence around a
//! [`Circle`](crate::Circle) or [`Ellipse`](crate::Ellipse) and return a
//! [`score::SupportScore`]. Annulus sampling and angular-coverage estimation
//! are internal stages of that computation.

pub mod annulus;
pub(crate) mod coverage;
pub(crate) mod evidence;
pub mod score;
