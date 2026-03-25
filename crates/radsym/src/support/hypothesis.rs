//! Geometric hypothesis types for local support analysis.
//!
//! These types pair geometric primitives with a confidence score and are the
//! primary inputs to support extraction, scoring, and refinement.

use crate::core::geometry::{Annulus, Circle, Ellipse};
use crate::core::scalar::Scalar;

/// A circle hypothesis with confidence.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CircleHypothesis {
    /// The circle geometry.
    pub circle: Circle,
    /// Confidence score from proposal or prior stage.
    pub confidence: Scalar,
}

impl CircleHypothesis {
    /// Create a new circle hypothesis.
    #[inline]
    pub fn new(circle: Circle, confidence: Scalar) -> Self {
        Self { circle, confidence }
    }
}

/// An ellipse hypothesis with confidence.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EllipseHypothesis {
    /// The ellipse geometry.
    pub ellipse: Ellipse,
    /// Confidence score.
    pub confidence: Scalar,
}

impl EllipseHypothesis {
    /// Create a new ellipse hypothesis.
    #[inline]
    pub fn new(ellipse: Ellipse, confidence: Scalar) -> Self {
        Self {
            ellipse,
            confidence,
        }
    }
}

impl From<CircleHypothesis> for EllipseHypothesis {
    fn from(ch: CircleHypothesis) -> Self {
        Self {
            ellipse: ch.circle.into(),
            confidence: ch.confidence,
        }
    }
}

/// An annulus (ring) hypothesis with confidence.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnnulusHypothesis {
    /// The annulus geometry.
    pub annulus: Annulus,
    /// Confidence score.
    pub confidence: Scalar,
}

impl AnnulusHypothesis {
    /// Create a new annulus hypothesis.
    #[inline]
    pub fn new(annulus: Annulus, confidence: Scalar) -> Self {
        Self {
            annulus,
            confidence,
        }
    }
}

/// A concentric pair hypothesis: two ellipses sharing a center.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConcentricPairHypothesis {
    /// Inner ellipse.
    pub inner: Ellipse,
    /// Outer ellipse.
    pub outer: Ellipse,
    /// Confidence score.
    pub confidence: Scalar,
}

impl ConcentricPairHypothesis {
    /// Create a new concentric pair hypothesis.
    #[inline]
    pub fn new(inner: Ellipse, outer: Ellipse, confidence: Scalar) -> Self {
        Self {
            inner,
            outer,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::coords::PixelCoord;

    #[test]
    fn circle_to_ellipse_hypothesis() {
        let c = Circle::new(PixelCoord::new(10.0, 20.0), 5.0);
        let ch = CircleHypothesis::new(c, 0.9);
        let eh: EllipseHypothesis = ch.into();
        assert_eq!(eh.ellipse.semi_major, 5.0);
        assert_eq!(eh.ellipse.semi_minor, 5.0);
        assert_eq!(eh.confidence, 0.9);
    }

    #[test]
    fn annulus_hypothesis() {
        let a = Annulus::new(PixelCoord::new(0.0, 0.0), 5.0, 10.0);
        let ah = AnnulusHypothesis::new(a, 0.8);
        assert_eq!(ah.annulus.thickness(), 5.0);
        assert_eq!(ah.confidence, 0.8);
    }

    #[test]
    fn concentric_pair() {
        let inner = Ellipse::new(PixelCoord::new(50.0, 50.0), 10.0, 8.0, 0.0);
        let outer = Ellipse::new(PixelCoord::new(50.0, 50.0), 20.0, 16.0, 0.0);
        let cp = ConcentricPairHypothesis::new(inner, outer, 0.75);
        assert_eq!(cp.inner.semi_major, 10.0);
        assert_eq!(cp.outer.semi_major, 20.0);
    }
}
