//! Error types for the `radsym` library.

/// Errors that can occur in `radsym` operations.
#[derive(thiserror::Error, Debug, Clone)]
#[non_exhaustive]
pub enum RadSymError {
    /// Image dimensions are invalid (zero or incompatible).
    #[error("invalid dimensions: width={width} height={height}")]
    InvalidDimensions {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },

    /// Provided buffer is too small for the given dimensions and stride.
    #[error("buffer too small: needed={needed} got={got}")]
    BufferTooSmall {
        /// Minimum required buffer length.
        needed: usize,
        /// Actual buffer length.
        got: usize,
    },

    /// Stride is smaller than the image width.
    #[error("invalid stride: width={width} stride={stride}")]
    InvalidStride {
        /// Image width.
        width: usize,
        /// Provided stride.
        stride: usize,
    },

    /// Configuration parameter is invalid.
    #[error("invalid config: {reason}")]
    InvalidConfig {
        /// Description of the invalid parameter.
        reason: &'static str,
    },

    /// The hypothesis is degenerate (e.g. zero radius, insufficient support).
    #[error("degenerate hypothesis: {reason}")]
    DegenerateHypothesis {
        /// Description of the degeneracy.
        reason: &'static str,
    },

    /// Refinement failed to converge or produced invalid results.
    #[error("refinement failed: {reason}")]
    RefinementFailed {
        /// Description of the failure.
        reason: &'static str,
    },

    /// Image I/O error (feature-gated: `image-io`).
    #[cfg(feature = "image-io")]
    #[error("image I/O error: {reason}")]
    ImageIo {
        /// Description of the I/O failure.
        reason: String,
    },
}

/// Convenience alias for `std::result::Result<T, RadSymError>`.
pub type Result<T> = std::result::Result<T, RadSymError>;
