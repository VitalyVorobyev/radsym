//! Feature-gated unchecked slice access for voting hot loops.
//!
//! The center-voting passes read the gradient at `y*w + x` (always in bounds by
//! construction) and scatter-add into an accumulator at an index that a
//! preceding `px/py` range test has already proven `< w*h`. The `[]` operator
//! re-checks that bound on every access, which shows up in the profile for the
//! scatter-heavy inner loop.
//!
//! With the `unsafe-opt` feature these helpers skip the redundant check; without
//! it they are plain checked indexing with **identical** numerical output. Every
//! call site documents why its index is in bounds, and a `debug_assert!` guards
//! the invariant in debug/test builds.

use super::scalar::Scalar;

/// Load `s[i]`.
///
/// # Safety contract (only relevant under `unsafe-opt`)
///
/// The caller must guarantee `i < s.len()`.
#[inline(always)]
pub(crate) fn ld(s: &[Scalar], i: usize) -> Scalar {
    #[cfg(feature = "unsafe-opt")]
    {
        debug_assert!(i < s.len(), "ld index {i} out of bounds (len {})", s.len());
        // SAFETY: the caller proves `i < s.len()` (see the call site).
        unsafe { *s.get_unchecked(i) }
    }
    #[cfg(not(feature = "unsafe-opt"))]
    {
        s[i]
    }
}

/// Scatter-add `s[i] += v`.
///
/// # Safety contract (only relevant under `unsafe-opt`)
///
/// The caller must guarantee `i < s.len()`.
#[inline(always)]
pub(crate) fn add_at(s: &mut [Scalar], i: usize, v: Scalar) {
    #[cfg(feature = "unsafe-opt")]
    {
        debug_assert!(
            i < s.len(),
            "add_at index {i} out of bounds (len {})",
            s.len()
        );
        // SAFETY: the caller proves `i < s.len()` (see the call site).
        unsafe {
            *s.get_unchecked_mut(i) += v;
        }
    }
    #[cfg(not(feature = "unsafe-opt"))]
    {
        s[i] += v;
    }
}
