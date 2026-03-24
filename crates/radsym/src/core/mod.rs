//! Fundamental types, image views, geometry, gradient computation, and NMS.
//!
//! This module provides the stable foundation that all algorithm modules
//! depend on. It must remain small and free of algorithm-specific logic.

pub mod coords;
pub mod error;
pub mod geometry;
pub mod gradient;
pub mod image_view;
pub mod nms;
pub mod polarity;
pub mod scalar;
