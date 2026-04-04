//! Generic access to box-filter pyramid levels and image-frame remapping.

use box_image_pyramid::{
    ImageView as PyramidImageView, PyramidBuffers as PyramidBuffersImpl, PyramidParams,
    build_pyramid,
};

use crate::core::coords::PixelCoord;
use crate::core::error::{RadSymError, Result};
use crate::core::geometry::{Circle, Ellipse};
use crate::core::image_view::{ImageView, OwnedImage};
use crate::core::scalar::Scalar;

/// Reusable workspace for repeated pyramid-level extraction.
///
/// Construct this once and reuse it across frames to avoid reallocating
/// intermediate downsample buffers.
#[derive(Default)]
pub struct PyramidWorkspace {
    buffers: PyramidBuffersImpl,
    contiguous_base: Vec<u8>,
}

impl PyramidWorkspace {
    /// Create an empty reusable pyramid workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a workspace with storage reserved for `num_levels`.
    #[must_use]
    pub fn with_capacity(num_levels: u8) -> Self {
        Self {
            buffers: PyramidBuffersImpl::with_capacity(num_levels),
            contiguous_base: Vec::new(),
        }
    }

    /// Extract a borrowed pyramid level from `image`.
    ///
    /// `level = 0` returns the input image view unchanged. Higher levels apply
    /// repeated 2x box downsampling using `box-image-pyramid`.
    pub fn level<'a>(
        &'a mut self,
        image: ImageView<'a, u8>,
        level: u8,
    ) -> Result<PyramidLevelView<'a>> {
        validate_level(level)?;
        if level == 0 {
            return Ok(PyramidLevelView { image, level });
        }

        let base = prepare_base_view(&mut self.contiguous_base, image)?;
        let mut params = PyramidParams::default();
        params.num_levels = level.saturating_add(1);
        params.min_size = 1;
        let pyramid = build_pyramid(base, &params, &mut self.buffers);
        let level_image = pyramid
            .levels
            .get(level as usize)
            .ok_or(RadSymError::InvalidConfig {
                reason: "pyramid level exceeds image size",
            })?
            .img;

        Ok(PyramidLevelView {
            image: ImageView::from_slice(level_image.data, level_image.width, level_image.height)?,
            level,
        })
    }
}

/// Borrowed view of a single pyramid level.
#[derive(Clone, Copy)]
pub struct PyramidLevelView<'a> {
    image: ImageView<'a, u8>,
    level: u8,
}

impl PyramidLevelView<'_> {
    /// Pyramid level index, where `0` is the base image.
    #[inline]
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Integer scale factor from this level back to the base image.
    #[inline]
    pub fn factor(&self) -> usize {
        checked_level_factor(self.level).expect("validated pyramid level")
    }

    /// Borrow the working image for this level.
    #[inline]
    pub fn image(&self) -> ImageView<'_, u8> {
        self.image
    }

    /// Map a point from working-image coordinates to base-image coordinates.
    #[inline]
    pub fn map_point_to_image(&self, point: PixelCoord) -> PixelCoord {
        map_point_to_image_level(point, self.level)
    }

    /// Map a circle from working-image coordinates to base-image coordinates.
    #[inline]
    pub fn map_circle_to_image(&self, circle: Circle) -> Circle {
        map_circle_to_image_level(circle, self.level)
    }

    /// Map an ellipse from working-image coordinates to base-image coordinates.
    #[inline]
    pub fn map_ellipse_to_image(&self, ellipse: Ellipse) -> Ellipse {
        map_ellipse_to_image_level(ellipse, self.level)
    }
}

/// Owned image for a single pyramid level.
#[derive(Debug, Clone)]
pub struct OwnedPyramidLevel {
    image: OwnedImage<u8>,
    level: u8,
}

impl OwnedPyramidLevel {
    /// Pyramid level index, where `0` is the base image.
    #[inline]
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Integer scale factor from this level back to the base image.
    #[inline]
    pub fn factor(&self) -> usize {
        checked_level_factor(self.level).expect("validated pyramid level")
    }

    /// Borrow the working image for this level.
    #[inline]
    pub fn image(&self) -> ImageView<'_, u8> {
        self.image.view()
    }

    /// Map a point from working-image coordinates to base-image coordinates.
    #[inline]
    pub fn map_point_to_image(&self, point: PixelCoord) -> PixelCoord {
        map_point_to_image_level(point, self.level)
    }

    /// Map a circle from working-image coordinates to base-image coordinates.
    #[inline]
    pub fn map_circle_to_image(&self, circle: Circle) -> Circle {
        map_circle_to_image_level(circle, self.level)
    }

    /// Map an ellipse from working-image coordinates to base-image coordinates.
    #[inline]
    pub fn map_ellipse_to_image(&self, ellipse: Ellipse) -> Ellipse {
        map_ellipse_to_image_level(ellipse, self.level)
    }
}

/// Build one pyramid level as an owned image.
///
/// This is the one-shot convenience entry point for callers that do not need a
/// reusable workspace.
pub fn pyramid_level_owned(image: &ImageView<'_, u8>, level: u8) -> Result<OwnedPyramidLevel> {
    let mut workspace = PyramidWorkspace::with_capacity(level.saturating_add(1));
    let level_view = workspace.level(*image, level)?;
    let owned = OwnedImage::from_vec(
        copy_rows(level_view.image()),
        level_view.image().width(),
        level_view.image().height(),
    )?;
    Ok(OwnedPyramidLevel {
        image: owned,
        level,
    })
}

#[inline]
fn copy_rows(image: ImageView<'_, u8>) -> Vec<u8> {
    if image.stride() == image.width() {
        return image.as_slice()[..image.width() * image.height()].to_vec();
    }

    let mut copied = Vec::with_capacity(image.width() * image.height());
    for y in 0..image.height() {
        copied.extend_from_slice(image.row(y).expect("validated image row"));
    }
    copied
}

fn prepare_base_view<'a>(
    contiguous_base: &'a mut Vec<u8>,
    image: ImageView<'a, u8>,
) -> Result<PyramidImageView<'a>> {
    if image.stride() == image.width() {
        return PyramidImageView::new(image.width(), image.height(), image.as_slice()).ok_or(
            RadSymError::InvalidDimensions {
                width: image.width(),
                height: image.height(),
            },
        );
    }

    contiguous_base.clear();
    contiguous_base.reserve(image.width().saturating_mul(image.height()));
    for y in 0..image.height() {
        let row = image.row(y).ok_or(RadSymError::InvalidDimensions {
            width: image.width(),
            height: image.height(),
        })?;
        contiguous_base.extend_from_slice(row);
    }

    PyramidImageView::new(image.width(), image.height(), contiguous_base).ok_or(
        RadSymError::InvalidDimensions {
            width: image.width(),
            height: image.height(),
        },
    )
}

#[inline]
fn map_point_to_image_level(point: PixelCoord, level: u8) -> PixelCoord {
    let factor = factor_scalar(level);
    let offset = 0.5 * (factor - 1.0);
    PixelCoord::new(point.x * factor + offset, point.y * factor + offset)
}

#[inline]
fn map_circle_to_image_level(circle: Circle, level: u8) -> Circle {
    Circle::new(
        map_point_to_image_level(circle.center, level),
        circle.radius * factor_scalar(level),
    )
}

#[inline]
fn map_ellipse_to_image_level(ellipse: Ellipse, level: u8) -> Ellipse {
    let factor = factor_scalar(level);
    Ellipse::new(
        map_point_to_image_level(ellipse.center, level),
        ellipse.semi_major * factor,
        ellipse.semi_minor * factor,
        ellipse.angle,
    )
}

#[inline]
fn factor_scalar(level: u8) -> Scalar {
    2.0f32.powi(level as i32)
}

#[inline]
fn validate_level(level: u8) -> Result<()> {
    checked_level_factor(level)
        .ok_or(RadSymError::InvalidConfig {
            reason: "pyramid level is too large",
        })
        .map(|_| ())
}

#[inline]
fn checked_level_factor(level: u8) -> Option<usize> {
    1usize.checked_shl(level as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_zero_owned_preserves_base_image() {
        let image = ImageView::from_slice(&[1u8, 2, 3, 4, 5, 6], 3, 2).unwrap();

        let level = pyramid_level_owned(&image, 0).unwrap();

        assert_eq!(level.level(), 0);
        assert_eq!(level.factor(), 1);
        assert_eq!(level.image().width(), 3);
        assert_eq!(level.image().height(), 2);
        assert_eq!(level.image().row(0).unwrap(), &[1, 2, 3]);
        assert_eq!(level.image().row(1).unwrap(), &[4, 5, 6]);
    }

    #[test]
    fn odd_dimensions_follow_box_pyramid_rounding_and_truncation() {
        let data = (0u8..15).collect::<Vec<_>>();
        let image = ImageView::from_slice(&data, 5, 3).unwrap();

        let level = pyramid_level_owned(&image, 1).unwrap();

        assert_eq!(level.image().width(), 2);
        assert_eq!(level.image().height(), 1);
        assert_eq!(level.image().row(0).unwrap(), &[3, 5]);
    }

    #[test]
    fn workspace_reuses_buffers_and_matches_one_shot() {
        let data = (0u8..64).collect::<Vec<_>>();
        let image = ImageView::from_slice(&data, 8, 8).unwrap();
        let mut workspace = PyramidWorkspace::new();

        let borrowed = workspace.level(image, 2).unwrap();
        let owned = pyramid_level_owned(&image, 2).unwrap();

        assert_eq!(borrowed.image().width(), owned.image().width());
        assert_eq!(borrowed.image().height(), owned.image().height());
        assert_eq!(copy_rows(borrowed.image()), copy_rows(owned.image()));
    }

    #[test]
    fn remap_helpers_use_image_pixel_centers() {
        let level = OwnedPyramidLevel {
            image: OwnedImage::from_vec(vec![0u8; 4], 2, 2).unwrap(),
            level: 3,
        };

        let point = level.map_point_to_image(PixelCoord::new(10.0, 5.0));
        let circle = level.map_circle_to_image(Circle::new(PixelCoord::new(2.0, 3.0), 4.0));
        let ellipse =
            level.map_ellipse_to_image(Ellipse::new(PixelCoord::new(1.0, 2.0), 6.0, 4.0, 0.25));

        assert_eq!(point, PixelCoord::new(83.5, 43.5));
        assert_eq!(circle.center, PixelCoord::new(19.5, 27.5));
        assert_eq!(circle.radius, 32.0);
        assert_eq!(ellipse.center, PixelCoord::new(11.5, 19.5));
        assert_eq!(ellipse.semi_major, 48.0);
        assert_eq!(ellipse.semi_minor, 32.0);
        assert_eq!(ellipse.angle, 0.25);
    }
}
