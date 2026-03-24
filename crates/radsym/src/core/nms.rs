//! Non-maximum suppression for response maps.

use super::coords::PixelCoord;
use super::image_view::ImageView;
use super::scalar::Scalar;

/// Configuration for non-maximum suppression.
#[derive(Debug, Clone)]
pub struct NmsConfig {
    /// Suppression radius in pixels (half-window size).
    pub radius: usize,
    /// Minimum response value to consider a pixel as a candidate.
    pub threshold: Scalar,
    /// Maximum number of detections to return (budget cap).
    pub max_detections: usize,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            radius: 5,
            threshold: 0.0,
            max_detections: 1000,
        }
    }
}

/// A detected peak with position and score.
#[derive(Debug, Clone, Copy)]
pub struct Peak {
    /// Peak position in pixel coordinates.
    pub position: PixelCoord,
    /// Peak response value.
    pub score: Scalar,
}

/// Extract local maxima from a response map via non-maximum suppression.
///
/// Returns peaks sorted by descending score, up to `config.max_detections`.
/// A pixel is a local maximum if it is strictly greater than all neighbors
/// within the suppression radius and at least `config.threshold`.
pub fn non_maximum_suppression(response: &ImageView<'_, Scalar>, config: &NmsConfig) -> Vec<Peak> {
    let w = response.width();
    let h = response.height();
    let r = config.radius;
    let mut peaks = Vec::new();

    for y in r..h.saturating_sub(r) {
        for x in r..w.saturating_sub(r) {
            let val = *response.get(x, y).unwrap();
            if val < config.threshold {
                continue;
            }

            let mut is_max = true;
            'outer: for ny in y.saturating_sub(r)..=(y + r).min(h - 1) {
                for nx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                    if nx == x && ny == y {
                        continue;
                    }
                    if let Some(&neighbor) = response.get(nx, ny) {
                        if neighbor >= val {
                            is_max = false;
                            break 'outer;
                        }
                    }
                }
            }

            if is_max {
                peaks.push(Peak {
                    position: PixelCoord::new(x as Scalar, y as Scalar),
                    score: val,
                });
            }
        }
    }

    // Sort by descending score (deterministic: break ties by y then x)
    peaks.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.position
                    .y
                    .partial_cmp(&b.position.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.position
                    .x
                    .partial_cmp(&b.position.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    peaks.truncate(config.max_detections);
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image_view::OwnedImage;

    fn make_response_with_peaks(
        width: usize,
        height: usize,
        peaks: &[(usize, usize, f32)],
    ) -> OwnedImage<Scalar> {
        let mut img = OwnedImage::<Scalar>::zeros(width, height).unwrap();
        for &(x, y, val) in peaks {
            *img.get_mut(x, y).unwrap() = val;
        }
        img
    }

    #[test]
    fn single_peak() {
        let img = make_response_with_peaks(20, 20, &[(10, 10, 5.0)]);
        let config = NmsConfig {
            radius: 3,
            threshold: 1.0,
            max_detections: 10,
        };
        let peaks = non_maximum_suppression(&img.view(), &config);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].position.x, 10.0);
        assert_eq!(peaks[0].position.y, 10.0);
        assert_eq!(peaks[0].score, 5.0);
    }

    #[test]
    fn two_peaks_sorted_by_score() {
        let img = make_response_with_peaks(30, 30, &[(8, 8, 3.0), (20, 20, 7.0)]);
        let config = NmsConfig {
            radius: 3,
            threshold: 1.0,
            max_detections: 10,
        };
        let peaks = non_maximum_suppression(&img.view(), &config);
        assert_eq!(peaks.len(), 2);
        assert_eq!(peaks[0].score, 7.0); // higher score first
        assert_eq!(peaks[1].score, 3.0);
    }

    #[test]
    fn budget_cap() {
        let img = make_response_with_peaks(30, 30, &[(5, 5, 1.0), (10, 10, 2.0), (20, 20, 3.0)]);
        let config = NmsConfig {
            radius: 2,
            threshold: 0.5,
            max_detections: 2,
        };
        let peaks = non_maximum_suppression(&img.view(), &config);
        assert_eq!(peaks.len(), 2);
    }

    #[test]
    fn threshold_filter() {
        let img = make_response_with_peaks(20, 20, &[(10, 10, 0.5)]);
        let config = NmsConfig {
            radius: 3,
            threshold: 1.0,
            max_detections: 10,
        };
        let peaks = non_maximum_suppression(&img.view(), &config);
        assert!(peaks.is_empty());
    }

    #[test]
    fn suppression_of_neighbors() {
        // Two adjacent peaks within suppression radius — only the stronger survives
        let img = make_response_with_peaks(20, 20, &[(10, 10, 5.0), (11, 10, 3.0)]);
        let config = NmsConfig {
            radius: 3,
            threshold: 1.0,
            max_detections: 10,
        };
        let peaks = non_maximum_suppression(&img.view(), &config);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].score, 5.0);
    }
}
