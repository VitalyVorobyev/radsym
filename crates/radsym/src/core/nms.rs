//! Non-maximum suppression for response maps.

use super::coords::PixelCoord;
use super::image_view::ImageView;
use super::scalar::Scalar;

/// Configuration for non-maximum suppression.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Peak {
    /// Peak position in pixel coordinates.
    pub position: PixelCoord,
    /// Peak response value.
    pub score: Scalar,
}

#[derive(Clone, Copy, Debug, Default)]
struct MaxCount {
    value: Scalar,
    count: u32,
}

impl MaxCount {
    #[inline]
    fn from_value(value: Scalar) -> Self {
        Self { value, count: 1 }
    }

    #[inline]
    fn combine(lhs: Self, rhs: Self) -> Self {
        if lhs.value > rhs.value {
            lhs
        } else if rhs.value > lhs.value {
            rhs
        } else {
            Self {
                value: lhs.value,
                count: lhs.count + rhs.count,
            }
        }
    }
}

#[derive(Default)]
struct AggStack {
    entries: Vec<(MaxCount, MaxCount)>,
}

impl AggStack {
    #[inline]
    fn clear(&mut self) {
        self.entries.clear();
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    fn aggregate(&self) -> Option<MaxCount> {
        self.entries.last().map(|(_, agg)| *agg)
    }

    #[inline]
    fn push_back(&mut self, item: MaxCount) {
        let aggregate = self
            .aggregate()
            .map_or(item, |prev| MaxCount::combine(prev, item));
        self.entries.push((item, aggregate));
    }

    #[inline]
    fn push_front_transfer(&mut self, item: MaxCount) {
        let aggregate = self
            .aggregate()
            .map_or(item, |prev| MaxCount::combine(item, prev));
        self.entries.push((item, aggregate));
    }

    #[inline]
    fn pop(&mut self) -> Option<MaxCount> {
        self.entries.pop().map(|(item, _)| item)
    }
}

#[derive(Default)]
struct AggQueue {
    front: AggStack,
    back: AggStack,
    len: usize,
}

impl AggQueue {
    #[inline]
    fn clear(&mut self) {
        self.front.clear();
        self.back.clear();
        self.len = 0;
    }

    #[inline]
    fn push(&mut self, item: MaxCount) {
        self.back.push_back(item);
        self.len += 1;
    }

    #[inline]
    fn pop(&mut self) -> Option<MaxCount> {
        if self.len == 0 {
            return None;
        }
        if self.front.is_empty() {
            while let Some(item) = self.back.pop() {
                self.front.push_front_transfer(item);
            }
        }
        self.len -= 1;
        self.front.pop()
    }

    #[inline]
    fn aggregate(&self) -> Option<MaxCount> {
        match (self.front.aggregate(), self.back.aggregate()) {
            (Some(front), Some(back)) => Some(MaxCount::combine(front, back)),
            (Some(front), None) => Some(front),
            (None, Some(back)) => Some(back),
            (None, None) => None,
        }
    }
}

fn compute_window_len(radius: usize) -> Option<usize> {
    radius.checked_mul(2)?.checked_add(1)
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
    let Some(window) = compute_window_len(r) else {
        return Vec::new();
    };
    if window > w || window > h {
        return Vec::new();
    }

    let mut row_windows = vec![MaxCount::default(); w * h];
    let mut queue = AggQueue::default();
    for y in 0..h {
        queue.clear();
        let row = response.row(y).unwrap();
        let row_start = y * w;

        for (x, &value) in row.iter().enumerate() {
            queue.push(MaxCount::from_value(value));
            if x + 1 > window {
                queue.pop();
            }
            if x + 1 >= window {
                let center_x = x - r;
                row_windows[row_start + center_x] = queue.aggregate().unwrap();
            }
        }
    }

    let mut window_stats = vec![MaxCount::default(); w * h];
    for x in r..w - r {
        queue.clear();
        for y in 0..h {
            queue.push(row_windows[y * w + x]);
            if y + 1 > window {
                queue.pop();
            }
            if y + 1 >= window {
                let center_y = y - r;
                window_stats[center_y * w + x] = queue.aggregate().unwrap();
            }
        }
    }

    let mut peaks = Vec::with_capacity(config.max_detections.min(w * h));
    for y in r..h - r {
        let row = response.row(y).unwrap();
        let stats_row = &window_stats[y * w..(y + 1) * w];
        for x in r..w - r {
            let val = row[x];
            if val < config.threshold {
                continue;
            }

            let stats = stats_row[x];
            if stats.count == 1 && val == stats.value {
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
    use rand::{rngs::StdRng, Rng, SeedableRng};

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

    fn non_maximum_suppression_reference(
        response: &ImageView<'_, Scalar>,
        config: &NmsConfig,
    ) -> Vec<Peak> {
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

    #[test]
    fn equal_neighbors_are_not_peaks() {
        let img = make_response_with_peaks(20, 20, &[(10, 10, 5.0), (12, 10, 5.0)]);
        let config = NmsConfig {
            radius: 3,
            threshold: 1.0,
            max_detections: 10,
        };

        let peaks = non_maximum_suppression(&img.view(), &config);
        assert!(peaks.is_empty());
    }

    #[test]
    fn optimized_nms_matches_reference_on_seeded_random_maps() {
        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);

        for _case in 0..200 {
            let width = rng.random_range(1..=8);
            let height = rng.random_range(1..=8);
            let values = (0..width * height)
                .map(|_| rng.random_range(0..=12) as f32 * 0.25)
                .collect::<Vec<_>>();
            let image = OwnedImage::from_vec(values, width, height).unwrap();

            let max_radius = ((width.min(height)).saturating_sub(1) / 2).min(3);
            let config = NmsConfig {
                radius: rng.random_range(0..=max_radius),
                threshold: rng.random_range(0..=12) as f32 * 0.25,
                max_detections: rng.random_range(1..=width * height),
            };

            let optimized = non_maximum_suppression(&image.view(), &config);
            let reference = non_maximum_suppression_reference(&image.view(), &config);

            assert_eq!(optimized.len(), reference.len(), "config={config:?}");
            for (lhs, rhs) in optimized.iter().zip(reference.iter()) {
                assert_eq!(lhs.position.x, rhs.position.x, "config={config:?}");
                assert_eq!(lhs.position.y, rhs.position.y, "config={config:?}");
                assert_eq!(lhs.score, rhs.score, "config={config:?}");
            }
        }
    }
}
