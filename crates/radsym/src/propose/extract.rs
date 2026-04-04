//! Proposal extraction from response maps.

use crate::core::image_view::OwnedImage;
use crate::core::nms::{NmsConfig, Peak, non_maximum_suppression};
use crate::core::polarity::Polarity;
use crate::core::scalar::Scalar;

use super::seed::{Proposal, ProposalSource, SeedPoint};

/// A wrapper around an FRST (or other) response map with metadata.
#[derive(Debug)]
pub struct ResponseMap {
    /// The response image.
    data: OwnedImage<Scalar>,
    /// Source algorithm that produced this response.
    source: ProposalSource,
}

impl ResponseMap {
    /// Create a new response map.
    pub fn new(data: OwnedImage<Scalar>, source: ProposalSource) -> Self {
        Self { data, source }
    }

    /// Borrowed view of the response image.
    pub fn view(&self) -> crate::core::image_view::ImageView<'_, Scalar> {
        self.data.view()
    }

    /// Reference to the underlying response image.
    pub fn response(&self) -> &OwnedImage<Scalar> {
        &self.data
    }

    /// The source algorithm.
    pub fn source(&self) -> ProposalSource {
        self.source
    }

    /// Consume the response map and return the underlying image.
    pub fn into_response(self) -> OwnedImage<Scalar> {
        self.data
    }
}

/// Extract proposals from a response map using non-maximum suppression.
///
/// Returns proposals sorted by descending score, up to the NMS budget.
///
/// # Example
///
/// ```rust
/// use radsym::{ImageView, FrstConfig, NmsConfig, Polarity,
///              sobel_gradient, frst_response, extract_proposals};
///
/// let size = 64usize;
/// let mut data = vec![0u8; size * size];
/// for y in 0..size {
///     for x in 0..size {
///         let dx = x as f32 - 32.0;
///         let dy = y as f32 - 32.0;
///         if (dx * dx + dy * dy).sqrt() <= 10.0 { data[y * size + x] = 255; }
///     }
/// }
/// let image = ImageView::from_slice(&data, size, size).unwrap();
/// let grad = sobel_gradient(&image).unwrap();
/// let config = FrstConfig { radii: vec![9, 10, 11], ..FrstConfig::default() };
/// let response = frst_response(&grad, &config).unwrap();
/// let nms = NmsConfig { radius: 5, threshold: 0.0, max_detections: 10 };
/// let proposals = extract_proposals(&response, &nms, Polarity::Bright);
/// assert!(!proposals.is_empty());
/// // Proposals are sorted by descending score
/// if proposals.len() > 1 {
///     assert!(proposals[0].seed.score >= proposals[1].seed.score);
/// }
/// ```
pub fn extract_proposals(
    response: &ResponseMap,
    nms_config: &NmsConfig,
    polarity: Polarity,
) -> Vec<Proposal> {
    let peaks: Vec<Peak> = non_maximum_suppression(&response.view(), nms_config);

    peaks
        .into_iter()
        .map(|peak| Proposal {
            seed: SeedPoint {
                position: peak.position,
                score: peak.score,
            },
            scale_hint: None,
            polarity,
            source: response.source(),
        })
        .collect()
}

/// Greedily suppress proposals that are closer than `min_distance`.
///
/// The input order is preserved for retained proposals, so callers should sort
/// or rank proposals before calling this helper when they want the strongest
/// candidate in each spatial neighborhood to survive.
pub fn suppress_proposals_by_distance(
    proposals: &[Proposal],
    min_distance: Scalar,
    max_detections: usize,
) -> Vec<Proposal> {
    if proposals.is_empty() || max_detections == 0 {
        return Vec::new();
    }

    let min_distance_sq = min_distance.max(0.0).powi(2);
    let mut kept: Vec<Proposal> = Vec::with_capacity(proposals.len().min(max_detections));

    'proposal: for proposal in proposals {
        let position = proposal.seed.position;
        for other in &kept {
            let dx = position.x - other.seed.position.x;
            let dy = position.y - other.seed.position.y;
            if dx * dx + dy * dy < min_distance_sq {
                continue 'proposal;
            }
        }

        kept.push(proposal.clone());
        if kept.len() == max_detections {
            break;
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image_view::OwnedImage;

    #[test]
    fn extract_from_response_map() {
        let mut response_data = OwnedImage::<Scalar>::zeros(30, 30).unwrap();
        *response_data.get_mut(10, 10).unwrap() = 5.0;
        *response_data.get_mut(20, 20).unwrap() = 3.0;

        let response = ResponseMap::new(response_data, ProposalSource::Frst);

        let nms = NmsConfig {
            radius: 3,
            threshold: 1.0,
            max_detections: 10,
        };

        let proposals = extract_proposals(&response, &nms, Polarity::Both);
        assert_eq!(proposals.len(), 2);
        assert_eq!(proposals[0].seed.score, 5.0);
        assert_eq!(proposals[1].seed.score, 3.0);
        assert_eq!(proposals[0].source, ProposalSource::Frst);
        assert_eq!(proposals[0].polarity, Polarity::Both);
    }

    #[test]
    fn extract_respects_budget() {
        let mut response_data = OwnedImage::<Scalar>::zeros(40, 40).unwrap();
        *response_data.get_mut(5, 5).unwrap() = 1.0;
        *response_data.get_mut(15, 15).unwrap() = 2.0;
        *response_data.get_mut(30, 30).unwrap() = 3.0;

        let response = ResponseMap::new(response_data, ProposalSource::Frst);

        let nms = NmsConfig {
            radius: 2,
            threshold: 0.5,
            max_detections: 2,
        };

        let proposals = extract_proposals(&response, &nms, Polarity::Bright);
        assert_eq!(proposals.len(), 2);
        // Top 2 by score
        assert_eq!(proposals[0].seed.score, 3.0);
        assert_eq!(proposals[1].seed.score, 2.0);
    }

    #[test]
    fn suppress_proposals_by_distance_keeps_strongest_per_cluster() {
        let proposals = vec![
            Proposal {
                seed: SeedPoint {
                    position: crate::core::coords::PixelCoord::new(10.0, 10.0),
                    score: 5.0,
                },
                scale_hint: Some(12.0),
                polarity: Polarity::Dark,
                source: ProposalSource::Frst,
            },
            Proposal {
                seed: SeedPoint {
                    position: crate::core::coords::PixelCoord::new(14.0, 12.0),
                    score: 4.0,
                },
                scale_hint: Some(12.0),
                polarity: Polarity::Dark,
                source: ProposalSource::Frst,
            },
            Proposal {
                seed: SeedPoint {
                    position: crate::core::coords::PixelCoord::new(40.0, 40.0),
                    score: 3.0,
                },
                scale_hint: Some(12.0),
                polarity: Polarity::Dark,
                source: ProposalSource::Frst,
            },
        ];

        let kept = suppress_proposals_by_distance(&proposals, 6.0, 8);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].seed.position.x, 10.0);
        assert_eq!(kept[0].seed.position.y, 10.0);
        assert_eq!(kept[1].seed.position.x, 40.0);
        assert_eq!(kept[1].seed.position.y, 40.0);
    }
}
