//! Proposal extraction from response maps.

use crate::core::image_view::OwnedImage;
use crate::core::nms::{non_maximum_suppression, NmsConfig, Peak};
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
}

/// Extract proposals from a response map using non-maximum suppression.
///
/// Returns proposals sorted by descending score, up to the NMS budget.
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
}
