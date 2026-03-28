//! Remap proposal coordinates from a pyramid working image back to the base image.

use crate::core::coords::PixelCoord;
use crate::core::scalar::Scalar;

use super::seed::{Proposal, SeedPoint};

/// Remap a proposal from a pyramid working image to base-image coordinates.
///
/// The `level` uses repeated 2x box-pyramid semantics, so the scale factor is
/// `2^level`. Proposal centers map from working-frame pixel centers to
/// image-frame pixel centers as:
///
/// `image_center = working_center * factor + 0.5 * (factor - 1)`.
///
/// If `scale_hint` is present, it is multiplied by the same factor.
#[must_use]
pub fn remap_proposal_to_image(proposal: &Proposal, level: u8) -> Proposal {
    let factor = level_factor(level);
    let offset = 0.5 * (factor - 1.0);

    Proposal {
        seed: SeedPoint {
            position: PixelCoord::new(
                proposal.seed.position.x * factor + offset,
                proposal.seed.position.y * factor + offset,
            ),
            score: proposal.seed.score,
        },
        scale_hint: proposal.scale_hint.map(|hint| hint * factor),
        polarity: proposal.polarity,
        source: proposal.source,
    }
}

/// Remap a proposal slice from a pyramid working image to base-image coordinates.
#[must_use]
pub fn remap_proposals_to_image(proposals: &[Proposal], level: u8) -> Vec<Proposal> {
    proposals
        .iter()
        .map(|proposal| remap_proposal_to_image(proposal, level))
        .collect()
}

#[inline]
fn level_factor(level: u8) -> Scalar {
    2.0f32.powi(level as i32)
}

#[cfg(test)]
mod tests {
    use crate::core::polarity::Polarity;

    use super::super::seed::ProposalSource;
    use super::*;

    #[test]
    fn remap_scales_position_and_hint() {
        let proposal = Proposal {
            seed: SeedPoint {
                position: PixelCoord::new(10.0, 12.0),
                score: 7.5,
            },
            scale_hint: Some(4.0),
            polarity: Polarity::Bright,
            source: ProposalSource::Frst,
        };

        let remapped = remap_proposal_to_image(&proposal, 3);

        assert_eq!(remapped.seed.position, PixelCoord::new(83.5, 99.5));
        assert_eq!(remapped.seed.score, proposal.seed.score);
        assert_eq!(remapped.scale_hint, Some(32.0));
        assert_eq!(remapped.polarity, proposal.polarity);
        assert_eq!(remapped.source, proposal.source);
    }

    #[test]
    fn remap_batch_preserves_order() {
        let proposals = vec![
            Proposal {
                seed: SeedPoint {
                    position: PixelCoord::new(1.0, 2.0),
                    score: 3.0,
                },
                scale_hint: None,
                polarity: Polarity::Dark,
                source: ProposalSource::Rsd,
            },
            Proposal {
                seed: SeedPoint {
                    position: PixelCoord::new(4.0, 5.0),
                    score: 6.0,
                },
                scale_hint: Some(2.5),
                polarity: Polarity::Bright,
                source: ProposalSource::External,
            },
        ];

        let remapped = remap_proposals_to_image(&proposals, 1);

        assert_eq!(remapped.len(), 2);
        assert_eq!(remapped[0].seed.position, PixelCoord::new(2.5, 4.5));
        assert_eq!(remapped[1].seed.position, PixelCoord::new(8.5, 10.5));
        assert_eq!(remapped[1].scale_hint, Some(5.0));
    }
}
