# Support Scoring

## Overview

Support scoring quantifies how strongly local image evidence supports a
geometric hypothesis (circle or ellipse). The output is a `SupportScore` with
component breakdown, used to rank hypotheses and reject weak detections.

## Score components

### Ringness

The **ringness** component measures gradient alignment within the annulus. It
is the mean absolute radial alignment across all valid gradient samples:

$$\text{ringness} = \frac{1}{N} \sum_{i=1}^{N} |a_i|,$$

where $a_i$ is the radial alignment of the $i$-th sample (see
[annulus sampling](annulus-sampling.md)). A value near 1.0 indicates strong
ring-like support; values near 0.0 indicate random or tangential gradients.

### Angular coverage

The **angular_coverage** component measures what fraction of the angular range
$[0, 2\pi)$ is covered by sufficiently strong gradient responses. The annulus
is divided into angular bins, and each bin is considered "covered" if at least
one sample in that bin has a gradient magnitude above a threshold. Coverage is
the fraction of covered bins.

This component guards against the case where a few strong edge fragments
dominate the ringness score despite most of the ring being absent.

## Weighted total

The combined score is a weighted sum clamped to $[0, 1]$:

$$\text{total} = \text{clamp}\bigl(w_r \cdot \text{ringness} + w_c \cdot \text{coverage},\; 0,\; 1\bigr),$$

where $w_r$ = `weight_ringness` (default 0.6) and $w_c$ = `weight_coverage`
(default 0.4).

## Degeneracy detection

If the number of valid gradient samples falls below `min_samples` (default 8),
the score is flagged as degenerate (`is_degenerate = true`) and the total is
set to 0.0. This prevents small hypotheses near image borders or in flat
regions from receiving artificially high scores.

## Rectified variant

`score_rectified_circle_support` computes the support score for a circle
defined in rectified space (via a `Homography`). It transforms sample
positions from the rectified circle into image coordinates, reads the
gradient in image space, and pulls back the radial direction through the
homography Jacobian to compute alignment. This ensures correct scoring even
under perspective distortion.

## Configuration

```rust
pub struct ScoringConfig {
    /// Annulus sampling parameters.
    pub sampling: AnnulusSamplingConfig,
    /// Fractional margin around the radius. Default: 0.3.
    pub annulus_margin: f32,
    /// Minimum samples to avoid degeneracy. Default: 8.
    pub min_samples: usize,
    /// Weight for ringness in total score. Default: 0.6.
    pub weight_ringness: f32,
    /// Weight for angular coverage in total score. Default: 0.4.
    pub weight_coverage: f32,
}
```

| Field | Effect of increasing |
|-------|---------------------|
| `annulus_margin` | Wider sampling band, more tolerant of radius error |
| `min_samples` | Stricter degeneracy threshold |
| `weight_ringness` | Favors gradient alignment over coverage |
| `weight_coverage` | Favors complete rings over strong partial arcs |
