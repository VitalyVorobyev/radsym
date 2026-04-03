# RSD: Radial Symmetry Detector

## Problem

RSD addresses the same center-detection problem as
[FRST](frst.md) -- finding centers of radially symmetric structures in
grayscale images -- but trades discrimination for speed. In time-critical
pipelines (real-time video, large radius sweeps), FRST's orientation
accumulator and $\alpha$-exponent combination step become the bottleneck. RSD
drops both, retaining only magnitude-based voting.

## Algorithm

### Magnitude-only voting

As in FRST, each pixel $p$ with gradient $\mathbf{g}(p)$ and unit direction
$\hat{\mathbf{g}}(p)$ defines affected pixels at radius $n$:

$$p_{+\text{ve}}(n) = p + \operatorname{round}\!\bigl(\hat{\mathbf{g}}(p) \cdot n\bigr),$$

$$p_{-\text{ve}}(n) = p - \operatorname{round}\!\bigl(\hat{\mathbf{g}}(p) \cdot n\bigr).$$

RSD maintains a **single accumulator** $M_n$ per radius. Each voter adds its
gradient magnitude at the affected pixel(s):

$$M_n(q) = \sum_{\{p \,:\, p_{+\text{ve}} = q\}} |\mathbf{g}(p)| \;+\;
           \sum_{\{p \,:\, p_{-\text{ve}} = q\}} |\mathbf{g}(p)|.$$

There is **no orientation accumulator** $O_n$ and **no $\alpha$-exponent
step**. The absence of $O_n$ means RSD cannot distinguish a location where
gradient directions converge coherently from one where strong but
randomly-oriented gradients happen to land. This is the source of the speed /
discrimination trade-off.

### Smoothing and fusion

The accumulator is Gaussian-smoothed exactly as in FRST:

$$S_n = G_{\sigma_n} * M_n, \qquad \sigma_n = k_n \cdot n,$$

and the multi-radius response is the sum over all tested radii:

$$S = \sum_{n \in \mathcal{N}} S_n.$$

### Computational advantage

Per radius, FRST performs three operations per voting pixel (orientation
increment, magnitude increment, combination via $\alpha$-power), plus
normalizes $O_n$ globally. RSD performs one magnitude accumulation per voter.
The elimination of $O_n$ and the power-law combination yields roughly a
**2x wall-clock speedup** with identical memory layout, making RSD the
preferred proposer when many radii must be tested under a time budget.

### Polarity handling

Polarity logic is identical to FRST: `Bright` votes only at $p_{+\text{ve}}$,
`Dark` only at $p_{-\text{ve}}$, and `Both` votes at both locations. Each
polarity branch is implemented as a separate tight loop to avoid per-pixel
branching.

## When to use

- **Real-time pipelines** where proposal generation must finish within a fixed
  time budget.
- **Large radius sweeps** (many values of $n$): RSD's per-radius cost is lower,
  so the total cost scales more favorably.
- **Coarse proposal stage** followed by a discriminative refinement step (e.g.,
  radial center or ellipse fit) that will reject false positives anyway.

When orientation selectivity matters -- for example, distinguishing a bullseye
from a textured corner -- prefer FRST.

## Configuration

```rust
pub struct RsdConfig {
    /// Discrete radii to test (pixels). Default: [3, 5, 7, 9, 11].
    pub radii: Vec<u32>,
    /// Minimum gradient magnitude for voting. Default: 0.0.
    pub gradient_threshold: f32,
    /// Which polarity to detect. Default: Both.
    pub polarity: Polarity,
    /// Gaussian sigma = smoothing_factor * n. Default: 0.5.
    pub smoothing_factor: f32,
}
```

Note the absence of `alpha` -- no strictness exponent is needed because there
is no orientation accumulator to normalize.

## API usage

```rust
use radsym::core::gradient::sobel_gradient;
use radsym::core::image_view::ImageView;
use radsym::propose::rsd::{rsd_response, RsdConfig};
use radsym::core::polarity::Polarity;

let image = ImageView::from_slice(&pixels, width, height)?;
let gradient = sobel_gradient(&image)?;

let config = RsdConfig {
    radii: vec![8, 10, 12],
    gradient_threshold: 2.0,
    polarity: Polarity::Bright,
    smoothing_factor: 0.5,
};

let response_map = rsd_response(&gradient, &config)?;
// response_map.response() is an OwnedImage<f32> — apply NMS to extract peaks
```

## Reference

Barnes, N., Zelinsky, A., Fletcher, L.S. "Real-time radial symmetry for speed
sign detection." *IEEE Intelligent Vehicles Symposium*, 566--571 (2008).
doi:[10.1109/IVS.2008.4621217](https://doi.org/10.1109/IVS.2008.4621217)
