# FRST: Fast Radial Symmetry Transform

## Problem

Detecting the centers of radially symmetric structures (circles, rings,
bullseyes) in a grayscale image is a recurring primitive in computer vision --
from traffic-sign detection to particle tracking. Classical approaches such as
the Hough transform for circles are expensive: they accumulate votes in a
three-dimensional $(x, y, r)$ parameter space, scaling poorly with image size
and radius range.

FRST sidesteps the cubic cost by exploiting gradient orientation. Instead of
testing every possible circle, each pixel votes for a single center location at
each radius, using its gradient direction to determine *where* to vote. The
result is a per-pixel response map that peaks at radial symmetry centers.

## Algorithm

### Gradient and affected pixels

Let $I$ be a grayscale image and let $\mathbf{g}(p) = (\partial_x I,\,
\partial_y I)$ be the image gradient at pixel $p = (x, y)$. The gradient
magnitude is $|\mathbf{g}(p)| = \sqrt{g_x^2 + g_y^2}$, and the unit gradient
direction is

$$\hat{\mathbf{g}}(p) = \frac{\mathbf{g}(p)}{|\mathbf{g}(p)|}.$$

For a given test radius $n$, each pixel $p$ with nonzero gradient defines two
**affected pixels** by displacing along the gradient direction:

$$p_{+\text{ve}}(n) = p + \operatorname{round}\!\bigl(\hat{\mathbf{g}}(p) \cdot n\bigr),$$

$$p_{-\text{ve}}(n) = p - \operatorname{round}\!\bigl(\hat{\mathbf{g}}(p) \cdot n\bigr).$$

The positive-affected pixel $p_{+\text{ve}}$ lies *downstream* in the gradient
direction -- toward a brighter center. The negative-affected pixel
$p_{-\text{ve}}$ lies *upstream* -- toward a darker center. Rounding to
integer coordinates maps each vote onto the discrete pixel grid.

### Accumulator images

For each radius $n$, FRST maintains two accumulator images:

1. **Orientation projection** $O_n$: records the *direction* of votes.
   At the positive-affected pixel, $O_n$ is incremented by $+1$;
   at the negative-affected pixel, it is decremented by $-1$:

$$O_n(q) = \sum_{\{p \,:\, p_{+\text{ve}} = q\}} (+1) \;+\;
           \sum_{\{p \,:\, p_{-\text{ve}} = q\}} (-1).$$

2. **Magnitude projection** $M_n$: records the *strength* of votes.
   At both affected pixels, $M_n$ is incremented by $|\mathbf{g}(p)|$:

$$M_n(q) = \sum_{\{p \,:\, p_{+\text{ve}} = q\}} |\mathbf{g}(p)| \;+\;
           \sum_{\{p \,:\, p_{-\text{ve}} = q\}} |\mathbf{g}(p)|.$$

The orientation accumulator distinguishes genuine radial symmetry (where
gradients converge consistently toward a single point) from accidental
magnitude pile-ups.

### Normalization and combination

The raw accumulators are normalized by their respective maxima to produce
dimensionless quantities in $[-1, 1]$ and $[0, 1]$:

$$\tilde{O}_n = \frac{O_n}{\max\!\bigl(1,\; \max_q |O_n(q)|\bigr)}, \qquad
  \tilde{M}_n = \frac{M_n}{\max\!\bigl(1,\; \max_q M_n(q)\bigr)}.$$

These are combined into the per-radius contribution

$$F_n = \bigl|\tilde{O}_n\bigr|^{\alpha} \cdot \tilde{M}_n,$$

where $\alpha \geq 1$ is the **radial strictness exponent**. At $\alpha = 1$,
the orientation term acts as a linear gate; at $\alpha = 2$ (the default),
pixels that receive orientation-inconsistent votes are suppressed
quadratically. Increasing $\alpha$ improves selectivity for true radial
symmetry at the expense of weaker response to imperfect targets.

### Gaussian smoothing

Each $F_n$ is convolved with a Gaussian kernel whose standard deviation scales
with the radius:

$$S_n = G_{\sigma_n} * F_n, \qquad \sigma_n = k_n \cdot n,$$

where $k_n$ is the **smoothing factor** (default 0.5). Smoothing merges nearby
votes that would otherwise form a speckled peak, particularly important for
large radii where the ring of voting pixels is thin.

### Multi-radius fusion

The full FRST response is the sum across all tested radii:

$$S = \sum_{n \in \mathcal{N}} S_n,$$

where $\mathcal{N}$ is the configured set of discrete radii. Peaks of $S$ are
candidate centers of radially symmetric features. Non-maximum suppression is
applied downstream to extract discrete proposals.

### Polarity modes

Which affected pixels participate depends on the target polarity:

| Mode | Votes at $p_{+\text{ve}}$ | Votes at $p_{-\text{ve}}$ | Detects |
|------|:---:|:---:|---------|
| **Bright** | yes | no | bright centers (gradient points inward) |
| **Dark** | no | yes | dark centers (gradient points outward) |
| **Both** | yes | yes | either polarity |

## Implementation notes

- **Gradient thresholding**: pixels with $|\mathbf{g}| < t$ (the
  `gradient_threshold` parameter) are skipped entirely. This eliminates votes
  from flat regions and sensor noise, reducing computation and background
  clutter.
- **Rayon parallelism**: when the `rayon` feature is enabled, per-radius
  computation runs in parallel. The voting pass within each radius is
  sequential (accumulator writes are not atomic), but radius-level parallelism
  provides near-linear speedup for large radius sets.
- **Gaussian blur**: smoothing is skipped when $\sigma_n < 0.5$ pixels, since
  at that width the kernel has negligible effect.

## Configuration

```rust
pub struct FrstConfig {
    /// Discrete radii to test (pixels). Default: [3, 5, 7, 9, 11].
    pub radii: Vec<u32>,
    /// Radial strictness exponent. Default: 2.0.
    pub alpha: f32,
    /// Minimum gradient magnitude for voting. Default: 0.0.
    pub gradient_threshold: f32,
    /// Which polarity to detect. Default: Both.
    pub polarity: Polarity,
    /// Gaussian sigma = smoothing_factor * n. Default: 0.5.
    pub smoothing_factor: f32,
}
```

| Field | Effect of increasing |
|-------|---------------------|
| `alpha` | Sharper discrimination, weaker response on imperfect targets |
| `gradient_threshold` | Fewer votes, less noise, possible loss of weak edges |
| `smoothing_factor` | Broader peaks, better tolerance to discretization |

## API usage

```rust
use radsym::core::gradient::sobel_gradient;
use radsym::core::image_view::ImageView;
use radsym::propose::frst::{frst_response, FrstConfig};
use radsym::core::polarity::Polarity;

let image = ImageView::from_slice(&pixels, width, height)?;
let gradient = sobel_gradient(&image)?;

let config = FrstConfig {
    radii: vec![8, 10, 12],
    alpha: 2.0,
    gradient_threshold: 2.0,
    polarity: Polarity::Bright,
    smoothing_factor: 0.5,
};

let response_map = frst_response(&gradient, &config)?;
// response_map.response() is an OwnedImage<f32> — apply NMS to extract peaks
```

## Reference

Loy, G. and Zelinsky, A. "Fast radial symmetry for detecting points of
interest." *IEEE Transactions on Pattern Analysis and Machine Intelligence*
**25**(8), 959--973 (2003).
doi:[10.1109/TPAMI.2003.1217601](https://doi.org/10.1109/TPAMI.2003.1217601)
