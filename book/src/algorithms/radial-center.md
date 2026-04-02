# Radial Center Refinement

## Problem

Given a coarse seed location (e.g., from FRST or RSD), find the **subpixel
center** of a radially symmetric intensity pattern. The algorithm must be fast
enough for real-time tracking of many particles and robust to noise and
partial occlusion.

## Algorithm

The radial center method, introduced by Parthasarathy (2012), exploits the
fact that gradient vectors of a radially symmetric pattern all point toward
(or away from) the center. The center is the point that minimizes the
weighted sum of squared distances to the lines defined by all gradient
vectors in a local patch.

### Gradient lines

At each pixel $p_i = (x_i, y_i)$, compute the image gradient
$\mathbf{g}_i = (g_x^{(i)},\, g_y^{(i)})$ and its magnitude
$m_i = \|\mathbf{g}_i\|$. The unit gradient direction is

$$\hat{\mathbf{n}}_i = \frac{1}{m_i}\begin{pmatrix} g_x^{(i)} \\ g_y^{(i)} \end{pmatrix}.$$

This direction defines a line through $p_i$. A candidate center
$\mathbf{c} = (c_x, c_y)$ has signed distance to this line equal to

$$d_i(\mathbf{c}) = \hat{\mathbf{n}}_i \cdot (\mathbf{c} - p_i)
  = \hat{n}_x^{(i)}(c_x - x_i) + \hat{n}_y^{(i)}(c_y - y_i).$$

### Weighted least squares

The center is found by minimizing

$$E(\mathbf{c}) = \sum_i w_i \bigl[ d_i(\mathbf{c}) \bigr]^2$$

with weights $w_i = m_i^2$ (gradient magnitude squared). Weighting by $m_i^2$
down-weights pixels with weak gradients, which carry little directional
information and would otherwise bias the estimate.

Expanding the objective and setting $\partial E / \partial \mathbf{c} = 0$
yields the $2 \times 2$ normal equations

$$\mathbf{H}\,\mathbf{c} = \mathbf{b}$$

where

$$H = \sum_i w_i
  \begin{pmatrix}
    \hat{n}_x^2 & \hat{n}_x \hat{n}_y \\
    \hat{n}_x \hat{n}_y & \hat{n}_y^2
  \end{pmatrix}, \qquad
b = \sum_i w_i
  \begin{pmatrix}
    \hat{n}_x^2\, x_i + \hat{n}_x \hat{n}_y\, y_i \\
    \hat{n}_x \hat{n}_y\, x_i + \hat{n}_y^2\, y_i
  \end{pmatrix}.$$

Note that $\mathbf{H}$ is the weighted sum of the outer products
$\hat{\mathbf{n}}_i \hat{\mathbf{n}}_i^T$, and $\mathbf{b} = \sum_i w_i
(\hat{\mathbf{n}}_i \hat{\mathbf{n}}_i^T) p_i$, so the normal equations
can also be written compactly as

$$\biggl(\sum_i w_i\, \hat{\mathbf{n}}_i \hat{\mathbf{n}}_i^T\biggr)
  \mathbf{c}
  = \sum_i w_i\, (\hat{\mathbf{n}}_i \hat{\mathbf{n}}_i^T)\, p_i.$$

### Closed-form solution

Since $\mathbf{H}$ is $2 \times 2$, the solution is obtained by direct
inversion:

$$\det(\mathbf{H}) = H_{00}\, H_{11} - H_{01}^2,$$

$$c_x = \frac{H_{11}\, b_x - H_{01}\, b_y}{\det(\mathbf{H})}, \qquad
  c_y = \frac{H_{00}\, b_y - H_{01}\, b_x}{\det(\mathbf{H})}.$$

The algorithm is **non-iterative**: a single pass over the patch pixels
followed by one $2 \times 2$ linear solve.

### Two variants in radsym

| Variant | Gradient | Grid | Function |
|---------|----------|------|----------|
| **Reference** | Roberts cross | half-pixel $(i+0.5,\, j+0.5)$ | `radial_center_refine` |
| **Production** | Precomputed Sobel | integer pixel | `radial_center_refine_from_gradient` |

The **reference** variant computes Roberts-cross gradients on the half-pixel
grid, matching the original paper exactly:

$$g_x(i\!+\!\tfrac{1}{2},\, j\!+\!\tfrac{1}{2})
  = \tfrac{1}{2}\bigl[I(i\!+\!1,j) - I(i,j) + I(i\!+\!1,j\!+\!1) - I(i,j\!+\!1)\bigr],$$

$$g_y(i\!+\!\tfrac{1}{2},\, j\!+\!\tfrac{1}{2})
  = \tfrac{1}{2}\bigl[I(i,j\!+\!1) - I(i,j) + I(i\!+\!1,j\!+\!1) - I(i\!+\!1,j)\bigr].$$

The **production** variant accepts a precomputed `GradientField` (typically
Sobel), which avoids redundant gradient computation when the field is already
available from proposal generation.

## Implementation notes

- **Degenerate detection**: if $|\det(\mathbf{H})| < 10^{-12}$, the system
  is nearly singular (e.g., uniform patch or all gradients aligned). The
  function returns `RefinementStatus::Degenerate` and falls back to the
  original seed.
- **Gradient threshold**: pixels with $m_i$ below
  `RadialCenterConfig::gradient_threshold` (default $10^{-4}$) are excluded
  from the summation.
- **Patch radius**: the fit uses a square patch of side
  $2 \times \texttt{patch\_radius} + 1$ centered on the rounded seed. Default
  is 12, yielding a $25 \times 25$ patch.
- **Accumulation precision**: all sums are accumulated in `f64` to avoid
  catastrophic cancellation, then the final center is cast back to `f32`.

## Configuration

```rust
pub struct RadialCenterConfig {
    pub patch_radius: usize,       // default: 12
    pub gradient_threshold: f32,   // default: 1e-4
}
```

## API usage

```rust
use radsym::refine::{radial_center_refine_from_gradient, RadialCenterConfig};

let config = RadialCenterConfig::default();
let result = radial_center_refine_from_gradient(&gradient_field, seed, &config)?;

if result.converged() {
    let center = result.hypothesis; // PixelCoord (subpixel)
    let shift = result.residual;    // displacement from seed
}
```

## Reference

Parthasarathy, R. "Rapid, accurate particle tracking by calculation of radial
symmetry centers." *Nature Methods* **9**, 724--726 (2012).
doi:[10.1038/nmeth.2071](https://doi.org/10.1038/nmeth.2071)
