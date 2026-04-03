# Kasa Circle Fit

## Problem

Given a set of 2D points $\{(x_i, y_i)\}_{i=1}^{N}$ (possibly with
associated weights $w_i$), fit a circle that best approximates the point set.
This arises in radsym when refining ring-shaped support regions or when
converting annulus edge samples into a circle hypothesis.

## Algorithm

The Kasa method (1976) minimizes the **algebraic distance** to the implicit
circle equation. While geometric (orthogonal) distance fitting is
statistically superior, algebraic fitting reduces to a single linear system
and is fast enough for use in inner loops.

### Implicit circle equation

A circle with center $(a, b)$ and radius $r$ satisfies

$$
(x - a)^2 + (y - b)^2 = r^2.
$$

Expanding and rearranging gives the implicit form

$$
x^2 + y^2 + Dx + Ey + F = 0
$$

where $D = -2a$, $E = -2b$, and $F = a^2 + b^2 - r^2$. The circle
parameters are recovered as

$$
a = -\frac{D}{2}, \qquad b = -\frac{E}{2}, \qquad
r = \sqrt{\frac{D^2}{4} + \frac{E^2}{4} - F}.
$$

### Least-squares formulation

For each point $(x_i, y_i)$, define the algebraic residual

$$
e_i = x_i^2 + y_i^2 + D x_i + E y_i + F.
$$

The Kasa method minimizes the weighted sum of squared residuals:

$$
\min_{D,\,E,\,F} \sum_{i=1}^{N} w_i\, e_i^2.
$$

Letting $r_i = x_i^2 + y_i^2$ and $\mathbf{p} = (D,\, E,\, F)^T$, the
residual for point $i$ is

$$
e_i = r_i + \mathbf{a}_i^T \mathbf{p}, \qquad
\mathbf{a}_i = \begin{pmatrix} x_i \\ y_i \\ 1 \end{pmatrix}.
$$

### Normal equations

Setting $\partial / \partial \mathbf{p} \sum_i w_i e_i^2 = 0$ yields

$$
\mathbf{A}^T \mathbf{W} \mathbf{A}\, \mathbf{p}
  = -\mathbf{A}^T \mathbf{W}\, \mathbf{r}
$$

where $\mathbf{A}$ is the $N \times 3$ matrix with rows $\mathbf{a}_i^T$,
$\mathbf{W} = \mathrm{diag}(w_1, \ldots, w_N)$, and
$\mathbf{r} = (r_1, \ldots, r_N)^T$.

In the implementation, the $3 \times 3$ matrix $\mathbf{H} = \mathbf{A}^T
\mathbf{W} \mathbf{A}$ and the right-hand side $\mathbf{g} = -\mathbf{A}^T
\mathbf{W}\, \mathbf{r}$ are accumulated incrementally:

$$
\mathbf{H} = \sum_i w_i\, \mathbf{a}_i \mathbf{a}_i^T, \qquad
\mathbf{g} = -\sum_i w_i\, r_i\, \mathbf{a}_i.
$$

The system $\mathbf{H}\,\mathbf{p} = \mathbf{g}$ is solved via LU
decomposition (provided by nalgebra).

### Recovering circle parameters

From the solution vector $\mathbf{p} = (D,\, E,\, F)^T$:

$$
\text{center} = \left(-\frac{D}{2},\; -\frac{E}{2}\right), \qquad
r^2 = \frac{D^2}{4} + \frac{E^2}{4} - F.
$$

If $r^2 \le 10^{-6}$, the center coordinates are non-finite, or the LU
decomposition fails, the fit is rejected and the function returns `None`.

## Implementation notes

- **Minimum point count**: at least 3 points are required (a circle has 3
  degrees of freedom). Fewer points return `None`.
- **Weight clamping**: in `fit_circle_weighted`, each weight is clamped to a
  minimum of $10^{-3}$ to prevent near-zero weights from creating
  ill-conditioned systems.
- **Validity checks**: the solution is rejected if the center coordinates are
  not finite or $r^2 \le 10^{-6}$ (degenerate or imaginary radius).
- **Bias note**: the Kasa method is known to exhibit a systematic bias toward
  smaller radii when points cover only a short arc. For full or near-full
  arcs the bias is negligible.

## Configuration

The circle fit functions take no configuration struct. All behavior is
controlled through function parameters:

| Parameter | Description |
|-----------|-------------|
| `points: &[PixelCoord]` | Slice of 2D points on or near the circle |
| `weights: &[Scalar]` | Per-point weights (weighted variant only) |

## API usage

```rust
use radsym::core::circle_fit::{fit_circle, fit_circle_weighted};
use radsym::core::coords::PixelCoord;

// Unweighted fit
let points: Vec<PixelCoord> = /* edge samples */;
if let Some(circle) = fit_circle(&points) {
    println!("center=({}, {}), r={}", circle.center.x, circle.center.y, circle.radius);
}

// Weighted fit
let weights: Vec<f32> = /* per-point confidence */;
if let Some(circle) = fit_circle_weighted(&points, &weights) {
    // ...
}
```

## Reference

Kasa, I. "A circle fitting procedure and its error analysis."
*IEEE Transactions on Instrumentation and Measurement*, **25**(1), 8--14
(1976). doi:[10.1109/TIM.1976.6312298](https://doi.org/10.1109/TIM.1976.6312298)
