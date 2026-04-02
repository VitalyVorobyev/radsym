# Ellipse Refinement

## Problem

After the proposal and scoring stages, the best circle hypotheses may not
accurately describe the true feature boundary -- perspective distortion,
lens aberration, or an intrinsically elliptical target all produce
non-circular edges. Ellipse refinement upgrades a circle seed into a
five-parameter ellipse (center $c_x, c_y$; semi-axes $a, b$; orientation
$\theta$) by alternating between edge detection and robust fitting.

## Algorithm

### Parameterization

An ellipse is represented by a center $(c_x, c_y)$, semi-major axis $a$,
semi-minor axis $b$, and orientation angle $\theta$ (measured from the $x$-axis
to the semi-major axis). A point $(x, y)$ is rotated into the ellipse frame:

$$q_x = \cos\theta \cdot (x - c_x) + \sin\theta \cdot (y - c_y),$$

$$q_y = -\sin\theta \cdot (x - c_x) + \cos\theta \cdot (y - c_y).$$

The normalized radial distance is

$$s = \frac{q_x^2}{a^2} + \frac{q_y^2}{b^2},$$

and the geometric residual for fitting is $r = \sqrt{s} - 1$. A point
lying exactly on the ellipse gives $r = 0$.

### Iterative loop

Starting from the seed circle (treated as a degenerate ellipse with $a = b$
and $\theta = 0$), the refiner executes up to `max_iterations` rounds:

1. **Edge detection along normals.** The angular range $[0, 2\pi)$ is divided
   into `ray_count` sectors. For each sector, a search ray is cast along the
   outward normal of the current ellipse estimate. The gradient magnitude
   profile along this ray is scanned for peaks -- the strongest gradient
   response within `normal_search_half_width` pixels of the predicted boundary
   is accepted as an edge observation. Bilinear interpolation provides sub-pixel
   gradient sampling.

2. **Trimmed Gauss-Newton fitting.** Given $N$ edge observations, sort by
   residual magnitude $|r_i|$ and retain only the best 75% (the `TRIM_KEEP_FRACTION`).
   This trimming rejects outliers from texture, adjacent features, or missed edges.

   The Gauss-Newton update minimizes the sum of squared residuals over the
   five-dimensional parameter vector
   $\mathbf{p} = (c_x,\, c_y,\, \log a,\, \log b,\, \theta)^T$ (log-scale
   for the semi-axes ensures positivity):

   $$\mathbf{J}^T \mathbf{J}\, \Delta\mathbf{p} = -\mathbf{J}^T \mathbf{r},$$

   where $\mathbf{J}$ is the $N \times 5$ Jacobian with rows

   $$\frac{\partial r_i}{\partial \mathbf{p}} = \left(
     \frac{\partial r_i}{\partial c_x},\;
     \frac{\partial r_i}{\partial c_y},\;
     \frac{\partial r_i}{\partial \log a},\;
     \frac{\partial r_i}{\partial \log b},\;
     \frac{\partial r_i}{\partial \theta}
   \right).$$

   The individual Jacobian components are:

   $$\frac{\partial r}{\partial c_x} = \frac{-q_x \cos\theta / a^2 + q_y \sin\theta / b^2}{\sqrt{s}},$$

   $$\frac{\partial r}{\partial c_y} = \frac{-q_x \sin\theta / a^2 - q_y \cos\theta / b^2}{\sqrt{s}},$$

   $$\frac{\partial r}{\partial \log a} = \frac{-q_x^2 / a^2}{\sqrt{s}}, \qquad
     \frac{\partial r}{\partial \log b} = \frac{-q_y^2 / b^2}{\sqrt{s}},$$

   $$\frac{\partial r}{\partial \theta} = \frac{q_x q_y (1/a^2 - 1/b^2)}{\sqrt{s}}.$$

   The $5 \times 5$ normal equations are solved via Cholesky decomposition.
   Up to `SOLVER_MAX_STEPS` (8) inner iterations are run per outer loop.

3. **Guard constraints.** After each update the candidate ellipse is checked:
   - Semi-minor axis must be at least $0.55 \times r_{\text{seed}}$.
   - Semi-major axis must be at most $1.6 \times r_{\text{seed}}$.
   - Axis ratio $a/b$ must not exceed `max_axis_ratio` (default 1.8).
   - Center shift from the original seed must be within
     `max_center_shift_fraction` $\times\, r_{\text{seed}}$ (default 0.4).
   - The ellipse must remain within image bounds.

   If any constraint fails, the update is rejected and the previous estimate is
   kept.

4. **Convergence.** The loop terminates early when the center shift and axis
   changes between successive iterations fall below `convergence_tol` (default
   0.1 pixels).

### Initial edge search

Before entering the iterative loop, a radial search from
`radial_search_inner` to `radial_search_outer` times the seed radius gathers
the first set of boundary points. These are used to bootstrap an initial
ellipse via weighted covariance analysis of the edge locations, which provides
a better starting point than the seed circle alone.

## Configuration

```rust
pub struct EllipseRefineConfig {
    pub max_iterations: usize,          // 5
    pub convergence_tol: f32,           // 0.1
    pub annulus_margin: f32,            // 0.3
    pub radial_center: RadialCenterConfig,
    pub sampling: AnnulusSamplingConfig,
    pub min_alignment: f32,             // 0.3
    pub ray_count: usize,              // 96
    pub radial_search_inner: f32,      // 0.6
    pub radial_search_outer: f32,      // 1.45
    pub normal_search_half_width: f32, // 6.0
    pub min_inlier_coverage: f32,      // 0.6
    pub max_center_shift_fraction: f32,// 0.4
    pub max_axis_ratio: f32,           // 1.8
}
```

| Field | Effect of increasing |
|-------|---------------------|
| `ray_count` | Denser angular sampling, better coverage but slower |
| `normal_search_half_width` | Wider search window, tolerates larger initial error |
| `max_axis_ratio` | Permits more eccentric ellipses |
| `max_center_shift_fraction` | Allows larger center correction from the seed |

## Reference

Fitzgibbon, A., Pilu, M., Fisher, R. "Direct least squares fitting of
ellipses." *IEEE Transactions on Pattern Analysis and Machine Intelligence*
**21**(5), 476--480 (1999).
doi:[10.1109/34.765658](https://doi.org/10.1109/34.765658)
