# Pipeline Overview

## Architecture

The radsym detection pipeline has three stages, each independently usable:

```text
                   ┌──────────┐    ┌───────┐    ┌────────┐
  Image ──Gradient──▶ Propose ├───▶│ Score ├───▶│ Refine ├──▶ Detections
                   └──────────┘    └───────┘    └────────┘
```

The design deliberately avoids a monolithic detector. Each stage is a pure
function: it reads immutable inputs and returns a new value. You can swap,
skip, or repeat stages as your application requires.

## Stage 1: Propose

**Goal.** Convert the gradient field into a small set of candidate center
locations.

The primary algorithm is the **Fast Radial Symmetry Transform (FRST)** (Loy &
Zelinsky, ECCV 2002). For every pixel whose gradient magnitude exceeds a
threshold, FRST casts a vote at the position offset by $\pm n$ pixels along the
gradient direction, where $n$ is a candidate radius. Votes accumulate into an
orientation image $O_n$ and a magnitude image $M_n$, which are combined as:

$$F_n = \lvert \tilde{O}_n \rvert^{\alpha} \cdot \tilde{M}_n$$

where $\alpha$ controls radial strictness and $\tilde{\cdot}$ denotes
Gaussian-smoothed, clamped accumulators. The per-radius responses are summed
into a final `ResponseMap`.

An alternative proposer, **RSD** (Barnes et al. 2008), uses a faster
single-radius voting scheme optimized for real-time applications.

**Non-maximum suppression** (`extract_proposals`) then picks local peaks from
the response map, returning a ranked list of `Proposal` values.

### Key types

| Input | Output |
|-------|--------|
| `GradientField` | `ResponseMap` |
| `ResponseMap` + `NmsConfig` + `Polarity` | `Vec<Proposal>` |

## Stage 2: Score

**Goal.** Quantify how strongly image evidence supports each proposed circle or
ellipse.

Scoring samples the gradient field inside an annular region around the
hypothesis. For a circle of radius $r$, the annulus spans
$[(1 - m) \cdot r,\; (1 + m) \cdot r]$, where $m$ is the configurable margin.

Two complementary signals are extracted:

- **Ringness** — the mean alignment between each sampled gradient and the
  radial direction from the hypothesized center. A perfect circle gives
  ringness $\approx 1$.
- **Angular coverage** — the fraction of angular bins (around the center) that
  contain at least one supporting gradient sample. Full coverage means evidence
  is spread around the entire circumference, not concentrated in one arc.

These are combined into a single `SupportScore`:

$$\text{total} = w_r \cdot \text{ringness} + w_c \cdot \text{coverage}$$

where the weights $w_r$ and $w_c$ are set in `ScoringConfig`. Proposals whose
`total` falls below a threshold are discarded.

### Key types

| Input | Output |
|-------|--------|
| `GradientField` + `Circle` + `ScoringConfig` | `SupportScore` |
| `GradientField` + `Ellipse` + `ScoringConfig` | `SupportScore` |

## Stage 3: Refine

**Goal.** Improve center position and shape parameters to subpixel accuracy.

Refinement operates in two layers:

1. **Center refinement** via the Parthasarathy radial center algorithm (Nature
   Methods, 2012). Each pixel in a local patch defines a line through its
   position along its gradient direction. The subpixel center is the
   least-squares intersection of all such lines — a closed-form weighted linear
   solve, non-iterative and fast.

2. **Shape refinement** via iterative least-squares. Given the refined center,
   gradient-weighted points on the annulus are fit to a circle (Kasa, 1976) or
   an ellipse (Fitzgibbon et al., 1999). The fit updates the shape parameters,
   a new annulus is sampled, and the process repeats until convergence or the
   iteration limit.

The output is a `RefinementResult<H>` parameterized by the hypothesis type
(`Circle` or `Ellipse`). It contains the refined hypothesis, an RMS residual,
the iteration count, and a `RefinementStatus` flag (`Converged`,
`MaxIterations`, `Degenerate`, or `OutOfBounds`).

### Key types

| Input | Output |
|-------|--------|
| `GradientField` + `Circle` + `CircleRefineConfig` | `RefinementResult<Circle>` |
| `GradientField` + `Ellipse` + `EllipseRefineConfig` | `RefinementResult<Ellipse>` |

## Using Stages Independently

Each stage is a free function with no hidden state. Common patterns:

- **Skip proposal generation** when centers are known from an external source
  (e.g., a coarse detector or user annotation). Jump straight to scoring or
  refinement with a manually constructed `Circle` or `Ellipse`.
- **Use scoring without refinement** for fast accept/reject decisions on a
  large batch of candidates.
- **Run refinement without scoring** when you trust the initial hypotheses and
  only need subpixel geometry.

## Homography-Aware Variants

When the target circle is viewed under perspective, it projects to an ellipse
in the image. radsym provides homography-aware versions of the propose and
refine stages:

- `frst_response_homography` runs FRST voting in a rectified coordinate frame,
  so that circles remain circular during voting even when the original image
  shows a perspective-distorted view.
- `refine_ellipse_homography` fits a circle in rectified space and maps the
  result back to an image-space ellipse via the homography.

These functions accept a `Homography` (a 3x3 projective transform) and a
`RectifiedGrid` that defines the resampled workspace.
