# Introduction

**radsym** is a Rust library for detecting circles and ellipses in grayscale
images using gradient-based radial symmetry analysis. It turns raw pixel data
into precise geometric detections — center coordinates, radii, and ellipse
parameters — without requiring template matching, Hough accumulator grids, or
deep learning models.

## Design Philosophy

The library is built around four principles:

- **Composable stages.** Every algorithm is a standalone function with explicit
  inputs and outputs. You can use the full propose-score-refine pipeline, or
  pick individual stages and wire them into your own workflow.
- **CPU-first.** All computation runs on the CPU with no GPU or OpenCL
  dependency. The optional `rayon` feature flag enables data-parallel execution
  across multiple cores.
- **Deterministic.** Given the same input and configuration, radsym produces
  bit-identical output. There is no internal randomness and no
  ordering-dependent accumulation.
- **`f32` precision.** Pixel coordinates, gradient values, scores, and
  geometric parameters are all single-precision floats. This matches the
  accuracy regime of typical machine-vision sensors and avoids unnecessary
  bandwidth and cache pressure from `f64`.

## Who Is This For?

radsym targets developers working on computer vision tasks that involve circular
or elliptical features: industrial inspection (O-rings, bearings, nozzles),
metrology (fiducial rings, calibration targets), microscopy (particle tracking),
and autonomous systems (traffic sign detection, wheel localization).

## The Three-Stage Pipeline

Detection follows a propose-score-refine architecture:

1. **Propose.** Gradient voting (FRST or RSD) builds a response map that peaks
   at likely centers of radial symmetry. Non-maximum suppression extracts a
   sparse set of candidate locations.
2. **Score.** Each proposal is evaluated by sampling gradients in an annulus
   around the hypothesized circle or ellipse. The resulting `SupportScore`
   combines a *ringness* metric (gradient alignment) with *angular coverage*
   (how much of the circumference has evidence).
3. **Refine.** Surviving hypotheses are refined to subpixel accuracy using the
   Parthasarathy radial center algorithm for center position and iterative
   least-squares fitting for radius and ellipse parameters.

For the common case of detecting circles in a single image, the convenience
function `detect_circles` wraps all three stages into one call, configurable
through `DetectCirclesConfig`.
