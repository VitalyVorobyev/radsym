# Non-Maximum Suppression

## Overview

Non-maximum suppression (NMS) extracts discrete peak locations from a
continuous 2D response map (e.g., the output of [FRST](../algorithms/frst.md)
or [RSD](../algorithms/rsd.md)). It suppresses nearby weaker responses to
produce a sparse set of well-separated candidate centers.

## Algorithm

For each pixel in the response map:

1. Check if the pixel value exceeds `threshold`. Skip if not.
2. Compare against all pixels within a square window of half-size `radius`.
   The pixel must be strictly greater than every neighbor in the window to
   qualify as a peak.
3. Collect all qualifying peaks and sort by descending score.
4. Truncate to `max_detections` entries.

The result is a `Vec<Peak>`, where each `Peak` carries a `PixelCoord`
(sub-pixel position at integer coordinates) and a `score` (the response
value at that location). Peaks are returned in descending score order,
providing a natural ranking for downstream stages.

## Configuration

```rust
pub struct NmsConfig {
    /// Suppression radius (half-window size in pixels). Default: 5.
    pub radius: usize,
    /// Minimum response to consider. Default: 0.0.
    pub threshold: f32,
    /// Maximum number of peaks to return. Default: 1000.
    pub max_detections: usize,
}
```

| Field | Effect of increasing |
|-------|---------------------|
| `radius` | Fewer, more spread-out peaks |
| `threshold` | Eliminates weak candidates early |
| `max_detections` | Higher budget cap, more proposals to evaluate |
