# Python Demo Scripts

This directory contains small CLI demos built on the `radsym` Python bindings.

## Requirements

- Python 3.9+
- Rust toolchain
- `maturin`
- `numpy`
- `matplotlib`
- `rich`

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin numpy matplotlib rich
maturin develop --release --manifest-path crates/radsym-py/Cargo.toml
```

This builds and installs the local `radsym` extension module into the active
virtual environment.

If you `cd crates/radsym-py` first, use:

```bash
maturin develop --release
```

Do not run `maturin develop .` from the crate directory. That trailing `.`
gets forwarded into `cargo rustc` as an extra input filename and breaks the
build.

## Demos

### `detect_surf_hole.py`

Detects a single central hole-like structure in a `surf*.png` image.

The detection runs on an 8x downscaled working image by default, then maps the
refined ellipse back to full resolution for display.

Example:

```bash
python crates/radsym-py/examples/detect_surf_hole.py testdata/surf1.png
```

With no output flags, the script opens matplotlib windows instead of writing files.

Optional outputs:

```bash
python crates/radsym-py/examples/detect_surf_hole.py \
  testdata/surf4.png \
  --downscale 8 \
  --output output/surf4_overlay.png \
  --heatmap-output output/surf4_heatmap.png
```

What it does:

- loads the image with `radsym.load_grayscale`
- computes FRST proposals
- refines center candidates, then runs a post-center radius sweep
- runs an ellipse-shape sweep before final ellipse refinement
- ranks candidates by support score plus closeness to the image center
- refines the best candidate as an ellipse
- shows the annulus sample points that contribute to the final ellipse fit
- renders:
  - a matplotlib overlay on the source image
  - a matplotlib FRST heatmap overlay
- accepts `--downscale` to control the working resolution, default `8`
- prints a `rich` summary and performance table for the main `radsym` calls

### `detect_ringgrid.py`

Detects many ring-like structures in an input image such as `ringgrid.png`.

Example:

```bash
python crates/radsym-py/examples/detect_ringgrid.py testdata/ringgrid.png
```

With no output flags, the script opens matplotlib windows instead of writing files.

Optional outputs:

```bash
python crates/radsym-py/examples/detect_ringgrid.py \
  testdata/ringgrid.png \
  --output output/ringgrid_overlay.png \
  --heatmap-output output/ringgrid_heatmap.png
```

What it does:

- loads the image with `radsym.load_grayscale`
- computes outer-radius proposal responses with `RSD` by default, or `FRST` with `--detector frst`
- extracts and deduplicates center proposals
- runs in proposals-only mode by default
- optionally fits outer and inner ellipses with `--fit-ellipses`
- renders:
  - a matplotlib overlay on the source image
  - a matplotlib response heatmap overlay
- prints a `rich` summary and per-call `radsym` timing table

Ellipse fitting example:

```bash
python crates/radsym-py/examples/detect_ringgrid.py \
  testdata/ringgrid.png \
  --fit-ellipses \
  --output output/ringgrid_overlay.png \
  --heatmap-output output/ringgrid_heatmap.png
```

## Notes

- The scripts do not hard-code input images; the image path is always a CLI argument.
- By default the demos display interactive matplotlib figures.
- Files are only written when you pass `--output` and/or `--heatmap-output`.
- For realistic timings, install the Python extension with `maturin develop --release`.
- The demos assume the image polarity defaults that match the bundled test data:
  - `detect_surf_hole.py`: `--polarity bright`
  - `detect_ringgrid.py`: `--polarity dark`
