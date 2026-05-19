# radsym-py

`radsym-py` is the PyO3 extension crate behind the Python `radsym` package.
It exposes the Rust `radsym` algorithms to NumPy-based Python workflows.

This crate is not meant to be used as a standalone Rust dependency. For Rust
code, depend on `radsym` directly.

## Build the Python extension

From the workspace root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin numpy
maturin develop --release --manifest-path crates/radsym-py/Cargo.toml
```

## What it exposes

- Sobel gradient computation
- FRST and RSD proposal generation, including the fused single-pass variants
  `frst_response_fused` and `rsd_response_fused`
- proposal extraction and homography-aware reranking
- circle and ellipse support scoring
- circle, ellipse, and radial-center refinement
- NumPy-friendly geometry, config, and result types

## Result conventions

- `score_circle_support` / `score_ellipse_support` return a `SupportScore`
  object that exposes `total`, `ringness`, `angular_coverage`, and
  `is_degenerate`.
- A `Detection` from the one-call `detect_circles` pipeline exposes its
  headline support score directly as a `float` via `Detection.score` — it is
  the `total` only, not a structured object.

## Examples

See `examples/detect_ringgrid.py` and `examples/detect_surf_hole.py` for
end-to-end demo scripts built on the local extension module, plus the
`detect_ringgrid_frst.py`, `detect_ringgrid_frst_fused.py`, and
`detect_ringgrid_rsd_fused.py` proposal-generator variants.

## License

Licensed under MIT OR Apache-2.0.
