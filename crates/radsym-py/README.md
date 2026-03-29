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
- FRST and RSD proposal generation
- proposal extraction and homography-aware reranking
- circle and ellipse support scoring
- circle, ellipse, and radial-center refinement
- NumPy-friendly geometry, config, and result types

## Examples

See `examples/detect_ringgrid.py` and `examples/detect_surf_hole.py` for
end-to-end demo scripts built on the local extension module.

## License

Licensed under MIT OR Apache-2.0.
