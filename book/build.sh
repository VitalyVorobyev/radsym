#!/bin/bash
# Build the radsym book with the interactive WASM demo.
#
# Usage: ./book/build.sh
#
# Prerequisites:
#   - wasm-pack (cargo install wasm-pack)
#   - mdbook    (cargo install mdbook)

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Building WASM package..."
wasm-pack build crates/radsym-wasm --target web --release

echo "==> Copying WASM assets to book/src/demo/..."
mkdir -p book/src/demo/pkg
cp crates/radsym-wasm/pkg/radsym_wasm.js    book/src/demo/pkg/
cp crates/radsym-wasm/pkg/radsym_wasm_bg.wasm book/src/demo/pkg/
cp testdata/ringgrid.png                     book/src/demo/ringgrid.png

echo "==> Building mdBook..."
mdbook build book

echo "==> Done. Output in book/book/"
