#!/bin/bash
# Build the radsym book with the interactive WASM demo.
#
# Usage: ./book/build.sh
#
# Prerequisites:
#   - wasm-pack (cargo install wasm-pack)
#   - mdbook    (cargo install mdbook)
#
# The demo has a single canonical source in `demo/`. This script builds the
# WASM package and stages `demo/` (HTML/CSS/JS + sample images) together with
# the freshly built `pkg/` into `book/src/demo/` — a generated, gitignored
# directory — before rendering the book.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Building WASM package..."
wasm-pack build crates/radsym-wasm --target web --release

echo "==> Staging demo/ into book/src/demo/..."
rm -rf book/src/demo
mkdir -p book/src/demo/pkg book/src/demo/samples
cp demo/index.html demo/app.js demo/styles.css demo/vv-logo.svg demo/samples.json book/src/demo/
cp demo/samples/*                                                                  book/src/demo/samples/
cp crates/radsym-wasm/pkg/radsym_wasm.js                                           book/src/demo/pkg/
cp crates/radsym-wasm/pkg/radsym_wasm_bg.wasm                                      book/src/demo/pkg/

echo "==> Building mdBook..."
mdbook build book

echo "==> Done. Output in book/book/"
