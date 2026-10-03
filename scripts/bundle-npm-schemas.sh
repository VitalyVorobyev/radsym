#!/usr/bin/env bash
# Add the committed JSON Schemas (schemas/*.json) to the wasm-pack output so
# they ship in the npm tarball as `schemas/<name>.json`.
#
# Run after `wasm-pack build crates/radsym-wasm --target web --release`.
# Idempotent: `schemas` is appended to package.json `files` only if absent, and
# the existing entries (wasm, js, d.ts) are kept.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg="$root/crates/radsym-wasm/pkg"

[ -f "$pkg/package.json" ] || { echo "error: $pkg/package.json missing; run wasm-pack first" >&2; exit 1; }
ls "$root"/schemas/*.json >/dev/null

mkdir -p "$pkg/schemas"
cp "$root"/schemas/*.json "$pkg/schemas/"

node -e '
const fs = require("fs");
const file = process.argv[1];
const pkg = JSON.parse(fs.readFileSync(file, "utf8"));
const files = Array.isArray(pkg.files) ? pkg.files : [];
if (!files.includes("schemas")) files.push("schemas");
pkg.files = files;
fs.writeFileSync(file, JSON.stringify(pkg, null, 2) + "\n");
' "$pkg/package.json"
