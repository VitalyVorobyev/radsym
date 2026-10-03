#!/usr/bin/env bash
# Assert that the npm tarball built from the wasm-pack output contains the
# JSON Schemas. Run after scripts/bundle-npm-schemas.sh.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/crates/radsym-wasm/pkg"

listing="$(npm pack --dry-run --json | node -e '
const out = JSON.parse(require("fs").readFileSync(0, "utf8"));
for (const f of out[0].files) console.log(f.path);
')"
echo "$listing"

for required in schemas/detect_circles_config.json radsym_wasm_bg.wasm radsym_wasm.js radsym_wasm.d.ts; do
  if ! grep -qx "$required" <<<"$listing"; then
    echo "error: npm tarball is missing $required" >&2
    exit 1
  fi
done
echo "npm tarball contains the expected files"
