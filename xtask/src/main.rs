//! Workspace task runner. Currently exposes `emit-schemas`, which writes the
//! JSON Schema of the circle-detection config into `schemas/`. The schema is
//! shipped inside the npm package so a schema-driven form can edit the config.

use anyhow::{Result, bail};
use std::path::PathBuf;

mod emit_schemas;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        bail!("usage: cargo xtask <command>\n\ncommands:\n  emit-schemas [--check]");
    };
    match cmd.as_str() {
        "emit-schemas" => {
            let mut check = false;
            for arg in args {
                match arg.as_str() {
                    "--check" => check = true,
                    other => bail!("unknown argument `{other}`; usage: emit-schemas [--check]"),
                }
            }
            emit_schemas::run(&workspace_root(), check)
        }
        other => bail!("unknown xtask `{other}`; available: emit-schemas [--check]"),
    }
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at xtask/, so the workspace root is its parent.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask/ has a parent directory")
        .to_path_buf()
}
