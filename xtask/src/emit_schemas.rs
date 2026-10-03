//! Emit JSON Schemas for the user-facing config types.
//!
//! Output goes to `schemas/<name>.json` at the repository root. With
//! `--check`, the command instead verifies that the committed files match what
//! the current source generates; CI runs this to catch drift.

use anyhow::{Context, Result, bail};
use radsym::pipeline::DetectCirclesConfig;
use schemars::{JsonSchema, schema_for};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub fn run(workspace_root: &Path, check: bool) -> Result<()> {
    let out_dir = workspace_root.join("schemas");

    let entries: Vec<(&str, Value)> = vec![(
        "detect_circles_config",
        schema_value::<DetectCirclesConfig>(),
    )];

    let mut drift = Vec::new();
    if !check {
        std::fs::create_dir_all(&out_dir)
            .with_context(|| format!("creating {}", out_dir.display()))?;
    }
    for (name, schema) in &entries {
        let path = out_dir.join(format!("{name}.json"));
        write_or_check(&path, schema, check, &mut drift)?;
    }

    if !check {
        println!(
            "emitted {} schema(s) to {}",
            entries.len(),
            out_dir.display()
        );
        return Ok(());
    }
    if drift.is_empty() {
        println!("schemas up to date ({} file(s))", entries.len());
        return Ok(());
    }
    for entry in &drift {
        entry.report();
    }
    let hint = if drift.iter().any(|d| matches!(d, Drift::LineEndings(_))) {
        "; the CRLF ones need an LF checkout (`git add --renormalize .`), not a re-emit"
    } else {
        ""
    };
    bail!(
        "{} schema(s) out of date; run `cargo xtask emit-schemas` and commit the result{hint}",
        drift.len()
    )
}

fn schema_value<T: JsonSchema>() -> Value {
    let mut value =
        serde_json::to_value(schema_for!(T)).expect("JsonSchema serialization is infallible");
    tidy_schema(&mut value);
    value
}

/// Make the generated schema readable for form UIs, without touching its
/// meaning:
///
/// * Type-level descriptions (root and `$defs`) keep only their first
///   paragraph; the rest of a type's rustdoc is developer-oriented.
/// * Rustdoc intra-doc links are flattened to plain text.
/// * `f32` defaults, which schemars widens to `f64` (`0.3` becomes
///   `0.30000001192092896`), are restored to their shortest `f32` form.
fn tidy_schema(root: &mut Value) {
    if let Some(obj) = root.as_object_mut() {
        first_paragraph(obj);
        if let Some(Value::Object(defs)) = obj.get_mut("$defs") {
            for def in defs.values_mut() {
                if let Some(def) = def.as_object_mut() {
                    first_paragraph(def);
                }
            }
        }
    }
    tidy_node(root);
}

fn first_paragraph(obj: &mut serde_json::Map<String, Value>) {
    if let Some(Value::String(desc)) = obj.get_mut("description")
        && let Some((head, _)) = desc.split_once("\n\n")
    {
        *desc = head.to_owned();
    }
}

fn tidy_node(node: &mut Value) {
    match node {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                match child {
                    Value::String(desc) if key == "description" => {
                        *desc = strip_doc_links(desc);
                    }
                    _ => tidy_node(child),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(tidy_node),
        Value::Number(n) => {
            if n.is_f64()
                && let Some(f) = n.as_f64()
            {
                let narrow = f as f32;
                // Only numbers that are exactly an `f32` widened to `f64`.
                if f64::from(narrow) == f
                    && let Some(short) = format!("{narrow}")
                        .parse::<f64>()
                        .ok()
                        .and_then(serde_json::Number::from_f64)
                {
                    *node = Value::Number(short);
                }
            }
        }
        _ => {}
    }
}

/// Flatten rustdoc links: ``[`x`]`` and ``[`x`](path)`` become ``` `x` ```,
/// and reference definitions (`[`x`]: path`) are dropped.
fn strip_doc_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("[`") && trimmed.contains("]: ") {
            continue;
        }
        let mut rest = line;
        while let Some(start) = rest.find("[`") {
            let after = &rest[start + 1..];
            let Some(end) = after.find("`]") else { break };
            out.push_str(&rest[..start]);
            out.push_str(&after[..end + 1]);
            rest = &after[end + 2..];
            if let Some(tail) = rest.strip_prefix('(')
                && let Some(close) = tail.find(')')
            {
                rest = &tail[close + 1..];
            }
        }
        out.push_str(rest);
        out.push('\n');
    }
    out.truncate(out.trim_end().len());
    out
}

/// Why a committed schema no longer matches what the generator produces.
enum Drift {
    /// The file is missing or its content differs: a config type changed and
    /// the schema was not re-emitted.
    Stale(PathBuf),
    /// Content matches once CRLF is normalised: a wrong-line-ending checkout,
    /// not a stale schema. `.gitattributes` (`eol=lf`) prevents this.
    LineEndings(PathBuf),
}

impl Drift {
    fn report(&self) {
        match self {
            Self::Stale(path) => eprintln!("schema drift: {}", path.display()),
            Self::LineEndings(path) => {
                eprintln!("line-ending drift (CRLF on disk): {}", path.display());
            }
        }
    }
}

fn write_or_check(path: &Path, schema: &Value, check: bool, drift: &mut Vec<Drift>) -> Result<()> {
    let mut text =
        serde_json::to_string_pretty(schema).context("rendering schema as pretty JSON")?;
    text.push('\n');

    if check {
        match std::fs::read_to_string(path) {
            Ok(on_disk) if on_disk == text => {}
            Ok(on_disk) if on_disk.replace("\r\n", "\n") == text => {
                drift.push(Drift::LineEndings(path.to_path_buf()));
            }
            _ => drift.push(Drift::Stale(path.to_path_buf())),
        }
        return Ok(());
    }

    std::fs::write(path, &text).with_context(|| format!("writing {}", path.display()))
}
