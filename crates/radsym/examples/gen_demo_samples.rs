//! Generate the predefined sample images (and their manifest) for the web demo.
//!
//! Run from anywhere in the workspace:
//!
//! ```sh
//! cargo run -p radsym --example gen_demo_samples --features image-io
//! ```
//!
//! Writes deterministic PNGs plus `samples.json` into `demo/samples/` (resolved
//! relative to the workspace root, not the current directory). The existing
//! `testdata/ringgrid.png` is copied in as one of the samples.
//!
//! ## Coordinate convention
//!
//! Coordinates follow the library convention (ADR-001): integer `(x, y)` is the
//! **pixel center**. A disk centered at `(cx, cy)` therefore fills pixel
//! `(x, y)` according to the distance from `(cx, cy)` evaluated at the integer
//! pixel coordinate — the same rule the pipeline's synthetic tests use
//! (`pipeline.rs`). The ground-truth centers written to `samples.json` are in
//! this same convention, so the demo overlay must map an image coordinate `c`
//! to canvas `c + 0.5` (pixel-center → canvas corner origin).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use radsym::{OwnedImage, save_grayscale};

/// Dark background luma for the synthetic scenes.
const BG: f32 = 22.0;
/// Bright foreground luma for the synthetic features.
const FG: f32 = 235.0;

/// A float-valued grayscale canvas with 1px anti-aliased primitives.
struct Canvas {
    w: usize,
    h: usize,
    px: Vec<f32>,
}

impl Canvas {
    fn new(w: usize, h: usize, bg: f32) -> Self {
        Self {
            w,
            h,
            px: vec![bg; w * h],
        }
    }

    /// Alpha-composite `value` over the pixel at `(x, y)` with `coverage` in `[0, 1]`.
    fn blend(&mut self, x: usize, y: usize, value: f32, coverage: f32) {
        let i = y * self.w + x;
        self.px[i] = self.px[i] * (1.0 - coverage) + value * coverage;
    }

    /// Filled disk centered at `(cx, cy)` with radius `r`, 1px anti-aliased edge.
    fn disk(&mut self, cx: f32, cy: f32, r: f32, value: f32) {
        for y in 0..self.h {
            for x in 0..self.w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt() - r;
                let coverage = (0.5 - d).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    self.blend(x, y, value, coverage);
                }
            }
        }
    }

    /// Ring (annulus) centered at `(cx, cy)` at radius `r` with total thickness
    /// `2 * half_width`, 1px anti-aliased edges.
    fn ring(&mut self, cx: f32, cy: f32, r: f32, half_width: f32, value: f32) {
        for y in 0..self.h {
            for x in 0..self.w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let d = (dist - r).abs() - half_width;
                let coverage = (0.5 - d).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    self.blend(x, y, value, coverage);
                }
            }
        }
    }

    /// Filled ellipse centered at `(cx, cy)` with semi-axes `a`, `b` rotated by
    /// `angle` radians, 1px anti-aliased edge.
    fn ellipse(&mut self, cx: f32, cy: f32, a: f32, b: f32, angle: f32, value: f32) {
        let (sa, ca) = angle.sin_cos();
        let scale = a.min(b);
        for y in 0..self.h {
            for x in 0..self.w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let u = dx * ca + dy * sa;
                let v = -dx * sa + dy * ca;
                // Normalized radius; scale back to pixels for a ~1px AA band.
                let nr = ((u / a).powi(2) + (v / b).powi(2)).sqrt();
                let d = (nr - 1.0) * scale;
                let coverage = (0.5 - d).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    self.blend(x, y, value, coverage);
                }
            }
        }
    }

    /// Add deterministic, hash-based additive noise of amplitude `amp`.
    fn add_noise(&mut self, amp: f32, seed: u32) {
        for y in 0..self.h {
            for x in 0..self.w {
                let n = hash_noise(x as u32, y as u32, seed) - 0.5;
                let i = y * self.w + x;
                self.px[i] = (self.px[i] + n * 2.0 * amp).clamp(0.0, 255.0);
            }
        }
    }

    fn to_owned(&self) -> OwnedImage<u8> {
        let data: Vec<u8> = self.px.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
        OwnedImage::from_vec(data, self.w, self.h).expect("valid dimensions")
    }
}

/// Deterministic value-noise in `[0, 1)` from an integer coordinate + seed.
fn hash_noise(x: u32, y: u32, seed: u32) -> f32 {
    let mut h =
        x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ seed.wrapping_mul(83_492_791);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65_536.0
}

/// One ground-truth feature (image-space, pixel-center convention).
struct Gt {
    x: f32,
    y: f32,
    r: f32,
}

/// A generated sample plus its manifest metadata.
struct Sample {
    id: &'static str,
    label: &'static str,
    caption: &'static str,
    params: &'static str, // pre-rendered JSON object body
    gt: Vec<Gt>,
}

fn main() {
    let root = workspace_root();
    let out_dir = root.join("demo/samples");
    fs::create_dir_all(&out_dir).expect("create demo/samples");

    let mut manifest: Vec<Sample> = Vec::new();

    // 1) Ring grid — the existing real calibration target (copied, no GT).
    fs::copy(
        root.join("testdata/ringgrid.png"),
        out_dir.join("ringgrid.png"),
    )
    .expect("copy ringgrid.png");
    manifest.push(Sample {
        id: "ringgrid",
        label: "Ring grid",
        caption: "Real calibration target: a grid of concentric rings. Try Both polarity.",
        params: r#""algorithm":"frst","radii":[9,11,13,15,17],"polarity":"both","radiusHint":11,"minScore":0.0,"nmsRadius":8"#,
        gt: Vec::new(),
    });

    // 2) Concentric rings — the canonical radial-symmetry cue, one shared center.
    {
        let (w, h) = (320, 256);
        let (cx, cy) = (160.0, 128.0);
        let mut c = Canvas::new(w, h, BG);
        for &r in &[22.0, 42.0, 62.0, 82.0] {
            c.ring(cx, cy, r, 2.5, FG);
        }
        save_png(&c, &out_dir, "concentric-rings.png");
        manifest.push(Sample {
            id: "concentric-rings",
            label: "Concentric rings",
            caption: "Four rings sharing one center — pure radial symmetry. Every ring votes for the same point.",
            params: r#""algorithm":"frst","radii":[22,42,62,82],"polarity":"both","radiusHint":42,"minScore":0.0,"nmsRadius":12"#,
            gt: vec![Gt { x: cx, y: cy, r: 42.0 }],
        });
    }

    // 3) Dense field — many small disks; stresses NMS and density handling.
    {
        let (w, h) = (320, 256);
        let (r, spacing) = (7.0, 38.0);
        let (x0, y0) = (28.0, 28.0);
        let mut c = Canvas::new(w, h, BG);
        let mut gt = Vec::new();
        let cols = ((w as f32 - 2.0 * x0) / spacing) as usize + 1;
        let rows = ((h as f32 - 2.0 * y0) / spacing) as usize + 1;
        for j in 0..rows {
            for i in 0..cols {
                let cx = x0 + i as f32 * spacing;
                let cy = y0 + j as f32 * spacing;
                c.disk(cx, cy, r, FG);
                gt.push(Gt { x: cx, y: cy, r });
            }
        }
        save_png(&c, &out_dir, "dense-field.png");
        manifest.push(Sample {
            id: "dense-field",
            label: "Dense field",
            caption: "A regular grid of small bright disks. Lower the NMS radius to keep neighbours apart.",
            params: r#""algorithm":"frst","radii":[6,7,8],"polarity":"bright","radiusHint":7,"minScore":0.0,"nmsRadius":14,"maxDetections":200"#,
            gt,
        });
    }

    // 4) Perspective ellipses — circles seen under tilt; motivates ellipse/homography.
    {
        let (w, h) = (320, 256);
        let mut c = Canvas::new(w, h, BG);
        let mut gt = Vec::new();
        // A row of circles on a plane tilted about the vertical axis: eccentricity
        // and tilt grow toward the right edge.
        let specs = [
            (70.0, 90.0, 20.0, 19.0, 0.05),
            (150.0, 100.0, 22.0, 16.0, 0.25),
            (225.0, 120.0, 24.0, 13.0, 0.45),
            (140.0, 185.0, 21.0, 12.0, -0.35),
            (235.0, 195.0, 26.0, 11.0, 0.6),
        ];
        for &(cx, cy, a, b, ang) in &specs {
            c.ellipse(cx, cy, a, b, ang, FG);
            // The circle detector recovers the center; the enclosing-circle radius
            // is the semi-major axis. Ground truth is the center.
            gt.push(Gt { x: cx, y: cy, r: a });
        }
        save_png(&c, &out_dir, "perspective-ellipses.png");
        manifest.push(Sample {
            id: "perspective-ellipses",
            label: "Perspective ellipses",
            caption: "Circles under perspective become ellipses. Circle refinement finds the centers but not the true shape — this is where ellipse/homography refinement helps.",
            params: r#""algorithm":"frst","radii":[11,14,17,20],"polarity":"bright","radiusHint":15,"minScore":0.0,"nmsRadius":16"#,
            gt,
        });
    }

    // 5) Low-contrast + noise — robustness of gradient voting under poor SNR.
    {
        let (w, h) = (320, 256);
        let mut c = Canvas::new(w, h, 60.0);
        let mut gt = Vec::new();
        let specs = [
            (90.0, 90.0, 16.0),
            (210.0, 95.0, 20.0),
            (150.0, 180.0, 22.0),
        ];
        for &(cx, cy, r) in &specs {
            c.disk(cx, cy, r, 96.0); // only ~36 luma above background
            gt.push(Gt { x: cx, y: cy, r });
        }
        c.add_noise(12.0, 0x9e37);
        save_png(&c, &out_dir, "low-contrast-noisy.png");
        manifest.push(Sample {
            id: "low-contrast-noisy",
            label: "Low contrast + noise",
            caption: "Faint disks buried in noise. Raise the gradient threshold and smoothing to suppress noise votes.",
            params: r#""algorithm":"frst","radii":[14,16,18,20,22],"polarity":"bright","radiusHint":18,"minScore":0.0,"gradientThreshold":8.0,"smoothingFactor":1.0,"nmsRadius":16"#,
            gt,
        });
    }

    // 6) Mixed radii — one image spanning a wide size range (multi-radius voting).
    {
        let (w, h) = (320, 256);
        let mut c = Canvas::new(w, h, BG);
        let mut gt = Vec::new();
        let specs = [
            (64.0, 72.0, 8.0),
            (170.0, 80.0, 14.0),
            (110.0, 175.0, 22.0),
            (240.0, 165.0, 30.0),
        ];
        for &(cx, cy, r) in &specs {
            c.disk(cx, cy, r, FG);
            gt.push(Gt { x: cx, y: cy, r });
        }
        save_png(&c, &out_dir, "mixed-radii.png");
        manifest.push(Sample {
            id: "mixed-radii",
            label: "Mixed radii",
            caption: "Disks from 8px to 30px in one frame. The radii list must span every size you want detected.",
            params: r#""algorithm":"frst","radii":[8,12,16,20,24,28,30],"polarity":"bright","radiusHint":16,"minScore":0.0,"nmsRadius":18"#,
            gt,
        });
    }

    write_manifest(&out_dir, &manifest);
    println!(
        "Wrote {} samples + samples.json to {}",
        manifest.len(),
        out_dir.display()
    );
}

fn save_png(canvas: &Canvas, dir: &Path, name: &str) {
    save_grayscale(&canvas.to_owned(), dir.join(name)).expect("save png");
}

/// Emit `samples.json` (hand-written to avoid a serde dependency on the example).
fn write_manifest(dir: &Path, samples: &[Sample]) {
    let mut s = String::new();
    s.push_str("{\n  \"samples\": [\n");
    for (si, sample) in samples.iter().enumerate() {
        s.push_str("    {\n");
        let _ = writeln!(s, "      \"id\": \"{}\",", sample.id);
        let _ = writeln!(s, "      \"label\": \"{}\",", sample.label);
        let _ = writeln!(s, "      \"file\": \"samples/{}.png\",", sample.id);
        let _ = writeln!(s, "      \"caption\": {},", json_string(sample.caption));
        let _ = writeln!(s, "      \"params\": {{{}}},", sample.params);
        if sample.gt.is_empty() {
            s.push_str("      \"groundTruth\": null\n");
        } else {
            s.push_str("      \"groundTruth\": [\n");
            for (gi, g) in sample.gt.iter().enumerate() {
                let comma = if gi + 1 < sample.gt.len() { "," } else { "" };
                let _ = writeln!(
                    s,
                    "        {{ \"x\": {:.1}, \"y\": {:.1}, \"r\": {:.1} }}{}",
                    g.x, g.y, g.r, comma
                );
            }
            s.push_str("      ]\n");
        }
        s.push_str(if si + 1 < samples.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    s.push_str("  ]\n}\n");
    fs::write(dir.parent().unwrap().join("samples.json"), s).expect("write samples.json");
}

/// Minimal JSON string escaping for the caption text.
fn json_string(v: &str) -> String {
    let mut out = String::from("\"");
    for ch in v.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// Workspace root, resolved from this crate's manifest dir (CWD-independent).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}
