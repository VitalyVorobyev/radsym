//! Generate FRST heatmap and circle overlay diagnostic images.
//!
//! Usage:
//!   cargo run --example diagnostics_demo --features image-io,serde [-- path/to/config.json]
//!
//! Default config: examples/configs/diagnostics.json

use std::fs;

use radsym::diagnostics::overlay::overlay_proposals;
use radsym::{
    extract_proposals, frst_response, load_grayscale, overlay_circle, refine_circle,
    response_heatmap, save_diagnostic, sobel_gradient, Circle, CircleRefineConfig, Colormap,
    DiagnosticImage, FrstConfig, NmsConfig, Polarity,
};

#[derive(serde::Deserialize)]
struct Config {
    image_path: String,
    #[serde(default)]
    polarity: Polarity,
    #[serde(default = "default_radius_hint")]
    radius_hint: f32,
    #[serde(default)]
    frst: FrstConfig,
    #[serde(default)]
    nms: NmsConfig,
    #[serde(default)]
    refinement: CircleRefineConfig,
    #[serde(default = "default_output_dir")]
    output_dir: String,
}

fn default_radius_hint() -> f32 {
    12.0
}

fn default_output_dir() -> String {
    "output".into()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/configs/diagnostics.json".into());

    let config: Config = serde_json::from_str(&fs::read_to_string(&config_path)?)?;

    println!("Loading image: {}", config.image_path);
    let image = load_grayscale(&config.image_path)?;
    let w = image.width();
    let h = image.height();
    println!("Image size: {w}x{h}");

    let gradient = sobel_gradient(&image.view())?;

    let mut frst_config = config.frst.clone();
    frst_config.polarity = config.polarity;
    let response_map = frst_response(&gradient, &frst_config)?;
    let proposals = extract_proposals(&response_map, &config.nms, config.polarity);
    println!("Found {} proposals", proposals.len());

    fs::create_dir_all(&config.output_dir)?;

    // 1. FRST heatmap
    let heatmap = response_heatmap(response_map.response(), Colormap::Hot);
    let heatmap_path = format!("{}/frst_heatmap.png", config.output_dir);
    save_diagnostic(&heatmap, &heatmap_path)?;
    println!("Saved heatmap: {heatmap_path}");

    // 2. Grayscale image with circle overlays
    let mut overlay = DiagnosticImage::new(w, h);
    // Copy grayscale to RGBA
    let img_data = image.data();
    for y in 0..h {
        for x in 0..w {
            let v = img_data[y * w + x];
            overlay.set_pixel(x, y, [v, v, v, 255]);
        }
    }

    // Draw proposal markers (green)
    overlay_proposals(&mut overlay, &proposals, [0, 255, 0, 255], 3);

    // Refine and draw circles (red)
    for proposal in &proposals {
        let circle = Circle::new(proposal.seed.position, config.radius_hint);
        let refined = refine_circle(&gradient, &circle, &config.refinement)?;
        overlay_circle(&mut overlay, &refined.hypothesis, [255, 0, 0, 255]);
    }

    let overlay_path = format!("{}/circle_overlay.png", config.output_dir);
    save_diagnostic(&overlay, &overlay_path)?;
    println!("Saved overlay: {overlay_path}");

    Ok(())
}
