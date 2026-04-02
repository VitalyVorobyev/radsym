//! Detect ring-like structures using FRST + scoring + circle refinement.
//!
//! Usage:
//!   cargo run --example detect_rings --features image-io,serde [-- path/to/config.json]
//!
//! Default config: examples/configs/detect_rings.json

use std::fs;

use radsym::{
    extract_proposals, frst_response, load_grayscale, refine_circle, response_heatmap,
    save_diagnostic, score_circle_support, sobel_gradient, Circle, CircleRefineConfig, Colormap,
    FrstConfig, NmsConfig, Polarity, ScoringConfig,
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
    scoring: ScoringConfig,
    #[serde(default)]
    refinement: CircleRefineConfig,
    output_dir: Option<String>,
}

fn default_radius_hint() -> f32 {
    10.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/configs/detect_rings.json".into());

    let config: Config = serde_json::from_str(&fs::read_to_string(&config_path)?)?;

    println!("Loading image: {}", config.image_path);
    let image = load_grayscale(&config.image_path)?;
    println!("Image size: {}x{}", image.width(), image.height());

    let gradient = sobel_gradient(&image.view())?;

    let mut frst_config = config.frst.clone();
    frst_config.polarity = config.polarity;
    let response_map = frst_response(&gradient, &frst_config)?;
    let proposals = extract_proposals(&response_map, &config.nms, config.polarity);
    println!("Found {} proposals", proposals.len());

    // Score and refine each proposal
    println!(
        "{:>4} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "#", "x", "y", "radius", "score", "status"
    );
    println!("{}", "-".repeat(52));

    for (i, proposal) in proposals.iter().enumerate() {
        let center = proposal.seed.position;
        let circle = Circle::new(center, config.radius_hint);

        let score = score_circle_support(&gradient, &circle, &config.scoring);

        let refined = refine_circle(&gradient, &circle, &config.refinement)?;

        println!(
            "{:>4} {:>8.1} {:>8.1} {:>8.2} {:>8.3} {:>8}",
            i + 1,
            refined.hypothesis.center.x,
            refined.hypothesis.center.y,
            refined.hypothesis.radius,
            score.total,
            format!("{:?}", refined.status),
        );
    }

    // Optionally save heatmap overlay
    if let Some(ref dir) = config.output_dir {
        fs::create_dir_all(dir)?;
        let heatmap = response_heatmap(response_map.response(), Colormap::Hot);
        let path = format!("{dir}/detect_rings_heatmap.png");
        save_diagnostic(&heatmap, &path)?;
        println!("Saved heatmap to {path}");
    }

    Ok(())
}
