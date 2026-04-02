//! Detect a single circular/elliptical highlight and refine as ellipse.
//!
//! Usage:
//!   cargo run --example detect_highlight --features image-io,serde [-- path/to/config.json]
//!
//! Default config: examples/configs/detect_highlight.json

use std::fs;

use radsym::support::score::ScoringConfig;
use radsym::{
    extract_proposals, frst_response, load_grayscale, refine_ellipse, score_circle_support,
    sobel_gradient, Circle, Ellipse, EllipseRefineConfig, FrstConfig, NmsConfig, Polarity,
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
    ellipse_refinement: EllipseRefineConfig,
}

fn default_radius_hint() -> f32 {
    30.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/configs/detect_highlight.json".into());

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

    println!(
        "{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "#", "cx", "cy", "a", "b", "angle", "score", "status"
    );
    println!("{}", "-".repeat(68));

    for (i, proposal) in proposals.iter().enumerate() {
        let center = proposal.seed.position;
        let r = config.radius_hint;
        let circle = Circle::new(center, r);
        let score = score_circle_support(&gradient, &circle, &config.scoring);

        // Start ellipse refinement from a circular initial guess
        let initial = Ellipse::new(center, r, r, 0.0);
        let refined = refine_ellipse(&gradient, &initial, &config.ellipse_refinement).unwrap();

        let e = &refined.hypothesis;
        println!(
            "{:>4} {:>8.1} {:>8.1} {:>8.2} {:>8.2} {:>8.3} {:>8.3} {:>8}",
            i + 1,
            e.center.x,
            e.center.y,
            e.semi_major,
            e.semi_minor,
            e.angle.to_degrees(),
            score.total,
            format!("{:?}", refined.status),
        );
    }

    Ok(())
}
