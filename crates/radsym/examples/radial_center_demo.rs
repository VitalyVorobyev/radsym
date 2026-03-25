//! Demonstrate subpixel center refinement accuracy.
//!
//! Loads an image, detects proposals via FRST, then refines each using the
//! Parthasarathy radial center method. Reports residuals and convergence.
//!
//! Usage:
//!   cargo run --example radial_center_demo --features image-io,serde [-- path/to/config.json]
//!
//! Default config: examples/configs/radial_center.json

use std::fs;

use radsym::core::gradient::sobel_gradient;
use radsym::core::nms::NmsConfig;
use radsym::propose::extract::{extract_proposals, ResponseMap};
use radsym::propose::seed::ProposalSource;
use radsym::{
    frst_response, load_grayscale, radial_center_refine_from_gradient, FrstConfig, Polarity,
    RadialCenterConfig,
};

#[derive(serde::Deserialize)]
struct Config {
    image_path: String,
    #[serde(default)]
    polarity: Polarity,
    #[serde(default)]
    frst: FrstConfig,
    #[serde(default)]
    nms: NmsConfig,
    #[serde(default)]
    radial_center: RadialCenterConfig,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/configs/radial_center.json".into());

    let config: Config = serde_json::from_str(&fs::read_to_string(&config_path)?)?;

    println!("Loading image: {}", config.image_path);
    let image = load_grayscale(&config.image_path)?;
    println!("Image size: {}x{}", image.width(), image.height());

    let gradient = sobel_gradient(&image.view())?;

    let mut frst_config = config.frst.clone();
    frst_config.polarity = config.polarity;
    let response = frst_response(&gradient, &frst_config)?;

    let response_map = ResponseMap::new(response, ProposalSource::Frst);
    let proposals = extract_proposals(&response_map, &config.nms, config.polarity);
    println!("Found {} proposals\n", proposals.len());

    println!(
        "{:>4} {:>8} {:>8} {:>10} {:>10} {:>8} {:>10}",
        "#", "seed_x", "seed_y", "refined_x", "refined_y", "shift", "status"
    );
    println!("{}", "-".repeat(66));

    for (i, proposal) in proposals.iter().enumerate() {
        let seed = proposal.seed.position;

        let result = radial_center_refine_from_gradient(&gradient, seed, &config.radial_center);

        let refined = result.hypothesis;
        let shift = ((refined.x - seed.x).powi(2) + (refined.y - seed.y).powi(2)).sqrt();

        println!(
            "{:>4} {:>8.1} {:>8.1} {:>10.3} {:>10.3} {:>8.3} {:>10}",
            i + 1,
            seed.x,
            seed.y,
            refined.x,
            refined.y,
            shift,
            format!("{:?}", result.status),
        );
    }

    Ok(())
}
