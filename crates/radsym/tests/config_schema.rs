//! The JSON Schema of `DetectCirclesConfig` must describe the serde shape
//! exactly, and partial JSON configs must fall back to the defaults.

#![cfg(feature = "schemars")]

use radsym::pipeline::DetectCirclesConfig;
use radsym::{GradientOperator, Polarity, Rect};
use schemars::schema_for;
use serde_json::{Value, json};

fn schema() -> Value {
    serde_json::to_value(schema_for!(DetectCirclesConfig)).unwrap()
}

fn validator() -> jsonschema::Validator {
    jsonschema::validator_for(&schema()).expect("generated schema is a valid JSON Schema")
}

#[test]
fn default_config_validates_against_schema() {
    let instance = serde_json::to_value(DetectCirclesConfig::default()).unwrap();
    let v = validator();
    let errors: Vec<String> = v.iter_errors(&instance).map(|e| e.to_string()).collect();
    assert!(errors.is_empty(), "default config rejected: {errors:?}");
}

#[test]
fn populated_config_validates_against_schema() {
    let config = DetectCirclesConfig::for_radii([8, 12])
        .polarity(Polarity::Dark)
        .radius_hint(9.5)
        .min_score(0.25)
        .gradient_operator(GradientOperator::Scharr)
        .roi(Rect::new(4, 8, 100, 60));
    let instance = serde_json::to_value(&config).unwrap();
    assert!(validator().is_valid(&instance));
}

#[test]
fn schema_rejects_out_of_range_values() {
    let v = validator();
    assert!(!v.is_valid(&json!({ "radii": [] })));
    assert!(!v.is_valid(&json!({ "radii": [0, 5] })));
    assert!(!v.is_valid(&json!({ "min_score": 1.5 })));
    assert!(!v.is_valid(&json!({ "radius_hint": 0.0 })));
    assert!(!v.is_valid(&json!({ "polarity": "Sideways" })));
    assert!(!v.is_valid(&json!({ "advanced": { "nms": { "radius": 0 } } })));
    assert!(!v.is_valid(&json!({ "advanced": { "frst": { "smoothing_factor": 0.0 } } })));
    assert!(!v.is_valid(&json!({ "roi": { "x": 0, "y": 0, "width": 0, "height": 5 } })));
}

#[test]
fn schema_accepts_partial_configs() {
    // Every field has a default, so none is required.
    let v = validator();
    assert!(v.is_valid(&json!({})));
    assert!(v.is_valid(&json!({ "radii": [4, 6], "polarity": "Bright" })));
    assert!(v.is_valid(&json!({ "advanced": { "nms": { "radius": 7 } } })));
}

#[test]
fn partial_json_deserializes_over_defaults() {
    let config: DetectCirclesConfig =
        serde_json::from_value(json!({ "radii": [4, 6], "polarity": "Bright" })).unwrap();
    let expected = DetectCirclesConfig::for_radii([4, 6]).polarity(Polarity::Bright);
    // `DetectCirclesConfig` has no `PartialEq`; compare the serde shape.
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
}

#[test]
fn partial_nested_json_keeps_sibling_defaults() {
    let config: DetectCirclesConfig =
        serde_json::from_value(json!({ "advanced": { "nms": { "radius": 7 } } })).unwrap();
    let mut expected = DetectCirclesConfig::default();
    expected.advanced.nms.radius = 7;
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
    // Siblings of the overridden field are untouched.
    assert_eq!(config.advanced.nms.max_detections, 1000);
    assert_eq!(config.advanced.scoring.min_samples, 8);
}

#[test]
fn empty_object_is_the_default_config() {
    let config: DetectCirclesConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::to_value(DetectCirclesConfig::default()).unwrap()
    );
}

#[test]
fn schema_defaults_match_the_default_config() {
    // The `default` annotations a form would pre-fill are the real defaults.
    let schema = schema();
    let defaults = serde_json::to_value(DetectCirclesConfig::default()).unwrap();
    for (field, value) in defaults.as_object().unwrap() {
        assert_eq!(
            &schema["properties"][field]["default"], value,
            "schema default for `{field}` differs from DetectCirclesConfig::default()"
        );
    }
}
