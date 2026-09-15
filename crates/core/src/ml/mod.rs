use crate::types::Gpu;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub schema_version: String,
    pub model_type: String,
    pub base_score: f64,
    pub objective: String,
    pub num_trees: usize,
    pub features: Vec<String>,
    pub trees: Vec<Tree>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub split_feature: Option<usize>,
    pub threshold: Option<f64>,
    pub yes: Option<usize>,
    pub no: Option<usize>,
    pub missing: Option<usize>,
    pub leaf_value: Option<f64>,
}

pub struct XgbEvaluator {
    artifact: ModelArtifact,
    tree_node_maps: Vec<HashMap<usize, Node>>,
}

impl XgbEvaluator {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let artifact: ModelArtifact = serde_json::from_str(&content)?;
        Self::from_artifact(artifact)
    }

    pub fn from_artifact(artifact: ModelArtifact) -> anyhow::Result<Self> {
        let mut tree_node_maps = Vec::with_capacity(artifact.trees.len());
        for tree in &artifact.trees {
            let mut node_map = HashMap::new();
            for node in &tree.nodes {
                node_map.insert(node.id, node.clone());
            }
            tree_node_maps.push(node_map);
        }
        Ok(Self {
            artifact,
            tree_node_maps,
        })
    }

    /// Evaluates raw ensemble margin score
    pub fn evaluate_raw(&self, features: &[f64]) -> f64 {
        let mut total_score = 0.0;

        for node_map in &self.tree_node_maps {
            let mut curr_id = 0;
            loop {
                let node = match node_map.get(&curr_id) {
                    Some(n) => n,
                    None => break,
                };

                if let Some(leaf) = node.leaf_value {
                    total_score += leaf;
                    break;
                }

                let split_idx = match node.split_feature {
                    Some(idx) => idx,
                    None => break,
                };

                let threshold = match node.threshold {
                    Some(t) => t,
                    None => break,
                };

                let val = if split_idx < features.len() {
                    features[split_idx]
                } else {
                    f64::NAN
                };

                if val.is_nan() {
                    curr_id = node.missing.unwrap_or(node.yes.unwrap_or(0));
                } else if val <= threshold {
                    curr_id = match node.yes {
                        Some(y) => y,
                        None => break,
                    };
                } else {
                    curr_id = match node.no {
                        Some(n) => n,
                        None => break,
                    };
                }
            }
        }

        total_score
    }

    /// Evaluates probability under logistic sigmoid
    pub fn predict_probability(&self, features: &[f64]) -> f64 {
        let raw = self.evaluate_raw(features);
        // Base margin: logit(base_score)
        let base_margin = if (self.artifact.base_score - 0.5).abs() < 1e-6 {
            0.0
        } else {
            (self.artifact.base_score / (1.0 - self.artifact.base_score)).ln()
        };

        let logit = raw + base_margin;
        1.0 / (1.0 + (-logit).exp())
    }

    /// Extracts feature vector from a GPU according to canonical feature definitions
    pub fn extract_features(&self, gpu: &Gpu) -> Vec<f64> {
        let temp = gpu.temperature;
        let power = gpu.power;
        let perf = gpu.performance;
        let ecc_rate = gpu.ecc_window_errors as f64;
        let nvlink_rate = gpu.nvlink_errors as f64;

        // Compute rolling EWMA approximations from history buffer
        let ewma_5m_temp = if !gpu.history.is_empty() {
            let samples: Vec<f64> = gpu.history.iter().rev().take(10).map(|s| s.temperature).collect();
            calculate_ewma(&samples, 0.2)
        } else {
            temp
        };

        let ewma_15m_temp = if !gpu.history.is_empty() {
            let samples: Vec<f64> = gpu.history.iter().rev().take(30).map(|s| s.temperature).collect();
            calculate_ewma(&samples, 0.07)
        } else {
            temp
        };

        let ewma_5m_ecc = ecc_rate;
        let ewma_15m_ecc = ecc_rate;
        let interaction = (temp / 100.0) * (power / 700.0);

        vec![
            ewma_5m_temp,
            ewma_15m_temp,
            ewma_5m_ecc,
            ewma_15m_ecc,
            nvlink_rate,
            perf,
            power,
            interaction,
        ]
    }
}

fn calculate_ewma(values: &[f64], alpha: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ewma = values[0];
    for &v in &values[1..] {
        ewma = alpha * v + (1.0 - alpha) * ewma;
    }
    ewma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_vector_parity() {
        // Read golden artifact and test vectors
        let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/golden/failure_predictor.json");
        let vectors_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/golden/golden_vectors.json");

        let evaluator = XgbEvaluator::from_file(model_path).expect("Failed to load golden failure predictor model");
        let vectors_content = std::fs::read_to_string(vectors_path).expect("Failed to load golden test vectors");

        #[derive(Deserialize)]
        struct GoldenVector {
            description: String,
            features: Vec<f64>,
            raw_score: f64,
            expected_probability: f64,
        }

        let test_vectors: Vec<GoldenVector> = serde_json::from_str(&vectors_content).expect("Failed to parse golden vectors");

        for vector in test_vectors {
            let raw = evaluator.evaluate_raw(&vector.features);
            let prob = evaluator.predict_probability(&vector.features);

            assert!(
                (raw - vector.raw_score).abs() <= 1e-6,
                "Raw score mismatch for '{}': expected {}, got {}",
                vector.description,
                vector.raw_score,
                raw
            );

            assert!(
                (prob - vector.expected_probability).abs() <= 1e-6,
                "Probability parity failure for '{}': expected {}, got {}",
                vector.description,
                vector.expected_probability,
                prob
            );
        }
    }
}

