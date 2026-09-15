use crate::types::Gpu;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub gpu_id: String,
    pub node_id: String,
    pub current_perf: f64,
    pub baseline_perf: f64,
    pub regression_percent: f64,
}

pub struct RegressionDetector {
    threshold_percent: f64, // e.g. 15.0 for 15% drop
}

impl RegressionDetector {
    pub fn new(threshold_percent: f64) -> Self {
        Self { threshold_percent }
    }

    pub fn detect_stragglers(&self, fleet: &[Gpu]) -> Vec<RegressionAlert> {
        let mut alerts = Vec::new();

        for gpu in fleet {
            let drop = (gpu.baseline_perf - gpu.performance) / gpu.baseline_perf.max(0.001);
            let drop_percent = drop * 100.0;

            if drop_percent >= self.threshold_percent {
                alerts.push(RegressionAlert {
                    gpu_id: gpu.id.clone(),
                    node_id: gpu.node_id.clone(),
                    current_perf: gpu.performance,
                    baseline_perf: gpu.baseline_perf,
                    regression_percent: drop_percent,
                });
            }
        }

        alerts
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(15.0)
    }
}

