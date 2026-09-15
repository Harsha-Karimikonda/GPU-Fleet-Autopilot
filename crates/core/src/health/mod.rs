use crate::config::HealthConfig;
use crate::types::{Gpu, GpuStatus};

#[derive(Debug, Clone)]
pub struct HealthEngine {
    config: HealthConfig,
}

impl HealthEngine {
    pub fn new(config: HealthConfig) -> Self {
        Self { config }
    }

    /// Evaluates the composite health score (0.0 - 100.0) and updates the GPU status
    pub fn evaluate_gpu(&self, gpu: &mut Gpu) {
        let score = self.calculate_health_score(gpu);
        gpu.health_score = score;

        // Quarantined state is sticky — can only be cleared by explicit return_to_service
        if gpu.status == GpuStatus::Quarantined {
            return;
        }

        let new_status = self.score_to_status(score);
        gpu.status = new_status;
    }

    pub fn calculate_health_score(&self, gpu: &Gpu) -> f64 {
        let w = &self.config.weights;

        // 1. Temperature: baseline 70°C, ceiling 95°C
        let s_temp = ((gpu.temperature - 70.0).max(0.0) / 25.0).min(1.0);

        // 2. ECC Rate: 20 errors in window = fully degraded
        let s_ecc = (gpu.ecc_window_errors as f64 / 20.0).min(1.0);

        // 3. XID Errors: 3 XID events in window = fully degraded
        let s_xid = (gpu.xid_window_events as f64 / 3.0).min(1.0);

        // 4. Performance: 50% throughput drop = fully degraded
        let s_perf = ((1.0 - gpu.performance).max(0.0) / 0.50).min(1.0);

        // 5. NVLink Errors: 15 errors = fully degraded
        let s_nvlink = (gpu.nvlink_errors as f64 / 15.0).min(1.0);

        // 6. Network Errors: 10 errors = fully degraded
        let s_net = (gpu.network_errors as f64 / 10.0).min(1.0);

        // 7. Clock Throttle: binary flag
        let s_throttle = if gpu.clock_throttle != 0 || gpu.is_throttled {
            1.0
        } else {
            0.0
        };

        let weighted_penalty = w.temperature * s_temp
            + w.ecc_rate * s_ecc
            + w.xid * s_xid
            + w.performance * s_perf
            + w.nvlink * s_nvlink
            + w.network * s_net
            + w.clock_throttle * s_throttle;

        (100.0 * (1.0 - weighted_penalty)).max(0.0)
    }

    pub fn score_to_status(&self, score: f64) -> GpuStatus {
        let t = &self.config.thresholds;
        if score >= t.healthy {
            GpuStatus::Healthy
        } else if score >= t.degraded {
            GpuStatus::Degraded
        } else if score >= t.at_risk {
            GpuStatus::AtRisk
        } else {
            GpuStatus::Critical
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HealthConfig, HealthThresholds, HealthWeights};

    fn test_health_engine() -> HealthEngine {
        HealthEngine::new(HealthConfig {
            weights: HealthWeights {
                temperature: 0.20,
                ecc_rate: 0.20,
                xid: 0.20,
                performance: 0.15,
                nvlink: 0.10,
                network: 0.05,
                clock_throttle: 0.10,
            },
            thresholds: HealthThresholds {
                healthy: 80.0,
                degraded: 60.0,
                at_risk: 40.0,
            },
        })
    }

    #[test]
    fn test_pristine_gpu_is_healthy() {
        let engine = test_health_engine();
        let mut gpu = Gpu::new(
            "gpu-00001".to_string(),
            "H100".to_string(),
            "node-001".to_string(),
            "rack-01".to_string(),
            "us-east-1".to_string(),
        );
        gpu.temperature = 65.0;
        gpu.performance = 1.0;

        engine.evaluate_gpu(&mut gpu);
        assert_eq!(gpu.health_score, 100.0);
        assert_eq!(gpu.status, GpuStatus::Healthy);
    }

    #[test]
    fn test_thermal_and_ecc_degradation() {
        let engine = test_health_engine();
        let mut gpu = Gpu::new(
            "gpu-00002".to_string(),
            "H100".to_string(),
            "node-001".to_string(),
            "rack-01".to_string(),
            "us-east-1".to_string(),
        );
        // Temp 95°C (max penalty 0.20), ECC 20 (max penalty 0.20), throttled (0.10)
        gpu.temperature = 95.0;
        gpu.ecc_window_errors = 20;
        gpu.is_throttled = true;

        engine.evaluate_gpu(&mut gpu);
        // Penalty = 0.20 + 0.20 + 0.10 = 0.50 => Health = 50.0 => AtRisk
        assert!((gpu.health_score - 50.0).abs() < 1e-4);
        assert_eq!(gpu.status, GpuStatus::AtRisk);
    }

    #[test]
    fn test_quarantine_is_sticky() {
        let engine = test_health_engine();
        let mut gpu = Gpu::new(
            "gpu-00003".to_string(),
            "H100".to_string(),
            "node-001".to_string(),
            "rack-01".to_string(),
            "us-east-1".to_string(),
        );
        gpu.status = GpuStatus::Quarantined;
        gpu.temperature = 65.0;

        engine.evaluate_gpu(&mut gpu);
        assert_eq!(gpu.health_score, 100.0);
        assert_eq!(gpu.status, GpuStatus::Quarantined);
    }
}

