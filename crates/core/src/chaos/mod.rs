use crate::types::{Gpu, GpuStatus};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ChaosScenario {
    GpuEccFailure,
    GpuThermalFailure,
    GpuXidError,
    GpuMemoryDegradation,
    NvlinkDegradation,
    NetworkPacketLoss,
    RdmaFailure,
    StorageLatency,
    DriverCrash,
    PerformanceRegression,
}

impl ChaosScenario {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GPU_ECC_FAILURE" => Some(Self::GpuEccFailure),
            "GPU_THERMAL_FAILURE" => Some(Self::GpuThermalFailure),
            "GPU_XID_ERROR" => Some(Self::GpuXidError),
            "GPU_MEMORY_DEGRADATION" => Some(Self::GpuMemoryDegradation),
            "NVLINK_DEGRADATION" => Some(Self::NvlinkDegradation),
            "NETWORK_PACKET_LOSS" => Some(Self::NetworkPacketLoss),
            "RDMA_FAILURE" => Some(Self::RdmaFailure),
            "STORAGE_LATENCY" => Some(Self::StorageLatency),
            "DRIVER_CRASH" => Some(Self::DriverCrash),
            "PERFORMANCE_REGRESSION" => Some(Self::PerformanceRegression),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GpuEccFailure => "GPU_ECC_FAILURE",
            Self::GpuThermalFailure => "GPU_THERMAL_FAILURE",
            Self::GpuXidError => "GPU_XID_ERROR",
            Self::GpuMemoryDegradation => "GPU_MEMORY_DEGRADATION",
            Self::NvlinkDegradation => "NVLINK_DEGRADATION",
            Self::NetworkPacketLoss => "NETWORK_PACKET_LOSS",
            Self::RdmaFailure => "RDMA_FAILURE",
            Self::StorageLatency => "STORAGE_LATENCY",
            Self::DriverCrash => "DRIVER_CRASH",
            Self::PerformanceRegression => "PERFORMANCE_REGRESSION",
        }
    }
}

pub struct ChaosEngine;

impl ChaosEngine {
    pub fn inject(gpu: &mut Gpu, scenario: ChaosScenario) {
        gpu.active_chaos_scenario = Some(scenario.as_str().to_string());
    }

    pub fn clear(gpu: &mut Gpu) {
        gpu.active_chaos_scenario = None;
        gpu.thermal_resistance = 0.075;
        gpu.is_throttled = false;
        gpu.performance = gpu.baseline_perf;
        gpu.latest_xid = 0;
    }

    pub fn apply_tick(gpu: &mut Gpu, _dt_s: f64) {
        let scenario_str = match &gpu.active_chaos_scenario {
            Some(s) => s.clone(),
            None => return,
        };

        let scenario = match ChaosScenario::parse(&scenario_str) {
            Some(s) => s,
            None => return,
        };

        let mut rng = rand::thread_rng();

        match scenario {
            ChaosScenario::GpuEccFailure => {
                // SBEs accumulate rapidly
                let new_sbes: u64 = rng.gen_range(3..8);
                gpu.ecc_sbe_total += new_sbes;
                gpu.ecc_window_errors += new_sbes;

                if gpu.ecc_sbe_total > 40 {
                    // Culminates in fatal uncorrectable DBE
                    gpu.ecc_dbe_total += 1;
                    gpu.latest_xid = 62; // ECC DBE
                    gpu.xid_window_events += 1;
                    gpu.performance = 0.0;
                }
            }
            ChaosScenario::GpuThermalFailure => {
                // Thermal paste dryout / fan failure -> R_thermal creeps up
                gpu.thermal_resistance = (gpu.thermal_resistance + 0.015).min(0.25);
                // When temperature exceeds 92°C, throttling engages
                if gpu.temperature >= 92.0 {
                    gpu.is_throttled = true;
                    gpu.clock_throttle = 1; // Thermal bitmask
                    gpu.sm_clock = 500;
                    gpu.performance = 0.35;
                }
            }
            ChaosScenario::GpuXidError => {
                let xid_options = [31, 43, 62, 79, 92];
                let chosen_xid = xid_options[rng.gen_range(0..xid_options.len())];
                gpu.latest_xid = chosen_xid;
                gpu.xid_window_events += 1;
                gpu.performance = 0.20;
            }
            ChaosScenario::GpuMemoryDegradation => {
                gpu.memory_utilization = (gpu.memory_utilization + 10.0).min(100.0);
                gpu.performance = (gpu.performance - 0.15).max(0.30);
            }
            ChaosScenario::NvlinkDegradation => {
                let errs: u64 = rng.gen_range(4..12);
                gpu.nvlink_errors += errs;
                gpu.performance = (gpu.performance - 0.10).max(0.40);
                if gpu.nvlink_errors > 50 {
                    gpu.latest_xid = 92; // NVLink error
                    gpu.xid_window_events += 1;
                }
            }
            ChaosScenario::NetworkPacketLoss => {
                let drops: u64 = rng.gen_range(2..6);
                gpu.network_errors += drops;
                gpu.performance = (gpu.performance - 0.08).max(0.50);
            }
            ChaosScenario::RdmaFailure => {
                gpu.network_errors += rng.gen_range(5..15);
                // Periodic pause stall
                if rng.gen_bool(0.5) {
                    gpu.utilization = 0.0;
                    gpu.performance = 0.10;
                }
            }
            ChaosScenario::StorageLatency => {
                // Checkpoint I/O hang
                gpu.utilization = 5.0;
                gpu.performance = (gpu.performance - 0.20).max(0.20);
            }
            ChaosScenario::DriverCrash => {
                gpu.sm_clock = 0;
                gpu.utilization = 0.0;
                gpu.performance = 0.0;
                gpu.latest_xid = 79; // GPU fallen off bus
                gpu.xid_window_events += 1;
            }
            ChaosScenario::PerformanceRegression => {
                // Straggler slowdown without obvious hardware crash
                gpu.performance = 0.65; // -35% regression
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inject_thermal_failure() {
        let mut gpu = Gpu::new(
            "gpu-00042".to_string(),
            "H100".to_string(),
            "node-006".to_string(),
            "rack-01".to_string(),
            "us-east-1".to_string(),
        );

        ChaosEngine::inject(&mut gpu, ChaosScenario::GpuThermalFailure);
        assert_eq!(gpu.active_chaos_scenario, Some("GPU_THERMAL_FAILURE".to_string()));

        // Apply multiple ticks
        for _ in 0..10 {
            ChaosEngine::apply_tick(&mut gpu, 2.0);
        }

        assert!(gpu.thermal_resistance > 0.10);
    }
}

