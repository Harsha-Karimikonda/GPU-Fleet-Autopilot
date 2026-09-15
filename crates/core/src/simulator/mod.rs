use crate::chaos::ChaosEngine;
use crate::config::FleetConfig;
use crate::types::{Gpu, TelemetrySample};
use chrono::Utc;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct SimulatorEngine {
    config: FleetConfig,
}

impl SimulatorEngine {
    pub fn new(config: FleetConfig) -> Self {
        Self { config }
    }

    /// Initializes a fleet of GPUs arranged into nodes and racks
    pub fn create_fleet(&self) -> Vec<Gpu> {
        let count = self.config.gpu_count;
        let per_node = self.config.gpus_per_node;
        let mut fleet = Vec::with_capacity(count);

        for i in 0..count {
            let gpu_idx = i + 1;
            let node_idx = (i / per_node) + 1;
            let rack_idx = ((node_idx - 1) / 16) + 1; // 16 nodes per rack

            let gpu_id = format!("gpu-{:05}", gpu_idx);
            let node_id = format!("node-{:03}", node_idx);
            let rack_id = format!("rack-{:02}", rack_idx);
            let cluster_id = "us-east-cluster-1".to_string();

            let mut gpu = Gpu::new(gpu_id, self.config.gpu_model.clone(), node_id, rack_id, cluster_id);

            // Add slight natural variance across GPUs
            let mut rng = rand::thread_rng();
            gpu.ambient_temp = 23.0 + rng.gen_range(-1.5..1.5);
            gpu.temp_phys = 64.0 + rng.gen_range(-2.0..2.0);
            gpu.temperature = gpu.temp_phys;

            fleet.push(gpu);
        }

        fleet
    }

    /// Executes one physics simulation tick across all GPUs
    pub fn step_fleet(&self, fleet: &mut [Gpu], dt_s: f64) {
        let now = Utc::now();
        let now_millis = now.timestamp_millis();

        for gpu in fleet.iter_mut() {
            // 1. Apply active chaos scenario degradation
            ChaosEngine::apply_tick(gpu, dt_s);

            // 2. Workload & Power dynamics
            let mut rng = rand::thread_rng();
            let norm_power = Normal::new(0.0, 8.0).unwrap();
            let power_noise: f64 = norm_power.sample(&mut rng);

            let is_busy = gpu.allocated_job_id.is_some();
            let base_util = if is_busy {
                if gpu.active_chaos_scenario.is_some() && gpu.performance < 0.5 {
                    20.0
                } else {
                    90.0 + rng.gen_range(-5.0..5.0)
                }
            } else {
                5.0 + rng.gen_range(0.0..3.0)
            };
            gpu.utilization = base_util.clamp(0.0, 100.0);

            let u_norm = gpu.utilization / 100.0;
            let p_idle = 150.0;
            let p_dyn = 550.0;
            gpu.power = (p_idle + p_dyn * u_norm + power_noise).clamp(100.0, 750.0);

            // 3. Newton's Cooling Law ODE Euler Step
            // tau * dT/dt = -(T - T_ambient) + R_thermal * P(t)
            let dt_over_tau = dt_s / gpu.tau.max(1.0);
            let delta_temp = dt_over_tau * (-(gpu.temp_phys - gpu.ambient_temp) + gpu.thermal_resistance * gpu.power);
            gpu.temp_phys = (gpu.temp_phys + delta_temp).clamp(15.0, 115.0);

            // Observed sensor reading with noise
            let norm_sensor = Normal::new(0.0, 1.2).unwrap();
            let sensor_noise: f64 = norm_sensor.sample(&mut rng);
            gpu.temperature = (gpu.temp_phys + sensor_noise).clamp(15.0, 115.0);

            // 4. Thermal Throttling Hysteresis
            if gpu.temp_phys >= 92.0 {
                gpu.is_throttled = true;
                gpu.clock_throttle |= 1; // Thermal bitmask
            } else if gpu.temp_phys <= 83.0 && gpu.active_chaos_scenario.as_deref() != Some("GPU_THERMAL_FAILURE") {
                gpu.is_throttled = false;
                gpu.clock_throttle &= !1;
            }

            if gpu.is_throttled {
                let throttled_clock = 1980.0 - 15.0 * (gpu.temp_phys - 83.0).max(0.0);
                gpu.sm_clock = (throttled_clock as u32).clamp(500, 1980);
                gpu.performance = (gpu.sm_clock as f64 / 1980.0).clamp(0.25, 1.0);
            } else if gpu.active_chaos_scenario.is_none() {
                gpu.sm_clock = 1980;
                gpu.performance = gpu.baseline_perf;
            }

            gpu.last_updated = now;

            // 5. Append sample to history buffer
            let sample = TelemetrySample {
                timestamp: now_millis,
                temperature: gpu.temperature,
                power: gpu.power,
                utilization: gpu.utilization,
                memory_utilization: gpu.memory_utilization,
                sm_clock: gpu.sm_clock,
                clock_throttle: gpu.clock_throttle,
                ecc_sbe_total: gpu.ecc_sbe_total,
                ecc_dbe_total: gpu.ecc_dbe_total,
                xid_errors: gpu.latest_xid,
                nvlink_errors: gpu.nvlink_errors,
                network_errors: gpu.network_errors,
                performance_ratio: gpu.performance,
            };

            if gpu.history.len() >= 60 {
                gpu.history.pop_front();
            }
            gpu.history.push_back(sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FleetConfig;

    fn test_fleet_config() -> FleetConfig {
        FleetConfig {
            gpu_count: 16,
            gpus_per_node: 8,
            gpu_model: "H100".to_string(),
            tick_interval_ms: 2000,
            sampling_interval_s: 30,
            warm_spare_ratio: 0.05,
        }
    }

    #[test]
    fn test_fleet_creation() {
        let sim = SimulatorEngine::new(test_fleet_config());
        let fleet = sim.create_fleet();
        assert_eq!(fleet.len(), 16);
        assert_eq!(fleet[0].id, "gpu-00001");
        assert_eq!(fleet[0].node_id, "node-001");
        assert_eq!(fleet[8].node_id, "node-002");
    }

    #[test]
    fn test_step_fleet_thermal_ode() {
        let sim = SimulatorEngine::new(test_fleet_config());
        let mut fleet = sim.create_fleet();

        // Step simulation 5 times
        for _ in 0..5 {
            sim.step_fleet(&mut fleet, 2.0);
        }

        assert_eq!(fleet[0].history.len(), 5);
        assert!(fleet[0].temperature > 20.0 && fleet[0].temperature < 100.0);
    }
}

