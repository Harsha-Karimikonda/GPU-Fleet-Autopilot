use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub fleet: FleetConfig,
    pub health: HealthConfig,
    pub agent: AgentConfig,
    pub scheduler: SchedulerConfig,
    pub ml: MlConfig,
    pub telemetry: TelemetryConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetConfig {
    #[serde(default = "default_gpu_count")]
    pub gpu_count: usize,
    #[serde(default = "default_gpus_per_node")]
    pub gpus_per_node: usize,
    #[serde(default = "default_gpu_model")]
    pub gpu_model: String,
    #[serde(default = "default_tick_interval_ms")]
    pub tick_interval_ms: u64,
    #[serde(default = "default_sampling_interval_s")]
    pub sampling_interval_s: u64,
    #[serde(default = "default_warm_spare_ratio")]
    pub warm_spare_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthWeights {
    #[serde(default = "default_weight_02")]
    pub temperature: f64,
    #[serde(default = "default_weight_02")]
    pub ecc_rate: f64,
    #[serde(default = "default_weight_02")]
    pub xid: f64,
    #[serde(default = "default_weight_015")]
    pub performance: f64,
    #[serde(default = "default_weight_01")]
    pub nvlink: f64,
    #[serde(default = "default_weight_005")]
    pub network: f64,
    #[serde(default = "default_weight_01")]
    pub clock_throttle: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThresholds {
    #[serde(default = "default_threshold_healthy")]
    pub healthy: f64,
    #[serde(default = "default_threshold_degraded")]
    pub degraded: f64,
    #[serde(default = "default_threshold_at_risk")]
    pub at_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub weights: HealthWeights,
    pub thresholds: HealthThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_agent_mode")]
    pub mode: String,
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: usize,
    #[serde(default = "default_eval_interval")]
    pub evaluation_interval_s: u64,
    #[serde(default = "default_incident_trigger")]
    pub incident_trigger: String,
    #[serde(default = "default_cooldown_s")]
    pub cooldown_s: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_max_concurrent_jobs")]
    pub max_concurrent_jobs: usize,
    #[serde(default = "default_default_gpus_per_job")]
    pub default_gpus_per_job: usize,
    #[serde(default = "default_true")]
    pub auto_generate_jobs: bool,
    #[serde(default = "default_migration_time_s")]
    pub migration_time_s: f64,
    #[serde(default = "default_checkpoint_interval_s")]
    pub checkpoint_interval_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlConfig {
    pub model_path: String,
    pub features_manifest: String,
    #[serde(default = "default_prediction_interval_s")]
    pub prediction_interval_s: u64,
    #[serde(default = "default_warm_up_s")]
    pub warm_up_s: usize,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_horizons")]
    pub prediction_horizons: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_duration_hours")]
    pub dataset_duration_hours: u32,
    #[serde(default = "default_dataset_format")]
    pub dataset_format: String,
    #[serde(default = "default_partition_by")]
    pub partition_by: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_format")]
    pub log_format: String,
}

fn default_gpu_count() -> usize { 1000 }
fn default_gpus_per_node() -> usize { 8 }
fn default_gpu_model() -> String { "H100".to_string() }
fn default_tick_interval_ms() -> u64 { 2000 }
fn default_sampling_interval_s() -> u64 { 30 }
fn default_warm_spare_ratio() -> f64 { 0.05 }
fn default_weight_02() -> f64 { 0.20 }
fn default_weight_015() -> f64 { 0.15 }
fn default_weight_01() -> f64 { 0.10 }
fn default_weight_005() -> f64 { 0.05 }
fn default_threshold_healthy() -> f64 { 80.0 }
fn default_threshold_degraded() -> f64 { 60.0 }
fn default_threshold_at_risk() -> f64 { 40.0 }
fn default_agent_mode() -> String { "rules".to_string() }
fn default_max_tool_calls() -> usize { 10 }
fn default_eval_interval() -> u64 { 30 }
fn default_incident_trigger() -> String { "CRITICAL".to_string() }
fn default_cooldown_s() -> i64 { 300 }
fn default_max_concurrent_jobs() -> usize { 50 }
fn default_default_gpus_per_job() -> usize { 8 }
fn default_true() -> bool { true }
fn default_migration_time_s() -> f64 { 120.0 }
fn default_checkpoint_interval_s() -> f64 { 600.0 }
fn default_prediction_interval_s() -> u64 { 60 }
fn default_warm_up_s() -> usize { 900 }
fn default_horizons() -> Vec<u32> { vec![2, 6, 24] }
fn default_duration_hours() -> u32 { 168 }
fn default_dataset_format() -> String { "parquet".to_string() }
fn default_partition_by() -> String { "day".to_string() }
fn default_schema_version() -> String { "1.0".to_string() }
fn default_bind() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8080 }
fn default_log_format() -> String { "json".to_string() }

impl AppConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let weight_sum = self.health.weights.temperature
            + self.health.weights.ecc_rate
            + self.health.weights.xid
            + self.health.weights.performance
            + self.health.weights.nvlink
            + self.health.weights.network
            + self.health.weights.clock_throttle;

        if (weight_sum - 1.0).abs() > 1e-4 {
            anyhow::bail!("Health weights must sum to 1.0, got: {:.4}", weight_sum);
        }

        if self.fleet.gpu_count == 0 {
            anyhow::bail!("gpu_count must be greater than 0");
        }
        if self.fleet.gpus_per_node == 0 {
            anyhow::bail!("gpus_per_node must be greater than 0");
        }
        Ok(())
    }
}

