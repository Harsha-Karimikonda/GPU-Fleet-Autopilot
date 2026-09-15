use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalRecord {
    pub schema_version: String,
    pub timestamp: i64,
    pub gpu_id: String,
    pub node_id: String,
    pub rack_id: String,
    pub cluster_id: String,
    pub job_id: Option<String>,
    pub dcgm_gpu_temp: f64,
    pub dcgm_power_usage: f64,
    pub dcgm_gpu_utilization: f64,
    pub dcgm_mem_copy_utilization: f64,
    pub dcgm_sm_clock: u32,
    pub dcgm_clock_throttle_reasons: u32,
    pub dcgm_ecc_sbe_volatile_total: u64,
    pub dcgm_ecc_dbe_volatile_total: u64,
    pub dcgm_xid_errors: u32,
    pub dcgm_nvlink_error_count: u64,
    pub network_errors_total: u64,
    pub performance_ratio: f64,

    // Dataset-only ground truth fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_in_next_2h: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_in_next_6h: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_in_next_24h: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_type: Option<String>,
}

