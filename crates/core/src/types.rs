use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GpuStatus {
    Healthy,
    Degraded,
    AtRisk,
    Critical,
    Quarantined,
}

impl Default for GpuStatus {
    fn default() -> Self {
        Self::Healthy
    }
}

impl std::fmt::Display for GpuStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "HEALTHY"),
            Self::Degraded => write!(f, "DEGRADED"),
            Self::AtRisk => write!(f, "AT_RISK"),
            Self::Critical => write!(f, "CRITICAL"),
            Self::Quarantined => write!(f, "QUARANTINED"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySample {
    pub timestamp: i64,
    pub temperature: f64,
    pub power: f64,
    pub utilization: f64,
    pub memory_utilization: f64,
    pub sm_clock: u32,
    pub clock_throttle: u32,
    pub ecc_sbe_total: u64,
    pub ecc_dbe_total: u64,
    pub xid_errors: u32,
    pub nvlink_errors: u64,
    pub network_errors: u64,
    pub performance_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpu {
    pub id: String,
    pub model: String,
    pub node_id: String,
    pub rack_id: String,
    pub cluster_id: String,

    // Runtime telemetry signals
    pub temperature: f64,
    pub utilization: f64,
    pub power: f64,
    pub memory_utilization: f64,
    pub sm_clock: u32,
    pub clock_throttle: u32,
    pub performance: f64,
    pub baseline_perf: f64,

    // Internal physics state variables (ODE)
    #[serde(skip)]
    pub temp_phys: f64,
    #[serde(skip)]
    pub thermal_resistance: f64,
    #[serde(skip)]
    pub tau: f64,
    #[serde(skip)]
    pub ambient_temp: f64,
    #[serde(skip)]
    pub is_throttled: bool,

    // Error counters
    pub ecc_sbe_total: u64,
    pub ecc_dbe_total: u64,
    pub ecc_window_errors: u64,
    pub latest_xid: u32,
    pub xid_window_events: u32,
    pub nvlink_errors: u64,
    pub network_errors: u64,

    // History buffer for EWMA & feature engineering
    #[serde(skip)]
    pub history: VecDeque<TelemetrySample>,

    // Lifecycle state & scoring
    pub status: GpuStatus,
    pub health_score: f64,
    pub allocated_job_id: Option<String>,
    pub last_updated: DateTime<Utc>,

    // Active chaos injection if any
    pub active_chaos_scenario: Option<String>,
}

impl Gpu {
    pub fn new(
        id: String,
        model: String,
        node_id: String,
        rack_id: String,
        cluster_id: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            model,
            node_id,
            rack_id,
            cluster_id,
            temperature: 65.0,
            utilization: 10.0,
            power: 280.0,
            memory_utilization: 15.0,
            sm_clock: 1980,
            clock_throttle: 0,
            performance: 1.0,
            baseline_perf: 1.0,

            temp_phys: 65.0,
            thermal_resistance: 0.075,
            tau: 25.0,
            ambient_temp: 24.0,
            is_throttled: false,

            ecc_sbe_total: 0,
            ecc_dbe_total: 0,
            ecc_window_errors: 0,
            latest_xid: 0,
            xid_window_events: 0,
            nvlink_errors: 0,
            network_errors: 0,

            history: VecDeque::with_capacity(120),

            status: GpuStatus::Healthy,
            health_score: 100.0,
            allocated_job_id: None,
            last_updated: now,
            active_chaos_scenario: None,
        }
    }

    pub fn is_available_for_workload(&self) -> bool {
        self.status == GpuStatus::Healthy && self.allocated_job_id.is_none()
    }

    pub fn to_canonical_map(&self) -> std::collections::HashMap<String, serde_json::Value> {
        let mut map = std::collections::HashMap::new();
        map.insert("schema_version".to_string(), serde_json::json!("1.0"));
        map.insert("timestamp".to_string(), serde_json::json!(self.last_updated.timestamp_millis()));
        map.insert("gpu_id".to_string(), serde_json::json!(self.id));
        map.insert("node_id".to_string(), serde_json::json!(self.node_id));
        map.insert("rack_id".to_string(), serde_json::json!(self.rack_id));
        map.insert("cluster_id".to_string(), serde_json::json!(self.cluster_id));
        map.insert("job_id".to_string(), match &self.allocated_job_id {
            Some(job) => serde_json::json!(job),
            None => serde_json::Value::Null,
        });
        map.insert("dcgm_gpu_temp".to_string(), serde_json::json!(self.temperature));
        map.insert("dcgm_power_usage".to_string(), serde_json::json!(self.power));
        map.insert("dcgm_gpu_utilization".to_string(), serde_json::json!(self.utilization));
        map.insert("dcgm_mem_copy_utilization".to_string(), serde_json::json!(self.memory_utilization));
        map.insert("dcgm_sm_clock".to_string(), serde_json::json!(self.sm_clock));
        map.insert("dcgm_clock_throttle_reasons".to_string(), serde_json::json!(self.clock_throttle));
        map.insert("dcgm_ecc_sbe_volatile_total".to_string(), serde_json::json!(self.ecc_sbe_total));
        map.insert("dcgm_ecc_dbe_volatile_total".to_string(), serde_json::json!(self.ecc_dbe_total));
        map.insert("dcgm_xid_errors".to_string(), serde_json::json!(self.latest_xid));
        map.insert("dcgm_nvlink_error_count".to_string(), serde_json::json!(self.nvlink_errors));
        map.insert("network_errors_total".to_string(), serde_json::json!(self.network_errors));
        map.insert("performance_ratio".to_string(), serde_json::json!(self.performance));
        map
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobType {
    Training,
    Serving,
    Evaluation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStatus {
    Queued,
    Running,
    Paused,
    Migrating,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub job_type: JobType,
    pub priority: u32,
    pub gpu_count: usize,
    pub gpu_ids: Vec<String>,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub migration_started_at: Option<DateTime<Utc>>,
    pub checkpoint_age_s: f64,
    pub migrations_count: u32,
}

impl Job {
    pub fn new(id: String, job_type: JobType, priority: u32, gpu_count: usize) -> Self {
        Self {
            id,
            job_type,
            priority,
            gpu_count,
            gpu_ids: Vec::new(),
            status: JobStatus::Queued,
            created_at: Utc::now(),
            started_at: None,
            migration_started_at: None,
            checkpoint_age_s: 0.0,
            migrations_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IncidentSeverity {
    Warning,
    Degraded,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IncidentState {
    Detected,
    Investigating,
    Diagnosed,
    Remediating,
    Resolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Incident {
    pub id: String,
    pub gpu_id: String,
    pub node_id: String,
    pub severity: IncidentSeverity,
    pub state: IncidentState,
    pub detected_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub trigger_health: f64,
    pub diagnosis: Option<String>,
    pub confidence: Option<f64>,
    pub actions_taken: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub severity: String,
    pub gpu_id: String,
    pub node_id: String,
    pub details: serde_json::Value,
}

