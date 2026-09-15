use crate::types::{Gpu, GpuStatus, Job, JobStatus};
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct RemediationActionRecord {
    pub timestamp: chrono::DateTime<Utc>,
    pub gpu_id: String,
    pub action: String,
    pub job_id: Option<String>,
    pub spare_gpu_id: Option<String>,
    pub outcome: String,
}

pub struct RemediationController {
    records: Vec<RemediationActionRecord>,
}

impl RemediationController {
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    pub fn get_records(&self) -> &[RemediationActionRecord] {
        &self.records
    }

    /// Handles a degraded/quarantined GPU in an active job:
    /// Evicts GPU from job, transitions job to Migrating, and replaces with warm spare if available
    pub fn remediate_job_gpu(
        &mut self,
        failed_gpu_id: &str,
        fleet: &mut [Gpu],
        jobs: &mut [Job],
        warm_spares: &[String],
    ) -> Option<String> {
        let now = Utc::now();

        // Find the job using this GPU
        let mut affected_job_idx = None;
        for (idx, job) in jobs.iter().enumerate() {
            if job.status == JobStatus::Running && job.gpu_ids.contains(&failed_gpu_id.to_string()) {
                affected_job_idx = Some(idx);
                break;
            }
        }

        let job_idx = affected_job_idx?;
        let job = &mut jobs[job_idx];

        // 1. Evict failed GPU
        job.gpu_ids.retain(|id| id != failed_gpu_id);
        if let Some(failed_gpu) = fleet.iter_mut().find(|g| g.id == failed_gpu_id) {
            failed_gpu.allocated_job_id = None;
            failed_gpu.status = GpuStatus::Quarantined;
        }

        // 2. Allocate warm spare if available
        let spare_id = warm_spares.first().cloned();
        if let Some(ref sid) = spare_id {
            if let Some(spare_gpu) = fleet.iter_mut().find(|g| &g.id == sid) {
                spare_gpu.allocated_job_id = Some(job.id.clone());
                job.gpu_ids.push(sid.clone());
            }
            job.status = JobStatus::Migrating;
            job.migration_started_at = Some(now);
            job.migrations_count += 1;
        } else {
            job.status = JobStatus::Paused;
        }

        let record = RemediationActionRecord {
            timestamp: now,
            gpu_id: failed_gpu_id.to_string(),
            action: "EVICT_AND_FAILOVER".to_string(),
            job_id: Some(job.id.clone()),
            spare_gpu_id: spare_id.clone(),
            outcome: if spare_id.is_some() { "MIGRATION_INITIATED".to_string() } else { "PAUSED_NO_SPARES".to_string() },
        };
        self.records.push(record);

        spare_id
    }
}

impl Default for RemediationController {
    fn default() -> Self {
        Self::new()
    }
}

