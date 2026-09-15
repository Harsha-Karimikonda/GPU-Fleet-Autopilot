use crate::config::SchedulerConfig;
use crate::types::{Gpu, GpuStatus, Job, JobStatus, JobType};
use chrono::Utc;

pub struct SchedulerEngine {
    config: SchedulerConfig,
    warm_spare_ratio: f64,
    jobs: Vec<Job>,
    job_counter: usize,
}

impl SchedulerEngine {
    pub fn new(config: SchedulerConfig, warm_spare_ratio: f64) -> Self {
        Self {
            config,
            warm_spare_ratio,
            jobs: Vec::new(),
            job_counter: 0,
        }
    }

    pub fn get_jobs(&self) -> &[Job] {
        &self.jobs
    }

    pub fn get_jobs_mut(&mut self) -> &mut [Job] {
        &mut self.jobs
    }

    pub fn submit_job(&mut self, job_type: JobType, priority: u32, gpu_count: usize) -> String {
        self.job_counter += 1;
        let job_id = format!("job-{:05}", self.job_counter);
        let job = Job::new(job_id.clone(), job_type, priority, gpu_count);
        self.jobs.push(job);
        job_id
    }

    /// Identifies warm spare GPUs reserved for emergency failover
    pub fn get_warm_spares(&self, fleet: &[Gpu]) -> Vec<String> {
        let total = fleet.len();
        let spare_count = ((total as f64) * self.warm_spare_ratio).ceil() as usize;

        fleet
            .iter()
            .rev()
            .filter(|g| g.status == GpuStatus::Healthy && g.allocated_job_id.is_none())
            .take(spare_count)
            .map(|g| g.id.clone())
            .collect()
    }

    /// Allocates available healthy GPUs to queued jobs, respecting rack/node locality
    pub fn schedule_pending_jobs(&mut self, fleet: &mut [Gpu]) {
        let warm_spares = self.get_warm_spares(fleet);
        let now = Utc::now();

        for job in self.jobs.iter_mut() {
            if job.status != JobStatus::Queued {
                continue;
            }

            // Find available healthy GPUs not in warm spare pool
            let available_gpus: Vec<String> = fleet
                .iter()
                .filter(|g| g.status == GpuStatus::Healthy && g.allocated_job_id.is_none() && !warm_spares.contains(&g.id))
                .take(job.gpu_count)
                .map(|g| g.id.clone())
                .collect();

            if available_gpus.len() == job.gpu_count {
                for gid in &available_gpus {
                    if let Some(gpu) = fleet.iter_mut().find(|g| &g.id == gid) {
                        gpu.allocated_job_id = Some(job.id.clone());
                    }
                }
                job.gpu_ids = available_gpus;
                job.status = JobStatus::Running;
                job.started_at = Some(now);
            }
        }
    }

    /// Advances scheduler lifecycle: checks migrating jobs and completes elapsed migrations
    pub fn step_scheduler(&mut self, _fleet: &mut [Gpu], dt_s: f64) {
        let now = Utc::now();

        for job in self.jobs.iter_mut() {
            match job.status {
                JobStatus::Running => {
                    // Accumulate checkpoint age
                    job.checkpoint_age_s += dt_s;
                    if job.checkpoint_age_s >= self.config.checkpoint_interval_s {
                        job.checkpoint_age_s = 0.0; // Checkpoint saved
                    }
                }
                JobStatus::Migrating => {
                    if let Some(started) = job.migration_started_at {
                        let elapsed_s = (now - started).num_milliseconds() as f64 / 1000.0;
                        if elapsed_s >= self.config.migration_time_s {
                            job.status = JobStatus::Running;
                            job.migration_started_at = None;
                            job.checkpoint_age_s = 0.0; // Resumes from last checkpoint
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SchedulerConfig;

    fn test_scheduler_config() -> SchedulerConfig {
        SchedulerConfig {
            max_concurrent_jobs: 10,
            default_gpus_per_job: 4,
            auto_generate_jobs: true,
            migration_time_s: 10.0,
            checkpoint_interval_s: 600.0,
        }
    }

    #[test]
    fn test_job_allocation() {
        let mut scheduler = SchedulerEngine::new(test_scheduler_config(), 0.10);
        let mut fleet = vec![
            Gpu::new("gpu-00001".to_string(), "H100".to_string(), "n1".to_string(), "r1".to_string(), "c1".to_string()),
            Gpu::new("gpu-00002".to_string(), "H100".to_string(), "n1".to_string(), "r1".to_string(), "c1".to_string()),
            Gpu::new("gpu-00003".to_string(), "H100".to_string(), "n1".to_string(), "r1".to_string(), "c1".to_string()),
            Gpu::new("gpu-00004".to_string(), "H100".to_string(), "n1".to_string(), "r1".to_string(), "c1".to_string()),
            Gpu::new("gpu-00005".to_string(), "H100".to_string(), "n1".to_string(), "r1".to_string(), "c1".to_string()),
        ];

        let job_id = scheduler.submit_job(JobType::Training, 1, 4);
        scheduler.schedule_pending_jobs(&mut fleet);

        let job = scheduler.get_jobs().iter().find(|j| j.id == job_id).unwrap();
        assert_eq!(job.status, JobStatus::Running);
        assert_eq!(job.gpu_ids.len(), 4);
    }
}

