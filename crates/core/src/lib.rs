pub mod agent;
pub mod chaos;
pub mod config;
pub mod health;
pub mod ml;
pub mod regression;
pub mod remediation;
pub mod scheduler;
pub mod schema;
pub mod simulator;
pub mod telemetry;
pub mod types;

use anyhow::Context;
use chaos::ChaosScenario;
use config::AppConfig;
use health::HealthEngine;
use ml::XgbEvaluator;
use regression::RegressionDetector;
use remediation::RemediationController;
use scheduler::SchedulerEngine;
use simulator::SimulatorEngine;
use telemetry::MetricsRenderer;
use types::{Gpu, GpuStatus, Job, JobStatus, JobType};

pub struct FleetEngine {
    pub config: AppConfig,
    pub fleet: Vec<Gpu>,
    pub simulator: SimulatorEngine,
    pub health_engine: HealthEngine,
    pub agent: agent::AgentEngine,
    pub remediation: RemediationController,
    pub scheduler: SchedulerEngine,
    pub regression_detector: RegressionDetector,
    pub ml_evaluator: Option<XgbEvaluator>,
}

impl FleetEngine {
    pub fn new(config: AppConfig) -> anyhow::Result<Self> {
        let simulator = SimulatorEngine::new(config.fleet.clone());
        let health_engine = HealthEngine::new(config.health.clone());
        let agent = agent::AgentEngine::new(config.agent.clone());
        let remediation = RemediationController::new();
        let scheduler = SchedulerEngine::new(config.scheduler.clone(), config.fleet.warm_spare_ratio);
        let regression_detector = RegressionDetector::default();

        let fleet = simulator.create_fleet();

        let ml_evaluator = if config.ml.enabled && std::path::Path::new(&config.ml.model_path).exists() {
            match XgbEvaluator::from_file(&config.ml.model_path) {
                Ok(ev) => Some(ev),
                Err(e) => {
                    eprintln!("Warning: failed to load ML model from {}: {}", config.ml.model_path, e);
                    None
                }
            }
        } else {
            None
        };

        let mut engine = Self {
            config,
            fleet,
            simulator,
            health_engine,
            agent,
            remediation,
            scheduler,
            regression_detector,
            ml_evaluator,
        };

        // Seed initial workloads if requested
        if engine.config.scheduler.auto_generate_jobs {
            engine.seed_initial_jobs();
        }

        Ok(engine)
    }

    fn seed_initial_jobs(&mut self) {
        let num_jobs = 4.min(self.config.scheduler.max_concurrent_jobs);
        for _ in 0..num_jobs {
            self.scheduler.submit_job(
                JobType::Training,
                1,
                self.config.scheduler.default_gpus_per_job,
            );
        }
        self.scheduler.schedule_pending_jobs(&mut self.fleet);
    }

    /// Advances the entire fleet simulation by dt_s seconds
    pub fn tick(&mut self, dt_s: f64) {
        // 1. Physics ODE step (cooling, power, noise, chaos progression)
        self.simulator.step_fleet(&mut self.fleet, dt_s);

        // 2. Health engine scoring
        for gpu in self.fleet.iter_mut() {
            self.health_engine.evaluate_gpu(gpu);
        }

        // 3. Scheduler progression
        self.scheduler.step_scheduler(&mut self.fleet, dt_s);
        self.scheduler.schedule_pending_jobs(&mut self.fleet);

        // 4. Autonomous AI agent evaluation & incidents
        self.agent.step_autonomous_agent(&mut self.fleet);

        // 5. Remediation coordination for affected jobs
        let warm_spares = self.scheduler.get_warm_spares(&self.fleet);
        let jobs = self.scheduler.get_jobs_mut();

        // Check if any active job holds a quarantined or critical GPU
        let degraded_ids: Vec<String> = self
            .fleet
            .iter()
            .filter(|g| g.status == GpuStatus::Quarantined || g.status == GpuStatus::Critical)
            .map(|g| g.id.clone())
            .collect();

        for failed_id in degraded_ids {
            self.remediation.remediate_job_gpu(&failed_id, &mut self.fleet, jobs, &warm_spares);
        }
    }

    // --- Control API ---

    pub fn get_fleet(&self) -> &[Gpu] {
        &self.fleet
    }

    pub fn get_gpu(&self, id: &str) -> Option<&Gpu> {
        self.fleet.iter().find(|g| g.id == id)
    }

    pub fn inject_chaos(&mut self, gpu_id: &str, scenario_str: &str) -> anyhow::Result<()> {
        let scenario = ChaosScenario::parse(scenario_str)
            .with_context(|| format!("Invalid chaos scenario: '{}'", scenario_str))?;

        let gpu = self
            .fleet
            .iter_mut()
            .find(|g| g.id == gpu_id)
            .with_context(|| format!("GPU not found: '{}'", gpu_id))?;

        chaos::ChaosEngine::inject(gpu, scenario);
        Ok(())
    }

    pub fn clear_chaos(&mut self, gpu_id: &str) -> anyhow::Result<()> {
        let gpu = self
            .fleet
            .iter_mut()
            .find(|g| g.id == gpu_id)
            .with_context(|| format!("GPU not found: '{}'", gpu_id))?;

        chaos::ChaosEngine::clear(gpu);
        self.health_engine.evaluate_gpu(gpu);
        Ok(())
    }

    pub fn quarantine_gpu(&mut self, gpu_id: &str) -> anyhow::Result<()> {
        let gpu = self
            .fleet
            .iter_mut()
            .find(|g| g.id == gpu_id)
            .with_context(|| format!("GPU not found: '{}'", gpu_id))?;

        self.agent.tool_quarantine_gpu(gpu);
        Ok(())
    }

    pub fn return_to_service(&mut self, gpu_id: &str) -> anyhow::Result<bool> {
        let gpu = self
            .fleet
            .iter_mut()
            .find(|g| g.id == gpu_id)
            .with_context(|| format!("GPU not found: '{}'", gpu_id))?;

        let success = self.agent.tool_return_to_service(gpu);
        Ok(success)
    }

    pub fn drain_node(&mut self, node_id: &str) -> Vec<String> {
        self.agent.tool_drain_node(&mut self.fleet, node_id)
    }

    pub fn submit_job(&mut self, job_type: JobType, priority: u32, gpu_count: usize) -> String {
        let id = self.scheduler.submit_job(job_type, priority, gpu_count);
        self.scheduler.schedule_pending_jobs(&mut self.fleet);
        id
    }

    pub fn predict_failure(&self, gpu_id: &str) -> anyhow::Result<f64> {
        let evaluator = self.ml_evaluator.as_ref().context("ML model evaluator is not loaded")?;
        let gpu = self.get_gpu(gpu_id).context("GPU not found")?;
        let features = evaluator.extract_features(gpu);
        Ok(evaluator.predict_probability(&features))
    }

    pub fn render_metrics(&self) -> String {
        let active_incidents = self.agent.get_incidents().len();
        let remediations = self.remediation.get_records().len();
        MetricsRenderer::render_prometheus(&self.fleet, active_incidents, remediations)
    }
}

