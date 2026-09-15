use crate::config::AgentConfig;
use crate::types::{Gpu, GpuStatus, Incident, IncidentSeverity, IncidentState, TelemetryEvent};
use chrono::Utc;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    pub gpu_id: String,
    pub passed: bool,
    pub ecc_health: String,
    pub thermal_health: String,
    pub nvlink_health: String,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct RuleMatch {
    pub rule_name: String,
    pub diagnosis: String,
    pub confidence: f64,
    pub recommended_actions: Vec<String>,
}

pub struct AgentEngine {
    config: AgentConfig,
    active_incidents: HashMap<String, Incident>,
    cooldowns: HashMap<String, i64>,
    event_log: Vec<TelemetryEvent>,
    incident_counter: usize,
}

impl AgentEngine {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            active_incidents: HashMap::new(),
            cooldowns: HashMap::new(),
            event_log: Vec::new(),
            incident_counter: 0,
        }
    }

    pub fn get_incidents(&self) -> Vec<Incident> {
        self.active_incidents.values().cloned().collect()
    }

    pub fn get_event_log(&self) -> &[TelemetryEvent] {
        &self.event_log
    }

    // --- Diagnostic & Remediation Tools (1 to 10) ---

    /// Tool 1: get_gpu_health
    pub fn tool_get_gpu_health<'a>(&self, fleet: &'a [Gpu], gpu_id: &str) -> Option<&'a Gpu> {
        fleet.iter().find(|g| g.id == gpu_id)
    }

    /// Tool 2: get_node_health
    pub fn tool_get_node_health(&self, fleet: &[Gpu], node_id: &str) -> (f64, usize, usize) {
        let node_gpus: Vec<&Gpu> = fleet.iter().filter(|g| g.node_id == node_id).collect();
        if node_gpus.is_empty() {
            return (0.0, 0, 0);
        }
        let total_score: f64 = node_gpus.iter().map(|g| g.health_score).sum();
        let healthy_count = node_gpus.iter().filter(|g| g.status == GpuStatus::Healthy).count();
        (total_score / node_gpus.len() as f64, healthy_count, node_gpus.len())
    }

    /// Tool 3: get_recent_events
    pub fn tool_get_recent_events(&self, gpu_id: &str, limit: usize) -> Vec<TelemetryEvent> {
        self.event_log
            .iter()
            .rev()
            .filter(|e| e.gpu_id == gpu_id)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Tool 4: get_gpu_history
    pub fn tool_get_gpu_history<'a>(&self, fleet: &'a [Gpu], gpu_id: &str) -> Option<&'a std::collections::VecDeque<crate::types::TelemetrySample>> {
        fleet.iter().find(|g| g.id == gpu_id).map(|g| &g.history)
    }

    /// Tool 5: compare_with_healthy_gpu
    pub fn tool_compare_with_healthy_gpu<'a>(&self, fleet: &'a [Gpu], gpu_id: &str) -> Option<(&'a Gpu, &'a Gpu)> {
        let target = fleet.iter().find(|g| g.id == gpu_id)?;
        let healthy = fleet.iter().find(|g| g.status == GpuStatus::Healthy && g.id != gpu_id)?;
        Some((target, healthy))
    }

    /// Tool 6: run_diagnostics
    pub fn tool_run_diagnostics(&self, gpu: &Gpu) -> DiagnosticReport {
        let mut passed = true;
        let ecc_health = if gpu.ecc_dbe_total > 0 || gpu.ecc_window_errors > 15 {
            passed = false;
            "FAIL_DBE_DETECTED".to_string()
        } else {
            "PASS".to_string()
        };

        let thermal_health = if gpu.temperature >= 92.0 || gpu.thermal_resistance > 0.15 {
            passed = false;
            "FAIL_THERMAL_RESISTANCE_HIGH".to_string()
        } else {
            "PASS".to_string()
        };

        let nvlink_health = if gpu.nvlink_errors > 25 || gpu.latest_xid == 92 {
            passed = false;
            "FAIL_NVLINK_CRC_STORM".to_string()
        } else {
            "PASS".to_string()
        };

        DiagnosticReport {
            gpu_id: gpu.id.clone(),
            passed,
            ecc_health,
            thermal_health,
            nvlink_health,
            details: format!("Diagnostics complete. Overall: {}", if passed { "PASS" } else { "FAIL" }),
        }
    }

    /// Tool 7: drain_node
    pub fn tool_drain_node(&mut self, fleet: &mut [Gpu], node_id: &str) -> Vec<String> {
        let mut drained_gpus = Vec::new();
        for gpu in fleet.iter_mut().filter(|g| g.node_id == node_id) {
            gpu.status = GpuStatus::Quarantined;
            drained_gpus.push(gpu.id.clone());
        }
        drained_gpus
    }

    /// Tool 8: quarantine_gpu
    pub fn tool_quarantine_gpu(&mut self, gpu: &mut Gpu) {
        gpu.status = GpuStatus::Quarantined;
        self.emit_event(
            "GPU_QUARANTINED",
            "CRITICAL",
            &gpu.id,
            &gpu.node_id,
            serde_json::json!({
                "health_score": gpu.health_score,
                "reason": "Autonomous agent quarantine"
            }),
        );
    }

    /// Tool 9: restart_driver
    pub fn tool_restart_driver(&mut self, gpu: &mut Gpu) -> bool {
        self.emit_event(
            "DRIVER_RESTARTED",
            "WARNING",
            &gpu.id,
            &gpu.node_id,
            serde_json::json!({ "previous_xid": gpu.latest_xid }),
        );
        // Clear transient errors
        gpu.latest_xid = 0;
        gpu.clock_throttle = 0;
        gpu.sm_clock = 1980;
        gpu.performance = gpu.baseline_perf;
        true
    }

    /// Tool 10: return_to_service
    pub fn tool_return_to_service(&mut self, gpu: &mut Gpu) -> bool {
        let diag = self.tool_run_diagnostics(gpu);
        if diag.passed && gpu.health_score >= 80.0 {
            gpu.status = GpuStatus::Healthy;
            self.emit_event(
                "RETURN_TO_SERVICE",
                "INFO",
                &gpu.id,
                &gpu.node_id,
                serde_json::json!({ "health_score": gpu.health_score }),
            );
            true
        } else {
            false
        }
    }

    // --- Rule Engine Evaluation ---

    pub fn evaluate_rules(&self, gpu: &Gpu) -> Option<RuleMatch> {
        // Rule 1: Hardware degradation
        if gpu.ecc_window_errors > 10 && gpu.latest_xid != 0 && gpu.performance < 0.8 {
            return Some(RuleMatch {
                rule_name: "hardware_degradation".to_string(),
                diagnosis: "Probable uncorrectable memory or ASIC hardware fault".to_string(),
                confidence: 0.93,
                recommended_actions: vec!["quarantine_gpu".to_string(), "run_diagnostics".to_string(), "request_rma".to_string()],
            });
        }

        // Rule 2: Thermal runaway
        if gpu.temperature >= 92.0 && gpu.is_throttled {
            return Some(RuleMatch {
                rule_name: "thermal_runaway".to_string(),
                diagnosis: "Critical thermal dissipation failure or fan stall".to_string(),
                confidence: 0.95,
                recommended_actions: vec!["quarantine_gpu".to_string(), "run_diagnostics".to_string()],
            });
        }

        // Rule 3: Driver crash / bus drop
        if gpu.latest_xid == 79 || (gpu.sm_clock == 0 && gpu.utilization == 0.0) {
            return Some(RuleMatch {
                rule_name: "driver_crash_bus_drop".to_string(),
                diagnosis: "GPU fallen off PCIe/NVLink bus or kernel driver crashed".to_string(),
                confidence: 0.90,
                recommended_actions: vec!["restart_driver".to_string(), "quarantine_gpu".to_string()],
            });
        }

        // Rule 4: NVLink CRC storm
        if gpu.nvlink_errors > 25 || gpu.latest_xid == 92 {
            return Some(RuleMatch {
                rule_name: "nvlink_storm".to_string(),
                diagnosis: "NVLink high CRC replay rate or link retraining failure".to_string(),
                confidence: 0.88,
                recommended_actions: vec!["quarantine_gpu".to_string(), "run_diagnostics".to_string()],
            });
        }

        // Rule 5: Severe straggler / performance regression
        if gpu.performance < 0.70 && gpu.health_score < 70.0 {
            return Some(RuleMatch {
                rule_name: "performance_regression".to_string(),
                diagnosis: "Workload straggler bottlenecking distributed collective".to_string(),
                confidence: 0.85,
                recommended_actions: vec!["quarantine_gpu".to_string()],
            });
        }

        None
    }

    /// Autonomous loop: inspects fleet, handles critical threshold incidents, executes remediation
    pub fn step_autonomous_agent(&mut self, fleet: &mut [Gpu]) {
        let now = Utc::now();
        let now_epoch = now.timestamp();

        for i in 0..fleet.len() {
            let gpu = &fleet[i];
            let gpu_id = gpu.id.clone();
            let node_id = gpu.node_id.clone();
            let score = gpu.health_score;

            // Check if incident should be triggered (score < 40.0)
            if score < 40.0 && !self.active_incidents.contains_key(&gpu_id) && gpu.status != GpuStatus::Quarantined {
                let last_cooldown = self.cooldowns.get(&gpu_id).copied().unwrap_or(0);
                if now_epoch - last_cooldown >= self.config.cooldown_s {
                    self.incident_counter += 1;
                    let incident_id = format!("inc-{:06}", self.incident_counter);

                    let incident = Incident {
                        id: incident_id.clone(),
                        gpu_id: gpu_id.clone(),
                        node_id: node_id.clone(),
                        severity: IncidentSeverity::Critical,
                        state: IncidentState::Detected,
                        detected_at: now,
                        resolved_at: None,
                        trigger_health: score,
                        diagnosis: None,
                        confidence: None,
                        actions_taken: Vec::new(),
                    };

                    self.emit_event(
                        "INCIDENT_CREATED",
                        "CRITICAL",
                        &gpu_id,
                        &node_id,
                        serde_json::json!({
                            "incident_id": incident_id,
                            "trigger_health": score
                        }),
                    );

                    self.active_incidents.insert(gpu_id.clone(), incident);
                }
            }

            // If incident active for this GPU, evaluate rules and take actions
            if let Some(mut incident) = self.active_incidents.remove(&gpu_id) {
                incident.state = IncidentState::Investigating;

                let gpu_mut = &mut fleet[i];
                if let Some(rule_match) = self.evaluate_rules(gpu_mut) {
                    incident.state = IncidentState::Diagnosed;
                    incident.diagnosis = Some(rule_match.diagnosis.clone());
                    incident.confidence = Some(rule_match.confidence);

                    incident.state = IncidentState::Remediating;
                    for action in &rule_match.recommended_actions {
                        match action.as_str() {
                            "quarantine_gpu" => {
                                gpu_mut.status = GpuStatus::Quarantined;
                                incident.actions_taken.push("quarantine_gpu".to_string());
                            }
                            "restart_driver" => {
                                gpu_mut.latest_xid = 0;
                                incident.actions_taken.push("restart_driver".to_string());
                            }
                            "run_diagnostics" => {
                                incident.actions_taken.push("run_diagnostics".to_string());
                            }
                            "request_rma" => {
                                incident.actions_taken.push("request_rma_simulated".to_string());
                            }
                            _ => {}
                        }
                    }

                    incident.state = IncidentState::Resolved;
                    incident.resolved_at = Some(now);

                    self.emit_event(
                        "INCIDENT_RESOLVED",
                        "INFO",
                        &gpu_id,
                        &node_id,
                        serde_json::json!({
                            "incident_id": incident.id,
                            "actions": incident.actions_taken
                        }),
                    );

                    self.cooldowns.insert(gpu_id.clone(), now_epoch);
                } else {
                    // Re-insert if not yet diagnosed
                    self.active_incidents.insert(gpu_id, incident);
                }
            }
        }
    }

    fn emit_event(&mut self, event_type: &str, severity: &str, gpu_id: &str, node_id: &str, details: serde_json::Value) {
        let event = TelemetryEvent {
            event_id: format!("evt-{:08}", self.event_log.len() + 1),
            timestamp: Utc::now(),
            event_type: event_type.to_string(),
            severity: severity.to_string(),
            gpu_id: gpu_id.to_string(),
            node_id: node_id.to_string(),
            details,
        };
        self.event_log.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;

    fn test_agent_config() -> AgentConfig {
        AgentConfig {
            mode: "rules".to_string(),
            max_tool_calls: 10,
            evaluation_interval_s: 30,
            incident_trigger: "CRITICAL".to_string(),
            cooldown_s: 300,
        }
    }

    #[test]
    fn test_agent_evaluates_thermal_failure() {
        let mut agent = AgentEngine::new(test_agent_config());
        let mut gpu = Gpu::new(
            "gpu-00001".to_string(),
            "H100".to_string(),
            "node-001".to_string(),
            "rack-01".to_string(),
            "us-east-1".to_string(),
        );
        gpu.temperature = 94.0;
        gpu.is_throttled = true;

        let rule_match = agent.evaluate_rules(&gpu);
        assert!(rule_match.is_some());
        let r = rule_match.unwrap();
        assert_eq!(r.rule_name, "thermal_runaway");
        assert!(r.confidence >= 0.90);
    }
}

