use crate::types::{Gpu, GpuStatus};

pub struct MetricsRenderer;

impl MetricsRenderer {
    /// Renders all fleet and DCGM telemetry metrics into Prometheus text format
    pub fn render_prometheus(fleet: &[Gpu], active_incidents: usize, remediations_count: usize) -> String {
        let mut out = String::with_capacity(64 * 1024);

        // Fleet summary gauges
        let total = fleet.len();
        let healthy = fleet.iter().filter(|g| g.status == GpuStatus::Healthy).count();
        let degraded = fleet.iter().filter(|g| g.status == GpuStatus::Degraded).count();
        let at_risk = fleet.iter().filter(|g| g.status == GpuStatus::AtRisk).count();
        let critical = fleet.iter().filter(|g| g.status == GpuStatus::Critical).count();
        let quarantined = fleet.iter().filter(|g| g.status == GpuStatus::Quarantined).count();

        let avg_util: f64 = if total > 0 {
            fleet.iter().map(|g| g.utilization).sum::<f64>() / total as f64
        } else {
            0.0
        };

        let avail_ratio: f64 = if total > 0 {
            (healthy + degraded) as f64 / total as f64
        } else {
            0.0
        };

        out.push_str("# HELP fleet_gpus_total Total count of managed GPUs\n# TYPE fleet_gpus_total gauge\n");
        out.push_str(&format!("fleet_gpus_total {}\n", total));

        out.push_str("# HELP fleet_gpus_healthy Count of healthy GPUs\n# TYPE fleet_gpus_healthy gauge\n");
        out.push_str(&format!("fleet_gpus_healthy {}\n", healthy));

        out.push_str("# HELP fleet_gpus_degraded Count of degraded GPUs\n# TYPE fleet_gpus_degraded gauge\n");
        out.push_str(&format!("fleet_gpus_degraded {}\n", degraded));

        out.push_str("# HELP fleet_gpus_at_risk Count of at-risk GPUs\n# TYPE fleet_gpus_at_risk gauge\n");
        out.push_str(&format!("fleet_gpus_at_risk {}\n", at_risk));

        out.push_str("# HELP fleet_gpus_critical Count of critical GPUs\n# TYPE fleet_gpus_critical gauge\n");
        out.push_str(&format!("fleet_gpus_critical {}\n", critical));

        out.push_str("# HELP fleet_gpus_quarantined Count of quarantined GPUs\n# TYPE fleet_gpus_quarantined gauge\n");
        out.push_str(&format!("fleet_gpus_quarantined {}\n", quarantined));

        out.push_str("# HELP fleet_utilization_ratio Average fleet compute utilization [0.0 - 100.0]\n# TYPE fleet_utilization_ratio gauge\n");
        out.push_str(&format!("fleet_utilization_ratio {:.2}\n", avg_util));

        out.push_str("# HELP fleet_availability_ratio Fleet availability ratio [0.0 - 1.0]\n# TYPE fleet_availability_ratio gauge\n");
        out.push_str(&format!("fleet_availability_ratio {:.4}\n", avail_ratio));

        out.push_str("# HELP autopilot_active_incidents Currently active autopilot incidents\n# TYPE autopilot_active_incidents gauge\n");
        out.push_str(&format!("autopilot_active_incidents {}\n", active_incidents));

        out.push_str("# HELP autopilot_remediations_total Cumulative count of autonomous remediations\n# TYPE autopilot_remediations_total counter\n");
        out.push_str(&format!("autopilot_remediations_total {}\n", remediations_count));

        // DCGM metric declarations
        out.push_str("\n# HELP dcgm_gpu_temp GPU temperature in Celsius\n# TYPE dcgm_gpu_temp gauge\n");
        for g in fleet {
            out.push_str(&format!(
                "dcgm_gpu_temp{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {:.2}\n",
                g.id, g.node_id, g.rack_id, g.model, g.temperature
            ));
        }

        out.push_str("\n# HELP dcgm_power_usage GPU power consumption in Watts\n# TYPE dcgm_power_usage gauge\n");
        for g in fleet {
            out.push_str(&format!(
                "dcgm_power_usage{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {:.2}\n",
                g.id, g.node_id, g.rack_id, g.model, g.power
            ));
        }

        out.push_str("\n# HELP dcgm_gpu_utilization GPU SM utilization percentage\n# TYPE dcgm_gpu_utilization gauge\n");
        for g in fleet {
            out.push_str(&format!(
                "dcgm_gpu_utilization{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {:.2}\n",
                g.id, g.node_id, g.rack_id, g.model, g.utilization
            ));
        }

        out.push_str("\n# HELP dcgm_sm_clock GPU SM clock frequency in MHz\n# TYPE dcgm_sm_clock gauge\n");
        for g in fleet {
            out.push_str(&format!(
                "dcgm_sm_clock{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {}\n",
                g.id, g.node_id, g.rack_id, g.model, g.sm_clock
            ));
        }

        out.push_str("\n# HELP dcgm_ecc_sbe_volatile_total Cumulative single-bit ECC errors\n# TYPE dcgm_ecc_sbe_volatile_total counter\n");
        for g in fleet {
            out.push_str(&format!(
                "dcgm_ecc_sbe_volatile_total{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {}\n",
                g.id, g.node_id, g.rack_id, g.model, g.ecc_sbe_total
            ));
        }

        out.push_str("\n# HELP dcgm_ecc_dbe_volatile_total Cumulative double-bit ECC errors\n# TYPE dcgm_ecc_dbe_volatile_total counter\n");
        for g in fleet {
            out.push_str(&format!(
                "dcgm_ecc_dbe_volatile_total{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {}\n",
                g.id, g.node_id, g.rack_id, g.model, g.ecc_dbe_total
            ));
        }

        out.push_str("\n# HELP gpu_health_score Normalized composite health score [0.0 - 100.0]\n# TYPE gpu_health_score gauge\n");
        for g in fleet {
            out.push_str(&format!(
                "gpu_health_score{{gpu_id=\"{}\",node_id=\"{}\",rack_id=\"{}\",model=\"{}\"}} {:.2}\n",
                g.id, g.node_id, g.rack_id, g.model, g.health_score
            ));
        }

        out
    }
}

