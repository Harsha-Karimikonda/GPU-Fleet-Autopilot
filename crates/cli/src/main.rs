use anyhow::Context;
use clap::{Parser, Subcommand};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, Row, Table};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(name = "gpu-ctl", about = "GPU Fleet Autopilot Management CLI (Rust Edition)")]
struct Cli {
    #[arg(long, default_value = "http://127.0.0.1:8080", help = "Server base URL")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Display fleet summary and health tier breakdown")]
    Status,

    #[command(about = "List GPUs with optional status or node filters")]
    List {
        #[arg(short, long, help = "Filter by status: HEALTHY, DEGRADED, AT_RISK, CRITICAL, QUARANTINED")]
        status: Option<String>,

        #[arg(short, long, help = "Filter by node ID")]
        node: Option<String>,
    },

    #[command(about = "Inspect detailed health card and telemetry signals for a GPU")]
    Inspect {
        #[arg(help = "Target GPU ID (e.g. gpu-00042)")]
        gpu_id: String,
    },

    #[command(about = "Inject a chaos failure scenario into a GPU")]
    Inject {
        #[arg(help = "Failure scenario name (e.g. GPU_THERMAL_FAILURE, GPU_ECC_FAILURE)")]
        scenario: String,

        #[arg(help = "Target GPU ID")]
        gpu_id: String,
    },

    #[command(about = "Clear an active chaos failure scenario from a GPU")]
    Clear {
        #[arg(help = "Target GPU ID")]
        gpu_id: String,
    },

    #[command(about = "List active cluster jobs")]
    Jobs,

    #[command(about = "Submit a new workload job")]
    JobSubmit {
        #[arg(short, long, default_value = "8", help = "GPU count requested")]
        gpus: usize,

        #[arg(short, long, default_value = "TRAINING", help = "Job type: TRAINING, SERVING, EVALUATION")]
        job_type: String,

        #[arg(short, long, default_value = "1", help = "Priority")]
        priority: u32,
    },

    #[command(about = "Drain all GPUs on a specific node")]
    Drain {
        #[arg(help = "Target node ID (e.g. node-001)")]
        node_id: String,
    },

    #[command(about = "Quarantine a GPU, removing it from scheduling")]
    Quarantine {
        #[arg(help = "Target GPU ID")]
        gpu_id: String,
    },

    #[command(about = "Return a quarantined GPU to service after running diagnostics")]
    Return {
        #[arg(help = "Target GPU ID")]
        gpu_id: String,
    },

    #[command(about = "Query pure-Rust XGBoost failure prediction for a GPU")]
    Predict {
        #[arg(help = "Target GPU ID")]
        gpu_id: String,
    },

    #[command(about = "List active autopilot incidents and remediation traces")]
    Incidents,
}

#[derive(Deserialize, Debug)]
struct FleetStatusResponse {
    total: usize,
    healthy: usize,
    degraded: usize,
    at_risk: usize,
    critical: usize,
    quarantined: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();
    let base_url = cli.server.trim_end_matches('/');

    match cli.command {
        Commands::Status => {
            let res = client
                .get(format!("{}/api/v1/fleet/status", base_url))
                .send()
                .await
                .with_context(|| format!("Failed to connect to autopilot server at {}", base_url))?
                .json::<FleetStatusResponse>()
                .await?;

            let mut table = Table::new();
            table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
            table.set_header(vec![
                Cell::new("Total GPUs").fg(Color::Cyan),
                Cell::new("Healthy").fg(Color::Green),
                Cell::new("Degraded").fg(Color::Yellow),
                Cell::new("At Risk").fg(Color::Ansi256(208)), // Orange
                Cell::new("Critical").fg(Color::Red),
                Cell::new("Quarantined").fg(Color::Magenta),
            ]);

            table.add_row(Row::from(vec![
                Cell::new(res.total),
                Cell::new(res.healthy).fg(Color::Green),
                Cell::new(res.degraded).fg(Color::Yellow),
                Cell::new(res.at_risk).fg(Color::Ansi256(208)),
                Cell::new(res.critical).fg(Color::Red),
                Cell::new(res.quarantined).fg(Color::Magenta),
            ]));

            println!("{}", table);
        }

        Commands::List { status, node } => {
            let mut url = format!("{}/api/v1/gpus?", base_url);
            if let Some(ref s) = status {
                url.push_str(&format!("status={}&", s));
            }
            if let Some(ref n) = node {
                url.push_str(&format!("node_id={}&", n));
            }

            let gpus: Vec<serde_json::Value> = client.get(&url).send().await?.json().await?;

            let mut table = Table::new();
            table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
            table.set_header(vec!["GPU ID", "Node", "Rack", "Status", "Health", "Temp", "Power", "Perf"]);

            for g in gpus {
                let status_str = g["status"].as_str().unwrap_or("UNKNOWN");
                let status_cell = match status_str {
                    "HEALTHY" => Cell::new(status_str).fg(Color::Green),
                    "DEGRADED" => Cell::new(status_str).fg(Color::Yellow),
                    "AT_RISK" => Cell::new(status_str).fg(Color::Ansi256(208)),
                    "CRITICAL" => Cell::new(status_str).fg(Color::Red),
                    "QUARANTINED" => Cell::new(status_str).fg(Color::Magenta),
                    _ => Cell::new(status_str),
                };

                table.add_row(vec![
                    Cell::new(g["id"].as_str().unwrap_or("")),
                    Cell::new(g["node_id"].as_str().unwrap_or("")),
                    Cell::new(g["rack_id"].as_str().unwrap_or("")),
                    status_cell,
                    Cell::new(format!("{:.1}", g["health_score"].as_f64().unwrap_or(0.0))),
                    Cell::new(format!("{:.1}°C", g["temperature"].as_f64().unwrap_or(0.0))),
                    Cell::new(format!("{:.0}W", g["power"].as_f64().unwrap_or(0.0))),
                    Cell::new(format!("{:.0}%", g["performance"].as_f64().unwrap_or(0.0) * 100.0)),
                ]);
            }

            println!("{}", table);
        }

        Commands::Inspect { gpu_id } => {
            let res = client
                .get(format!("{}/api/v1/gpus/{}", base_url, gpu_id))
                .send()
                .await?;

            if !res.status().is_success() {
                eprintln!("Error inspecting GPU: {}", res.text().await?);
                return Ok(());
            }

            let g: serde_json::Value = res.json().await?;
            println!("\n=== GPU Health Card: {} ===", gpu_id);
            println!("Node:                 {}", g["node_id"].as_str().unwrap_or(""));
            println!("Rack:                 {}", g["rack_id"].as_str().unwrap_or(""));
            println!("Cluster:              {}", g["cluster_id"].as_str().unwrap_or(""));
            println!("Status:               {}", g["status"].as_str().unwrap_or(""));
            println!("Health Score:         {:.1} / 100.0", g["health_score"].as_f64().unwrap_or(0.0));
            println!("Active Job:           {}", g["allocated_job_id"].as_str().unwrap_or("None (idle)"));
            println!("Active Scenario:      {}", g["active_chaos_scenario"].as_str().unwrap_or("None"));
            println!("--- DCGM Telemetry ---");
            println!("Temperature:          {:.1} °C", g["temperature"].as_f64().unwrap_or(0.0));
            println!("Power Usage:          {:.1} W", g["power"].as_f64().unwrap_or(0.0));
            println!("SM Utilization:       {:.1} %", g["utilization"].as_f64().unwrap_or(0.0));
            println!("HBM Memory Util:      {:.1} %", g["memory_utilization"].as_f64().unwrap_or(0.0));
            println!("SM Clock:             {} MHz", g["sm_clock"].as_u64().unwrap_or(0));
            println!("Throttle Bitmask:     {}", g["clock_throttle"].as_u64().unwrap_or(0));
            println!("Performance Ratio:    {:.2}", g["performance"].as_f64().unwrap_or(0.0));
            println!("ECC SBE Total:        {}", g["ecc_sbe_total"].as_u64().unwrap_or(0));
            println!("ECC DBE Total:        {}", g["ecc_dbe_total"].as_u64().unwrap_or(0));
            println!("Latest XID:           {}", g["latest_xid"].as_u64().unwrap_or(0));
            println!("NVLink Errors:        {}", g["nvlink_errors"].as_u64().unwrap_or(0));
            println!("Network Errors:       {}", g["network_errors"].as_u64().unwrap_or(0));
            println!();
        }

        Commands::Inject { scenario, gpu_id } => {
            #[derive(Serialize)]
            struct InjectReq<'a> {
                gpu_id: &'a str,
                scenario: &'a str,
            }

            let res = client
                .post(format!("{}/api/v1/chaos/inject", base_url))
                .json(&InjectReq { gpu_id: &gpu_id, scenario: &scenario })
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Clear { gpu_id } => {
            #[derive(Serialize)]
            struct ClearReq<'a> {
                gpu_id: &'a str,
            }

            let res = client
                .post(format!("{}/api/v1/chaos/clear", base_url))
                .json(&ClearReq { gpu_id: &gpu_id })
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Jobs => {
            let jobs: Vec<serde_json::Value> = client
                .get(format!("{}/api/v1/jobs", base_url))
                .send()
                .await?
                .json()
                .await?;

            let mut table = Table::new();
            table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
            table.set_header(vec!["Job ID", "Type", "Status", "GPUs", "Migrations", "Allocated GPU IDs"]);

            for j in jobs {
                let status_str = j["status"].as_str().unwrap_or("");
                let gpu_ids_arr = j["gpu_ids"].as_array();
                let gpus_str = match gpu_ids_arr {
                    Some(arr) => arr.iter().map(|v| v.as_str().unwrap_or("")).collect::<Vec<_>>().join(", "),
                    None => "".to_string(),
                };

                table.add_row(vec![
                    Cell::new(j["id"].as_str().unwrap_or("")),
                    Cell::new(j["job_type"].as_str().unwrap_or("")),
                    Cell::new(status_str),
                    Cell::new(j["gpu_count"].as_u64().unwrap_or(0)),
                    Cell::new(j["migrations_count"].as_u64().unwrap_or(0)),
                    Cell::new(gpus_str),
                ]);
            }

            println!("{}", table);
        }

        Commands::JobSubmit { gpus, job_type, priority } => {
            #[derive(Serialize)]
            struct SubReq<'a> {
                gpu_count: usize,
                job_type: &'a str,
                priority: u32,
            }

            let res = client
                .post(format!("{}/api/v1/jobs", base_url))
                .json(&SubReq { gpu_count: gpus, job_type: &job_type, priority })
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Drain { node_id } => {
            let res = client
                .post(format!("{}/api/v1/nodes/{}/drain", base_url, node_id))
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Quarantine { gpu_id } => {
            let res = client
                .post(format!("{}/api/v1/gpus/{}/quarantine", base_url, gpu_id))
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Return { gpu_id } => {
            let res = client
                .post(format!("{}/api/v1/gpus/{}/return", base_url, gpu_id))
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Predict { gpu_id } => {
            let res = client
                .get(format!("{}/api/v1/predict/{}", base_url, gpu_id))
                .send()
                .await?;

            println!("{}", res.text().await?);
        }

        Commands::Incidents => {
            let incs: Vec<serde_json::Value> = client
                .get(format!("{}/api/v1/incidents", base_url))
                .send()
                .await?
                .json()
                .await?;

            let mut table = Table::new();
            table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
            table.set_header(vec!["Incident ID", "GPU ID", "State", "Severity", "Trigger Health", "Diagnosis", "Actions Taken"]);

            for inc in incs {
                let actions_arr = inc["actions_taken"].as_array();
                let actions_str = match actions_arr {
                    Some(arr) => arr.iter().map(|v| v.as_str().unwrap_or("")).collect::<Vec<_>>().join(", "),
                    None => "".to_string(),
                };

                table.add_row(vec![
                    Cell::new(inc["id"].as_str().unwrap_or("")),
                    Cell::new(inc["gpu_id"].as_str().unwrap_or("")),
                    Cell::new(inc["state"].as_str().unwrap_or("")),
                    Cell::new(inc["severity"].as_str().unwrap_or("")).fg(Color::Red),
                    Cell::new(format!("{:.1}", inc["trigger_health"].as_f64().unwrap_or(0.0))),
                    Cell::new(inc["diagnosis"].as_str().unwrap_or("Pending")),
                    Cell::new(actions_str),
                ]);
            }

            println!("{}", table);
        }
    }

    Ok(())
}

