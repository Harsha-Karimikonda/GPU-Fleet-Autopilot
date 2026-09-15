use anyhow::Context;
use clap::Parser;
use gpu_fleet_autopilot_core::config::AppConfig;
use gpu_fleet_autopilot_core::schema::CanonicalRecord;
use gpu_fleet_autopilot_core::FleetEngine;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "gpu-telemetry-gen", about = "High-throughput synthetic GPU telemetry dataset generator")]
struct Args {
    #[arg(short, long, default_value = "config.yaml", help = "Config file path")]
    config: String,

    #[arg(short, long, default_value = "100000", help = "Number of telemetry records to generate")]
    records: usize,

    #[arg(short, long, default_value = "telemetry.jsonl", help = "Output file path (.jsonl or .csv)")]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Loading config from '{}'...", args.config);
    let config = AppConfig::load_from_file(&args.config)?;

    println!(
        "Initializing Fleet Engine ({} GPUs)... Generating {} records to '{}'...",
        config.fleet.gpu_count, args.records, args.output.display()
    );

    let mut engine = FleetEngine::new(config)?;
    let dt_s = 2.0;

    let file = File::create(&args.output)
        .with_context(|| format!("Failed to create output file: {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);

    let is_csv = args.output.extension().and_then(|e| e.to_str()) == Some("csv");

    if is_csv {
        writeln!(
            writer,
            "schema_version,timestamp,gpu_id,node_id,rack_id,cluster_id,job_id,dcgm_gpu_temp,dcgm_power_usage,dcgm_gpu_utilization,dcgm_mem_copy_utilization,dcgm_sm_clock,dcgm_clock_throttle_reasons,dcgm_ecc_sbe_volatile_total,dcgm_ecc_dbe_volatile_total,dcgm_xid_errors,dcgm_nvlink_error_count,network_errors_total,performance_ratio,failure_in_next_2h,failure_in_next_6h,failure_in_next_24h,failure_type"
        )?;
    }

    let mut generated = 0;
    while generated < args.records {
        engine.tick(dt_s);

        for gpu in engine.get_fleet() {
            if generated >= args.records {
                break;
            }

            let failure_type = gpu.active_chaos_scenario.clone().unwrap_or_else(|| "NONE".to_string());
            let is_failing = failure_type != "NONE";

            let record = CanonicalRecord {
                schema_version: "1.0".to_string(),
                timestamp: gpu.last_updated.timestamp_millis(),
                gpu_id: gpu.id.clone(),
                node_id: gpu.node_id.clone(),
                rack_id: gpu.rack_id.clone(),
                cluster_id: gpu.cluster_id.clone(),
                job_id: gpu.allocated_job_id.clone(),
                dcgm_gpu_temp: gpu.temperature,
                dcgm_power_usage: gpu.power,
                dcgm_gpu_utilization: gpu.utilization,
                dcgm_mem_copy_utilization: gpu.memory_utilization,
                dcgm_sm_clock: gpu.sm_clock,
                dcgm_clock_throttle_reasons: gpu.clock_throttle,
                dcgm_ecc_sbe_volatile_total: gpu.ecc_sbe_total,
                dcgm_ecc_dbe_volatile_total: gpu.ecc_dbe_total,
                dcgm_xid_errors: gpu.latest_xid,
                dcgm_nvlink_error_count: gpu.nvlink_errors,
                network_errors_total: gpu.network_errors,
                performance_ratio: gpu.performance,
                failure_in_next_2h: Some(if is_failing { 1 } else { 0 }),
                failure_in_next_6h: Some(if is_failing { 1 } else { 0 }),
                failure_in_next_24h: Some(if is_failing { 1 } else { 0 }),
                failure_type: Some(failure_type),
            };

            if is_csv {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{},{},{},{},{},{},{},{:.3},{},{},{},{}",
                    record.schema_version,
                    record.timestamp,
                    record.gpu_id,
                    record.node_id,
                    record.rack_id,
                    record.cluster_id,
                    record.job_id.as_deref().unwrap_or(""),
                    record.dcgm_gpu_temp,
                    record.dcgm_power_usage,
                    record.dcgm_gpu_utilization,
                    record.dcgm_mem_copy_utilization,
                    record.dcgm_sm_clock,
                    record.dcgm_clock_throttle_reasons,
                    record.dcgm_ecc_sbe_volatile_total,
                    record.dcgm_ecc_dbe_volatile_total,
                    record.dcgm_xid_errors,
                    record.dcgm_nvlink_error_count,
                    record.network_errors_total,
                    record.performance_ratio,
                    record.failure_in_next_2h.unwrap(),
                    record.failure_in_next_6h.unwrap(),
                    record.failure_in_next_24h.unwrap(),
                    record.failure_type.as_deref().unwrap(),
                )?;
            } else {
                let json_line = serde_json::to_string(&record)?;
                writeln!(writer, "{}", json_line)?;
            }

            generated += 1;
        }

        if generated % 20000 == 0 || generated == args.records {
            println!("Progress: {} / {} records generated", generated, args.records);
        }
    }

    writer.flush()?;
    println!("Generation finished successfully! Wrote {} records.", generated);
    Ok(())
}

