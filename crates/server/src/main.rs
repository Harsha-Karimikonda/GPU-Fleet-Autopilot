mod routes;
mod state;

use anyhow::Context;
use clap::Parser;
use gpu_fleet_autopilot_core::config::AppConfig;
use gpu_fleet_autopilot_core::FleetEngine;
use state::AppState;
use std::net::SocketAddr;
use std::time::Duration;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "gpu-autopilot", about = "GPU Fleet Autopilot Server (Rust Edition)")]
struct Args {
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(long)]
    bind: Option<String>,

    #[arg(short, long)]
    port: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gpu_fleet_autopilot_server=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    info!("Loading configuration from '{}'...", args.config);

    let mut config = AppConfig::load_from_file(&args.config)
        .with_context(|| format!("Failed to read configuration file: {}", args.config))?;

    if let Some(bind_override) = args.bind {
        config.server.bind = bind_override;
    }
    if let Some(port_override) = args.port {
        config.server.port = port_override;
    }

    let bind_addr = config.server.bind.clone();
    let port = config.server.port;
    let tick_ms = config.fleet.tick_interval_ms;
    let dt_s = tick_ms as f64 / 1000.0;

    info!(
        "Initializing GPU Fleet Autopilot Engine ({} GPUs, {} per node)...",
        config.fleet.gpu_count, config.fleet.gpus_per_node
    );

    let engine = FleetEngine::new(config)?;
    let app_state = AppState::new(engine);

    // Spawn background simulation tick loop
    let tick_state = app_state.clone();
    tokio::spawn(async move {
        info!("Starting physics tick loop (interval: {}ms)...", tick_ms);
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));
        loop {
            interval.tick().await;
            {
                let mut engine = tick_state.engine.write();
                engine.tick(dt_s);
            }
        }
    });

    let app = routes::build_router(app_state);

    let addr_str = format!("{}:{}", bind_addr, port);
    let socket_addr: SocketAddr = addr_str.parse().with_context(|| format!("Invalid socket address: {}", addr_str))?;

    info!("Serving GPU Fleet Mission Control at http://{}", socket_addr);
    info!("Prometheus metrics exported at http://{}/metrics", socket_addr);

    let listener = tokio::net::TcpListener::bind(socket_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

