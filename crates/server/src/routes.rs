use crate::state::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use gpu_fleet_autopilot_core::types::{GpuStatus, JobType};
use serde::{Deserialize, Serialize};

pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Embedded Web Console
        .route("/", get(serve_console))
        // Prometheus Metrics
        .route("/metrics", get(serve_metrics))
        // REST API v1
        .route("/api/v1/fleet/status", get(get_fleet_status))
        .route("/api/v1/gpus", get(list_gpus))
        .route("/api/v1/gpus/:id", get(get_gpu_by_id))
        .route("/api/v1/chaos/inject", post(inject_chaos))
        .route("/api/v1/chaos/clear", post(clear_chaos))
        .route("/api/v1/gpus/:id/quarantine", post(quarantine_gpu))
        .route("/api/v1/gpus/:id/return", post(return_to_service))
        .route("/api/v1/nodes/:id/drain", post(drain_node))
        .route("/api/v1/jobs", get(list_jobs).post(submit_job))
        .route("/api/v1/incidents", get(list_incidents))
        .route("/api/v1/predict/:id", get(predict_gpu_failure))
        .with_state(state)
}

// Embedded Web Console HTML
static CONSOLE_HTML: &str = include_str!("console/index.html");

async fn serve_console() -> Html<&'static str> {
    Html(CONSOLE_HTML)
}

async fn serve_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let engine = state.engine.read();
    let metrics_text = engine.render_metrics();
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        metrics_text,
    )
}

#[derive(Serialize)]
struct FleetSummaryResponse {
    total: usize,
    healthy: usize,
    degraded: usize,
    at_risk: usize,
    critical: usize,
    quarantined: usize,
}

async fn get_fleet_status(State(state): State<AppState>) -> Json<FleetSummaryResponse> {
    let engine = state.engine.read();
    let fleet = engine.get_fleet();

    let summary = FleetSummaryResponse {
        total: fleet.len(),
        healthy: fleet.iter().filter(|g| g.status == GpuStatus::Healthy).count(),
        degraded: fleet.iter().filter(|g| g.status == GpuStatus::Degraded).count(),
        at_risk: fleet.iter().filter(|g| g.status == GpuStatus::AtRisk).count(),
        critical: fleet.iter().filter(|g| g.status == GpuStatus::Critical).count(),
        quarantined: fleet.iter().filter(|g| g.status == GpuStatus::Quarantined).count(),
    };

    Json(summary)
}

#[derive(Deserialize)]
struct ListGpusQuery {
    status: Option<String>,
    node_id: Option<String>,
}

async fn list_gpus(
    State(state): State<AppState>,
    Query(query): Query<ListGpusQuery>,
) -> Json<Vec<serde_json::Value>> {
    let engine = state.engine.read();
    let fleet = engine.get_fleet();

    let mut gpus_ref: Vec<&gpu_fleet_autopilot_core::types::Gpu> = fleet
        .iter()
        .filter(|g| {
            if let Some(ref s) = query.status {
                if !g.status.to_string().eq_ignore_ascii_case(s) {
                    return false;
                }
            }
            if let Some(ref n) = query.node_id {
                if !g.node_id.eq_ignore_ascii_case(n) {
                    return false;
                }
            }
            true
        })
        .collect();

    // Prioritize non-healthy or active chaos GPUs to appear first
    gpus_ref.sort_by_key(|g| {
        let is_healthy = g.status == GpuStatus::Healthy && g.active_chaos_scenario.is_none();
        (is_healthy, g.id.clone())
    });

    if query.status.is_none() && query.node_id.is_none() && gpus_ref.len() > 100 {
        gpus_ref.truncate(100);
    }

    let filtered: Vec<serde_json::Value> = gpus_ref.iter().map(|g| serde_json::to_value(g).unwrap()).collect();
    Json(filtered)
}

async fn get_gpu_by_id(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let engine = state.engine.read();
    match engine.get_gpu(&id) {
        Some(gpu) => Ok(Json(serde_json::to_value(gpu).unwrap())),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("GPU '{}' not found", id) })),
        )),
    }
}

#[derive(Deserialize)]
struct ChaosRequest {
    gpu_id: String,
    scenario: String,
}

async fn inject_chaos(
    State(state): State<AppState>,
    Json(payload): Json<ChaosRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut engine = state.engine.write();
    match engine.inject_chaos(&payload.gpu_id, &payload.scenario) {
        Ok(_) => Ok(Json(serde_json::json!({
            "status": "success",
            "message": format!("Injected scenario '{}' into GPU '{}'", payload.scenario, payload.gpu_id)
        }))),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

#[derive(Deserialize)]
struct ClearChaosRequest {
    gpu_id: String,
}

async fn clear_chaos(
    State(state): State<AppState>,
    Json(payload): Json<ClearChaosRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut engine = state.engine.write();
    match engine.clear_chaos(&payload.gpu_id) {
        Ok(_) => Ok(Json(serde_json::json!({
            "status": "success",
            "message": format!("Cleared chaos for GPU '{}'", payload.gpu_id)
        }))),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

async fn quarantine_gpu(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut engine = state.engine.write();
    match engine.quarantine_gpu(&id) {
        Ok(_) => Ok(Json(serde_json::json!({
            "status": "success",
            "message": format!("GPU '{}' quarantined", id)
        }))),
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

async fn return_to_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mut engine = state.engine.write();
    match engine.return_to_service(&id) {
        Ok(success) => {
            if success {
                Ok(Json(serde_json::json!({
                    "status": "success",
                    "message": format!("GPU '{}' returned to HEALTHY service", id)
                })))
            } else {
                Err((
                    StatusCode::PRECONDITION_FAILED,
                    Json(serde_json::json!({
                        "error": format!("GPU '{}' did not pass diagnostic checks or health score < 80", id)
                    })),
                ))
            }
        }
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

async fn drain_node(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    let mut engine = state.engine.write();
    let drained = engine.drain_node(&id);
    Json(serde_json::json!({
        "status": "success",
        "node_id": id,
        "drained_gpus": drained
    }))
}

async fn list_jobs(State(state): State<AppState>) -> Json<serde_json::Value> {
    let engine = state.engine.read();
    let jobs = engine.scheduler.get_jobs();
    Json(serde_json::to_value(jobs).unwrap())
}

#[derive(Deserialize)]
struct SubmitJobRequest {
    job_type: Option<String>,
    priority: Option<u32>,
    gpu_count: usize,
}

async fn submit_job(
    State(state): State<AppState>,
    Json(payload): Json<SubmitJobRequest>,
) -> Json<serde_json::Value> {
    let mut engine = state.engine.write();
    let job_type = match payload.job_type.as_deref() {
        Some("SERVING") => JobType::Serving,
        Some("EVALUATION") => JobType::Evaluation,
        _ => JobType::Training,
    };

    let job_id = engine.submit_job(job_type, payload.priority.unwrap_or(1), payload.gpu_count);
    Json(serde_json::json!({
        "status": "success",
        "job_id": job_id
    }))
}

async fn list_incidents(State(state): State<AppState>) -> Json<serde_json::Value> {
    let engine = state.engine.read();
    let incidents = engine.agent.get_incidents();
    Json(serde_json::to_value(incidents).unwrap())
}

async fn predict_gpu_failure(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let engine = state.engine.read();
    match engine.predict_failure(&id) {
        Ok(prob) => Ok(Json(serde_json::json!({
            "gpu_id": id,
            "failure_probability": prob,
            "horizon": "2h-24h",
            "is_high_risk": prob >= 0.70
        }))),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )),
    }
}

