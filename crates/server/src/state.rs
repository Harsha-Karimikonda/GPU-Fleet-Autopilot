use gpu_fleet_autopilot_core::FleetEngine;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<RwLock<FleetEngine>>,
}

impl AppState {
    pub fn new(engine: FleetEngine) -> Self {
        Self {
            engine: Arc::new(RwLock::new(engine)),
        }
    }
}

