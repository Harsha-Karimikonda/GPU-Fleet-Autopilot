# GPU-Fleet-Autopilot

A high-performance **Rust + Python** infrastructure platform for simulating, monitoring, and autonomously maintaining large-scale GPU clusters (100 to 10,000 GPUs).

The core runtime is written in pure **Rust** — single-process, zero GC pauses, deterministic concurrency, pure-Rust ML tree-walker inference, native Prometheus metrics, and an embedded zero-dependency Web Console. Python is used **offline only** for model training, dataset generation, and Kaggle/Colab notebooks.

---

## Key Features

- **Physics-Based GPU Simulation Engine**: Newton's cooling law Euler ODE integrator ($\tau \frac{dT}{dt} = -(T - T_{\text{amb}}) + R_{\text{thermal}} P(t)$), dynamic workload power draw, sensor noise, and thermal throttling hysteresis ($92^\circ\text{C}$ throttle / $83^\circ\text{C}$ recovery).
- **Chaos Injection Studio**: 10 progressive failure scenarios:
  1. `GPU_ECC_FAILURE` (SBE accumulation $\to$ DBE fatal crash $\to$ XID 62)
  2. `GPU_THERMAL_FAILURE` (cooling degradation $\to$ thermal throttling)
  3. `GPU_XID_ERROR` (NVIDIA XID 31, 43, 62, 79, 92 injection)
  4. `GPU_MEMORY_DEGRADATION` (HBM bandwidth collapse)
  5. `NVLINK_DEGRADATION` (CRC error storms)
  6. `NETWORK_PACKET_LOSS` (RoCE/InfiniBand optical drops)
  7. `RDMA_FAILURE` (PFC pause storms)
  8. `STORAGE_LATENCY` (Checkpoint I/O hang)
  9. `DRIVER_CRASH` (Bus drop off / XID 79)
  10. `PERFORMANCE_REGRESSION` (Straggler -35% throughput regression)
- **Composite Health Scoring Engine**: Normalized multi-signal scoring ($0.0 - 100.0$) across temperature, ECC rates, XIDs, performance, NVLink, network drops, and clock throttling. Enforces strict lifecycle state machines (`Healthy`, `Degraded`, `AtRisk`, `Critical`, `Quarantined`).
- **AI Infrastructure Agent**: Deterministic, priority-ordered rule engine equipped with 10 diagnostic & remediation tools (`get_gpu_health`, `get_node_health`, `get_recent_events`, `get_gpu_history`, `compare_with_healthy_gpu`, `run_diagnostics`, `drain_node`, `quarantine_gpu`, `restart_driver`, `return_to_service`).
- **Autonomous Remediation & Scheduler**: Gang scheduler simulating distributed workloads, warm spare reserves (5%), automated eviction of degraded GPUs, and migration delays.
- **Pure-Rust XGBoost v1 Tree Walker**: Zero-dependency decision tree ensemble evaluator in pure Rust. Evaluates failure probabilities in microseconds without Python or C FFI runtimes. Golden parity tested ($\le 10^{-6}$).
- **Production Observability**: Canonical NVIDIA DCGM Prometheus metric export (`/metrics`) and embedded zero-dependency Web Console served at `http://localhost:8080`.
- **High-Throughput Telemetry Generator**: Multi-threaded synthetic dataset generator outputting millions of canonical telemetry rows in CSV or JSON Lines.

---

## Workspace Architecture

```
GPU-Fleet-Autopilot/
├── Cargo.toml                       # Workspace definition
├── crates/
│   ├── core/                        # Pure domain engine (physics, health, chaos, agent, scheduler, ML)
│   ├── server/                      # Tokio + Axum single-process server binary (`gpu-autopilot`)
│   ├── cli/                         # Clap CLI utility binary (`gpu-ctl`)
│   └── telemetry-gen/               # High-throughput batch dataset generator (`gpu-telemetry-gen`)
├── schema/
│   ├── telemetry_v1.json            # Canonical telemetry schema contract
│   └── model_v1.json                # Neutral XGBoost tree model export schema
├── testdata/golden/                 # Golden models & vectors for parity testing
├── ml/                              # Python offline training and model export
├── config.yaml                      # Cluster & simulation configuration
└── README.md
```

---

## Quickstart

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (1.75+ recommended):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### 1. Build the Workspace
```bash
cargo build --release
```

### 2. Run the Unit & Parity Tests
```bash
cargo test
```

### 3. Launch the Autopilot Server
```bash
cargo run --bin gpu-autopilot -- --config config.yaml
```
- Embedded Web Console: [http://localhost:8080](http://localhost:8080)
- Prometheus Metrics: [http://localhost:8080/metrics](http://localhost:8080/metrics)

### 4. Interact via CLI (`gpu-ctl`)
In another terminal:
```bash
# View fleet summary
cargo run --bin gpu-ctl -- status

# List degraded or at-risk GPUs
cargo run --bin gpu-ctl -- list --status DEGRADED

# Inspect detailed health card for a GPU
cargo run --bin gpu-ctl -- inspect gpu-00042

# Inject a chaos failure scenario
cargo run --bin gpu-ctl -- inject GPU_THERMAL_FAILURE gpu-00042

# Observe real-time AI agent diagnosis and failover
cargo run --bin gpu-ctl -- incidents

# Query pure-Rust XGBoost failure prediction
cargo run --bin gpu-ctl -- predict gpu-00042
```

### 5. Generate Synthetic Telemetry Dataset
```bash
cargo run --bin gpu-telemetry-gen -- --records 100000 --output data/telemetry.csv
```

---

## License
Apache-2.0
