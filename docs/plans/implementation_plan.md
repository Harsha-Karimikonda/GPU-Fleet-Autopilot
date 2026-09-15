# Implementation Plan: GPU Fleet Autopilot — Rust Edition

A **single-process Rust server + lightweight CLI client** that simulates a fleet of **100 to 10,000 GPUs**, generates realistic infrastructure telemetry, injects controlled failure scenarios, calculates real-time fleet health, detects performance regressions, runs a **Rust-native AI Infrastructure Agent** equipped with 10 diagnostic/remediation tools, executes closed-loop autonomous remediation, simulates a cluster workload scheduler, runs **Rust-native XGBoost v1 inference** from a pre-trained model, exports **native Prometheus metrics**, provides **pre-built Grafana dashboards**, serves an **embedded zero-dependency Web Console**, and maps 1:1 to an **8-part Kaggle/Colab notebook architecture**.

Python is used **offline only** — for model training, dataset generation, and Kaggle/Colab notebooks.

---

## Architectural Philosophy: Single-Process Server, Zero Bloat

This system runs entirely on a **single Mac with no real GPUs**. Every design decision reflects that constraint:

- **Single-Process Rust Server (`gpu-autopilot`)**: The entire runtime — physics simulation, AI agent, ML inference, metrics, web console — runs as one process on one port (`:8080`). No FastAPI sidecar, no Docker, no multi-service coordination.
- **Zero-Pause Determinism**: Tight ODE integration loops without garbage collection pauses.
- **Pure-Rust ML Inference**: XGBoost v1 models are trained in Python (offline) and exported to JSON. Rust loads the tree structure at startup and evaluates predictions in nanoseconds — no Python runtime needed.
- **No Heavy UI Dependencies**: An embedded, zero-dependency Web Console served directly by the Rust binary via `include_str!`.
- **Production Standard Metrics**: Native Prometheus `/metrics` endpoint using standard NVIDIA DCGM exporter metric conventions.
- **Canonical Telemetry Schema**: One versioned schema (`schema/telemetry_v1.json`) defines all field names, units, types, and counter semantics.
- **Instant Out-of-the-Box Execution**: `cargo build --release` builds both `gpu-autopilot` and `gpu-ctl`.

---

## Runtime vs. Offline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RUNTIME  (Single Rust Binary — :8080)                    │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ 1. Simulator  │  │ 2. Chaos      │  │ 3. Health     │  │ 4. ML Infer  │ │
│  │ 100–10K GPUs  │←→│ 10 Scenarios  │  │ Normalized    │  │ XGBoost Tree │ │
│  │ Tick loop     │  │ Degradation   │  │ Scoring       │  │ Walker (Rust)│ │
│  └───────┬───────┘  └───────────────┘  │ 100 → 0       │  └──────────────┘ │
│          │                             └───────┬───────┘                    │
│  ┌───────▼───────┐  ┌───────────────┐  ┌───────▼───────┐  ┌──────────────┐ │
│  │ 7. Scheduler  │  │ 8. Regression │  │ 5. AI Agent   │  │ 6. Remediate │ │
│  │ Job alloc,    │←→│ Baseline vs   │  │ Rust Rule     │←→│ Quarantine,  │ │
│  │ eviction,     │  │ current perf  │  │ Engine + 10   │  │ Migrate,     │ │
│  │ warm spares   │  │ straggler det │  │ Tools         │  │ RMA          │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 9. OBSERVABILITY & INTERFACES                                          │ │
│  │ Web Console (:8080) │ Prometheus /metrics │ Grafana JSON │ gpu-ctl CLI │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE  (Python — Not Running at Runtime)                │
│                                                                             │
│  crates/telemetry-gen / ml/generator.py → Parquet/CSV telemetry records     │
│  ml/model/train.py                      → Trains XGBoost v1                 │
│  ml/model/export.py                     → Exports model → JSON tree format  │
│                                                                             │
│  10. KAGGLE / COLAB NOTEBOOKS (NB1 to NB8)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workspace Layout

```
GPU-Fleet-Autopilot/
├── Cargo.toml                       # Workspace definition
├── crates/
│   ├── core/                        # Pure domain logic (zero network dependencies)
│   ├── server/                      # Server runtime binary (`gpu-autopilot`)
│   ├── cli/                         # CLI utility binary (`gpu-ctl`)
│   └── telemetry-gen/               # High-throughput batch dataset generator (`gpu-telemetry-gen`)
├── schema/
│   ├── telemetry_v1.json            # Canonical telemetry schema contract
│   └── model_v1.json                # Neutral XGBoost tree model export schema
├── testdata/golden/                 # Golden models & vectors for parity testing
├── ml/                              # Python offline training and model export
├── config.yaml                      # Cluster & simulation configuration
└── README.md
```
