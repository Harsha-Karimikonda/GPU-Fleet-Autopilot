# Implementation Plan: GPU Fleet Autopilot — Simulation Edition

A **single-process Go server + lightweight CLI client** that simulates a fleet of **100 to 10,000 GPUs**, generates realistic infrastructure telemetry, injects controlled failure scenarios, calculates real-time fleet health, detects performance regressions, runs a **Go-native AI Infrastructure Agent** equipped with 10 diagnostic/remediation tools, executes closed-loop autonomous remediation, simulates a cluster workload scheduler, runs **Go-native XGBoost v1 inference** from a pre-trained model, exports **native Prometheus metrics**, provides **pre-built Grafana dashboards**, serves an **embedded zero-dependency Web Console**, and maps 1:1 to an **8-part Kaggle/Colab notebook architecture**.

Python is used **offline only** — for model training, dataset generation, and Kaggle/Colab notebooks.

---

## Architectural Philosophy: Single-Process Server, Zero Bloat

This system runs entirely on a **single Mac with no real GPUs**. Every design decision reflects that constraint:

- **Single-Process Go Server**: The entire runtime — simulation, AI agent, ML inference, metrics, web console — runs as one process on one port (`:8080`). No FastAPI sidecar, no Docker, no multi-service coordination.
- **No Heavy UI Dependencies**: An embedded, zero-dependency Web Console served directly by the Go binary. No Streamlit, Node.js, or complex UI build tools required.
- **Production Standard Metrics**: Native Prometheus `/metrics` endpoint using standard NVIDIA DCGM exporter metric conventions.
- **Canonical Telemetry Schema**: One versioned schema defines all field names, units, types, and counter semantics. Go structs, CSV exports, Prometheus metrics, and ML features all derive from the same source of truth.
- **Turnkey Grafana Integration**: Clean, pre-provisioned Grafana dashboard JSON ready to import for teams using Grafana.
- **Structured Event Logs**: Clean, structured JSON event logs for incidents, root causes, and remediation actions — no OTel collector pipelines.
- **Go-Native ML Inference**: XGBoost v1 models are trained in Python (offline) and exported to JSON. Go loads the tree structure at startup and evaluates predictions in microseconds — no Python runtime needed.
- **Instant Out-of-the-Box Execution**: `go build && ./gpu-autopilot` for the server, `go build && ./gpu-ctl` for the CLI.

---

## Runtime vs. Offline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RUNTIME  (Single Go Binary — :8080)                     │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ 1. Simulator  │  │ 2. Chaos      │  │ 3. Health     │  │ 4. ML Infer  │ │
│  │ 100–10K GPUs  │←→│ 10 Scenarios  │  │ Normalized    │  │ XGBoost Tree │ │
│  │ Tick loop     │  │ Degradation   │  │ Scoring       │  │ Walker (Go)  │ │
│  └───────┬───────┘  └───────────────┘  │ 100 → 0       │  └──────────────┘ │
│          │                             └───────┬───────┘                    │
│  ┌───────▼───────┐  ┌───────────────┐  ┌───────▼───────┐  ┌──────────────┐ │
│  │ 7. Scheduler  │  │ 8. Regression │  │ 5. AI Agent   │  │ 6. Remediate │ │
│  │ Job alloc,    │←→│ Baseline vs   │  │ Go Rule       │←→│ Quarantine,  │ │
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
│  ml/dataset/generator.py       → 28.8M–200M Parquet telemetry records      │
│  ml/model/train.py             → Trains XGBoost v1                         │
│  ml/model/export.py            → Exports model → JSON tree format          │
│                                                                             │
│  10. KAGGLE / COLAB NOTEBOOKS                                               │
│  NB1: Simulator  →  NB2: Failure Inj  →  NB3: Features  →  NB4: XGBoost   │
│                                                           ↘  NB5: Anomaly  │
│  NB1 → NB6: RCA Agent  →  NB7: Autonomous Remediation                     │
│                            NB4 + NB7  →  NB8: Fleet Evaluation & SLA       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration System

All tunable parameters are externalized in a single YAML file. The Go binary reads this at startup; environment variable overrides are supported for CI/scripting.

```yaml
fleet:
  gpu_count: 10000
  gpus_per_node: 8
  gpu_model: "H100"
  tick_interval_ms: 2000          # Physics timestep (internal ODE)
  sampling_interval_s: 30         # Telemetry sampling (persisted to dataset)
  warm_spare_ratio: 0.05

health:
  weights:                         # Sum to 1.0; inputs normalized 0.0–1.0
    temperature: 0.20
    ecc_rate: 0.20
    xid: 0.20
    performance: 0.15
    nvlink: 0.10
    network: 0.05
    clock_throttle: 0.10
  thresholds:
    healthy: 80
    degraded: 60
    at_risk: 40

ml:
  model_path: "pkg/ml/models/failure_predictor.json"
  features_manifest: "pkg/ml/models/features.json"
  prediction_interval_s: 60
  warm_up_s: 900                   # 15 min — suppress predictions until feature history fills
  enabled: true
  prediction_horizons: [2, 6, 24]  # Hours

agent:
  mode: "rules"
  max_tool_calls: 10
  evaluation_interval_s: 30
  incident_trigger: "CRITICAL"     # Health score below this tier triggers incident
  cooldown_s: 300                  # Min time before reopening an incident on same GPU

scheduler:
  max_concurrent_jobs: 50
  default_gpus_per_job: 8
  auto_generate_jobs: true
  migration_time_s: 120            # Simulated migration delay for job rescheduling
  checkpoint_interval_s: 600       # Jobs checkpoint every 10 min

telemetry:
  dataset_duration_hours: 168      # 7 days
  dataset_format: "parquet"        # parquet (default) or csv
  partition_by: "day"
  schema_version: "1.0"

server:
  bind: "127.0.0.1"               # Localhost only by default
  port: 8080
  log_format: "json"
```

---

## Canonical Telemetry Schema

One versioned schema defines all field names, units, types, and counter semantics. The Go struct uses internal names but maps to canonical names via adapters. All external interfaces (CSV, Parquet, Prometheus, REST API, ML features) use these canonical names.

### Canonical Telemetry Schema v1.0

| Canonical Field Name | Type | Description |
|----------------------|------|-------------|
| `schema_version` | string | "1.0" |
| `timestamp` | int64 | epoch milliseconds |
| `gpu_id` | string | "gpu-00001" .. "gpu-10000" |
| `node_id` | string | "node-001" .. "node-1250" |
| `rack_id` | string | "rack-01" .. "rack-XX" |
| `cluster_id` | string | "us-east-cluster-1" |
| `job_id` | string | allocated job ID, empty if idle |
| `dcgm_gpu_temp` | float64 | °C, GPU temperature. Gauge. |
| `dcgm_power_usage` | float64 | Watts, GPU power draw. Gauge. |
| `dcgm_gpu_utilization` | float64 | %, SM utilization 0-100. Gauge. |
| `dcgm_mem_copy_utilization` | float64 | %, HBM utilization 0-100. Gauge. |
| `dcgm_sm_clock` | int | MHz, SM clock speed. Gauge. |
| `dcgm_clock_throttle_reasons` | int | bitmask: 0=None, 1=Thermal, 2=Power, 4=Board. Gauge. |
| `dcgm_ecc_sbe_volatile_total` | int | Cumulative SBE count. Monotonic counter — never decrements. |
| `dcgm_ecc_dbe_volatile_total` | int | Cumulative DBE count. Monotonic counter. |
| `dcgm_xid_errors` | int | Last active XID code (0, 31, 43, 62, 79, 92). Point-in-time. |
| `dcgm_nvlink_error_count` | int | Cumulative NVLink CRC/replay errors. Monotonic counter. |
| `network_errors_total` | int | Cumulative IB/RoCE packet drops. Monotonic counter. |
| `performance_ratio` | float64 | Actual throughput / baseline. 1.0 = 100%. Gauge. |

**Dataset-only fields (not in runtime Prometheus, only in persisted training data):**
- `failure_in_next_2h` (int, 0 or 1): Ground-truth label for 2h prediction horizon.
- `failure_in_next_6h` (int, 0 or 1): Ground-truth label for 6h prediction horizon.
- `failure_in_next_24h` (int, 0 or 1): Ground-truth label for 24h prediction horizon.
- `failure_type` (string): `NONE`, `THERMAL`, `ECC_MEMORY`, `NVLINK`, `XID_CRASH`, `STRAGGLER`, `NETWORK`, `RDMA`, `STORAGE`, `DRIVER`, `MEMORY_DEGRADATION`

**Adapters:**
- **Go struct → canonical**: The simulator's internal representation provides a method to serialize its state into a map of canonical fields.
- **Canonical → Dataset**: Exporters format the canonical map into Parquet/CSV columns.
- **Canonical → Prometheus**: Prometheus collectors read the canonical values and expose them as metrics. Dataset-only labels are explicitly excluded from Prometheus.

---

## Detailed Specifications

### 1. GPU Fleet Simulator (Go Core)

- **Virtual GPU Model**:
  ```go
  type GPU struct {
      ID                string    `json:"id"`                 // e.g. "gpu-00042"
      Model             string    `json:"model"`              // "H100" (or A100, B200)
      NodeID            string    `json:"node_id"`            // "node-006"
      RackID            string    `json:"rack_id"`            // "rack-01"
      ClusterID         string    `json:"cluster_id"`         // "us-east-cluster-1"

      // Internal Go fields map to Canonical Telemetry Schema via adapters
      Temperature       float64   `json:"temperature"`        // °C (Normal: 65-72°C)
      Utilization       float64   `json:"utilization"`        // % (Normal: 0-100%)
      Power             float64   `json:"power"`              // Watts (Normal: 250-700W)
      MemoryUtilization float64   `json:"memory_utilization"` // % HBM used
      SMClock           int       `json:"sm_clock"`           // MHz (Normal: 1980MHz, Throttled: 500MHz)
      ClockThrottle     int       `json:"clock_throttle"`     // Bitmask throttle reasons
      Performance       float64   `json:"performance"`        // Relative perf (1.0 = 100%)
      BaselinePerf      float64   `json:"baseline_perf"`      // Benchmark baseline (1.0 = 100%)

      // Error counters (windowed — reset each scoring interval for rate calculation)
      ECCErrorsTotal    int       `json:"ecc_errors_total"`   // Cumulative lifetime SBE+DBE
      ECCErrorsWindow   int       `json:"ecc_errors_window"`  // Errors in current scoring window
      XIDErrors         int       `json:"xid_errors"`         // XID events in current window
      LatestXID         int       `json:"latest_xid"`         // e.g. 31, 43, 62, 79, 92
      NVLinkErrors      int       `json:"nvlink_errors"`      // Cumulative NVLink CRC/replay errors
      NetworkErrors     int       `json:"network_errors"`     // Cumulative IB/RoCE drops

      // Ring buffer of historical samples
      HistoryBuffer     []Sample  `json:"-"`                  // Last 60 samples (30min at 30s interval)

      // State
      Status            GPUStatus `json:"status"`             // HEALTHY, DEGRADED, AT_RISK, CRITICAL, QUARANTINED
      HealthScore       float64   `json:"health_score"`       // 0.0 – 100.0
      AllocatedJobID    string    `json:"allocated_job_id"`   // Active job ID (empty if idle)
      LastUpdated       time.Time `json:"last_updated"`
  }

  type GPUStatus string
  const (
      StatusHealthy     GPUStatus = "HEALTHY"
      StatusDegraded    GPUStatus = "DEGRADED"
      StatusAtRisk      GPUStatus = "AT_RISK"
      StatusCritical    GPUStatus = "CRITICAL"
      StatusQuarantined GPUStatus = "QUARANTINED"
  )
  ```

- **Scale**:
  - Configurable from **100 to 10,000 GPUs** (e.g. 1,250 nodes × 8 GPUs) via `config.yaml`.
  - High-throughput tick loop updating temperature dissipation, power fluctuations, and telemetry every `tick_interval_ms`.

- **Concurrency Model**:
  - **Single goroutine** iterates over all GPUs sequentially per tick. 10K structs is small — a sequential pass takes <1ms.
  - API reads use `sync.RWMutex`: the tick loop holds a write lock; HTTP handlers hold read locks.
  - This is simple and avoids race conditions. If benchmarks show bottlenecks, shard by rack with a worker pool later.

- **Bulk Telemetry Generation**:
  - Generates realistic datasets simulating cluster runtime for Kaggle/Colab ML workflows using canonical field names.
  - Sizing: 10,000 GPUs × 24h × 1 sample/30s = 28.8M rows/day. 7-day dataset = ~200M rows.
  - Format: Parquet, partitioned by day.
  - 1M rows for quick iteration (small fleet / short window); full production dataset is 28.8M–200M rows in partitioned Parquet.

---

### 2. Failure Injection Engine

Controlled scenarios with progressive, realistic degradation curves:

| # | Scenario | Degradation Curve |
|---|---|---|
| 1 | `GPU_ECC_FAILURE` | SBEs accumulate → row retirements → DBE fatal crash |
| 2 | `GPU_THERMAL_FAILURE` | Fan failure / thermal paste dryout → Temp rises to 92°C+ → SM clock drops to 500MHz |
| 3 | `GPU_XID_ERROR` | Injects specific XID events (31: page fault, 62: ECC DBE, 79: fallen off bus, 92: link error) |
| 4 | `GPU_MEMORY_DEGRADATION` | Memory bandwidth loss & leaks → compute throughput drops |
| 5 | `NVLINK_DEGRADATION` | Replay and CRC error storms → NVLink bandwidth collapses |
| 6 | `NETWORK_PACKET_LOSS` | RoCE/IB optical degradation → packet drops & TCP retries |
| 7 | `RDMA_FAILURE` | PFC pause storms → RDMA timeout hangs |
| 8 | `STORAGE_LATENCY` | Checkpoint sync stalls → GPU hangs in I/O wait |
| 9 | `DRIVER_CRASH` | Kernel deadlock or fatal crash → GPU drops off bus |
| 10 | `PERFORMANCE_REGRESSION` | Subtle straggler → clock jitter causes -20% to -50% slowdown |

---

### 3. Fleet Health Engine

**Normalized composite scoring** with signals scaled to 0.0–1.0 range and configurable weights summing to 1.0:

$$\text{Health} = \max\!\Bigl(0,\; 100 \times \bigl(1 - \sum_{i} w_i \cdot \hat{s}_i\bigr)\Bigr)$$

| Signal | Normalization ($\hat{s}$) | Default Weight ($w$) |
|---|---|---|
| Temperature | $\min(1,\; \frac{\max(0,\; T - 70)}{25})$ — baseline 70°C, ceiling 95°C | 0.20 |
| ECC Rate | $\min(1,\; \frac{\text{errors\_in\_window}}{20})$ — 20 errors/window = fully degraded | 0.20 |
| XID Errors | $\min(1,\; \frac{\text{xid\_count}}{3})$ — 3 XID events = fully degraded | 0.20 |
| Performance | $\min(1,\; \frac{\max(0,\; 1.0 - \text{perf\_ratio})}{0.5})$ — 50% drop = fully degraded | 0.15 |
| NVLink Errors | $\min(1,\; \frac{\text{nvlink\_errors}}{15})$ | 0.10 |
| Network Errors | $\min(1,\; \frac{\text{net\_errors}}{10})$ | 0.05 |
| Clock Throttle | $1.0$ if throttled, $0.0$ if not | 0.10 |

**Scoring Tiers** (configurable via `config.yaml`):
- `100 ──── 80`: **HEALTHY**
- ` 79 ──── 60`: **DEGRADED**
- ` 59 ──── 40`: **AT RISK**
- ` 39 ────  0`: **CRITICAL**

**GPU Lifecycle State Machine** — valid transitions are enforced:

```
    ┌──────────────────────────────────────────────┐
    │               Self-recovery                   │
    │    HEALTHY ←── DEGRADED ←── AT_RISK           │
    │       │            │            │              │
    │       ▼            ▼            ▼              │
    │    DEGRADED    AT_RISK      CRITICAL           │
    │                                 │              │
    │                                 ▼              │
    │                            QUARANTINED         │
    │                                 │              │
    │         return_to_service()     │              │
    │    HEALTHY ◄────────────────────┘              │
    └──────────────────────────────────────────────┘
```

Status is derived from health score on each tick, with one exception: `QUARANTINED` is only set by the AI agent and only cleared by explicit `return_to_service()`. A quarantined GPU's health score is still computed but does not drive status transitions.

---

### 4. ML Failure Predictor (Go-Native Inference, Python-Trained)

**Training (Python — offline)**:
- **Features**: Rolling 5m/15m/1h EWMA, degradation slopes ($\frac{d\text{temp}}{dt}$, $\frac{d\text{ecc}}{dt}$), interaction features ($\text{temp} \times \text{power}$).
- **Model**: XGBoost v1 binary classifier.
- **Training script**: `python ml/model/train.py --input data/telemetry.parquet --output pkg/ml/models/failure_predictor.json`

**Leakage Controls**:
- Excluded from ML features: `dcgm_xid_errors` (direct terminal indicator), `status`, `health_score`, or any field that is a direct consequence of the failure event.
- Allowed: `ecc_sbe_rate`, `temperature`, `nvlink_error_rate`, `performance_ratio` — but only values from before the terminal event.
- Diagnostic accuracy report: show model performance with and without terminal indicators.

**Prediction Horizon Labels**:
- `failure_in_next_2h`, `failure_in_next_6h`, `failure_in_next_24h`

**Model Export Pipeline**:
1. Train in Python using the `xgboost` library.
2. Export to XGBoost's native JSON dump format (neutral export format).
3. Alongside the model, export a `features.json` manifest that defines feature names, order, and computation method.
4. The exported `failure_predictor.json` and `features.json` are checked into the repo under `pkg/ml/models/`.

**Neutral Export Format**:
```json
{
  "schema_version": "1.0",
  "model_type": "xgboost",
  "base_score": 0.5,
  "objective": "binary:logistic",
  "num_trees": 100,
  "features": ["ewma_5m_temp", "ewma_5m_ecc_rate", "..."],
  "trees": [
    {
      "nodes": [
        {"id": 0, "split_feature": 2, "threshold": 0.75, "yes": 1, "no": 2, "missing": 1},
        {"id": 1, "leaf_value": 0.42},
        {"id": 2, "leaf_value": -0.31}
      ]
    }
  ]
}
```

**Inference (Go — runtime)**:
- `pkg/ml/evaluator.go` loads the JSON tree structure at startup, handling `base_score` and missing-value directions.
- Predicts probability of failure via pure Go tree walking.
- Runs every `prediction_interval_s` (default: 60s) across non-quarantined GPUs.
- **Warm-up**: Predictions are suppressed for the first `warm_up_s` (default: 900s) after startup to let the `HistoryBuffer` fill.
- **Golden-vector parity test**: 100 GPU feature vectors with expected predictions. Both Python and Go must match within ≤ 1e-6.

---

### 5. AI Infrastructure Agent (Go-Native Rule Engine)

A **deterministic, priority-ordered rule engine** implemented entirely in Go. No LLM API keys required. The agent evaluates GPU telemetry against a rulebook and produces structured diagnoses.

**10 Diagnostic & Remediation Tools** (all Go functions operating on in-process state):

| # | Tool | Type | Description |
|---|---|---|---|
| 1 | `get_gpu_health(gpu_id)` | Read | Returns GPU struct with health score, all signals |
| 2 | `get_node_health(node_id)` | Read | Aggregates health of all GPUs on a node |
| 3 | `get_recent_events(gpu_id)` | Read | Returns last N structured events for this GPU |
| 4 | `get_gpu_history(gpu_id, window)` | Read | Returns signal time-series over window |
| 5 | `compare_with_healthy_gpu(gpu_id, baseline_gpu_id)` | Read | Side-by-side signal comparison |
| 6 | `run_diagnostics(gpu_id)` | Read | Simulated diagnostic suite (ECC check, thermal, NVLink) |
| 7 | `drain_node(node_id)` | Write | Evicts all jobs from node, marks GPUs unavailable |
| 8 | `quarantine_gpu(gpu_id)` | Write | Sets status to QUARANTINED, removes from scheduler |
| 9 | `restart_driver(gpu_id)` | Write | Simulated driver reset — clears transient errors |
| 10 | `return_to_service(gpu_id)` | Write | Returns GPU to HEALTHY if diagnostics pass |

**Rule Engine Design**:

```go
// pkg/agent/rules.go
type Rule struct {
    Name       string
    Priority   int          // Lower = higher priority. First matching rule wins.
    Condition  func(g *GPU, events []Event) bool
    Diagnosis  string
    Confidence float64
    Actions    []Action     // Ordered list of tool calls to execute
}

var DiagnosticRules = []Rule{
    {
        Name:       "hardware_degradation",
        Priority:   1,
        Condition:  func(g *GPU, _ []Event) bool {
            return g.ECCErrorsWindow > 10 && g.XIDErrors > 0 && g.Performance < 0.8
        },
        Diagnosis:  "Probable GPU hardware degradation",
        Confidence: 0.93,
        // Note: ActionRequestRMA is an event-only outcome emitted after run_diagnostics fails
        Actions:    []Action{ActionQuarantine, ActionRunDiagnostics},
    },
    // ...
}
```

**Incident Lifecycle State Machine**:

Incidents are created automatically when a GPU transitions to the **CRITICAL threshold (health < 40)**.

- **Deduplication**: If an incident already exists for a GPU in DETECTED/INVESTIGATING/DIAGNOSED/REMEDIATING state, no new incident is created.
- **Cooldown & Reopen**: After an incident is RESOLVED, a new incident cannot be created for `cooldown_s` (default 300s). If it degrades past CRITICAL after the cooldown or returning to service, a new incident is created.

```
DETECTED → INVESTIGATING → DIAGNOSED → REMEDIATING → RESOLVED
                                                         │
                                                   ┌─────┘
                                                   ▼
                                              (archived)
```

**Tool execution model**: Tools are called synchronously by the agent goroutine. Write tools mutate GPU/scheduler state under the write lock.

---

### 6. Autonomous Remediation Engine

Closed-loop operational sequence — all in-process, no physical hardware risk:

```
Health score drops below CRITICAL threshold (health < 40)
        ↓
Incident created (DETECTED)
        ↓
AI Agent picks up incident
        ↓
Agent evaluates diagnostic rules
        ↓
Rule matched → quarantine_gpu(0042)
        ↓
GPU status → QUARANTINED
        ↓
Scheduler evicts GPU from active job
        ↓
Scheduler allocates warm spare, resumes job
        ↓
run_diagnostics(0042)
        ↓
GPU marked FAILED → RMA requested (simulated outcome event)
        ↓
Incident → RESOLVED (archived to event log)
```

---

### 7. Cluster Scheduler Simulation

**Job Model**:
```go
type Job struct {
    ID                string        `json:"id"`
    Type              JobType       `json:"type"`
    Priority          int           `json:"priority"`
    GPUCount          int           `json:"gpu_count"`
    GPUIDs            []string      `json:"gpu_ids"`
    Status            JobStatus     `json:"status"`
    CreatedAt         time.Time     `json:"created_at"`
    StartedAt         time.Time     `json:"started_at"`
    MigrationStartedAt time.Time    `json:"migration_started_at"`
    CheckpointAge     time.Duration `json:"checkpoint_age"`
    MigrationsCount   int           `json:"migrations_count"`
}

type JobStatus string
const (
    JobQueued    JobStatus = "QUEUED"
    JobRunning   JobStatus = "RUNNING"
    JobPaused    JobStatus = "PAUSED"
    JobMigrating JobStatus = "MIGRATING"
    JobCompleted JobStatus = "COMPLETED"
    JobFailed    JobStatus = "FAILED"
)
```

**Scheduling Behavior**:
- **Allocation**: The scheduler finds `gpu_count` HEALTHY GPUs.
- **Warm Spare Pool**: `warm_spare_ratio` (default 5%) of the fleet is reserved for failover.
- **Topology Preference**: Scheduler prefers spares on the same rack/node for NVLink topology preservation.
- **Eviction & Migration**: If a GPU in an active job transitions to CRITICAL or QUARANTINED, the scheduler evicts the degraded GPU. The job enters `MIGRATING` state for `migration_time_s` (default 120s) while a warm spare is allocated.
- **Checkpoint Freshness**: A migrating job loses `time_since_last_checkpoint` worth of progress, impacting completion time.
- **Spare Exhaustion**: If no warm spares are available, the job enters `PAUSED` and waits.
- **Availability Metrics**: These migration and queuing delays are directly reflected in MTTR, Availability, and Utilization metrics exported to Prometheus.

---

### 8. Performance Regression Detection

- Continuous benchmark tracking against golden baselines:
- **Regression threshold**: Configurable (default: -15%). Any GPU below this triggers an alert.
- **Automated Detection Trigger**: Drops below baseline are flagged as stragglers and can trigger investigations.

---

### 9. Production Observability: Native Prometheus + Grafana + Embedded Web Console

**9a. Prometheus Metrics Exporter (`/metrics`)**:
- Uses **canonical DCGM field names** from the schema (`dcgm_gpu_temp`, `dcgm_ecc_sbe_volatile_total`, etc.). Dataset-only labels are excluded.
- Fleet-level metrics: `fleet_gpus_total`, `fleet_availability_ratio`, `fleet_utilization_ratio`, etc.
- Operational metrics: `autopilot_remediations_total`, `autopilot_active_incidents`.

**9b. Pre-Built Grafana Dashboard (`observability/grafana/dashboards/fleet_overview.json`)**:
- Clean, zero-effort import for any team running Grafana.

**9c. Embedded Zero-Dependency Web Console (`http://localhost:8080`)**:
- Served directly by the Go binary. By default, the server binds to `127.0.0.1`. Use `--bind=0.0.0.0` to expose.

---

### 10. Structured Event Log

All incidents, diagnoses, and remediations emit structured JSON events to stdout and to an append-only log file:

```json
{
  "event_id": "evt-00001042",
  "timestamp": "2026-09-09T15:04:05Z",
  "type": "INCIDENT_CREATED",
  "severity": "CRITICAL",
  "gpu_id": "gpu-00042",
  "node_id": "node-006",
  "details": {
    "health_score": 38,
    "trigger": "health_below_threshold",
    "status": "CRITICAL"
  }
}
```

---

### 11. Evaluation Protocol

**Split protocol**:
- Time split: Train days 1-5, validate day 6, test day 7.
- Entity split: GPUs 1-7000 train, 7001-8500 val, 8501-10000 test.
- Scenario holdout: 8 of 10 failure scenarios in train; 2 held-out for test.
- Seed holdout: Different random seeds for noise generation and failure injection timing.
- OOD test: Different workload mix (90% serving vs 90% training), elevated ambient temp (+5°C), A100 profile.

**Metrics**:
- ROC-AUC ≥ 0.90
- PR-AUC ≥ 0.70
- Recall @ 2h horizon ≥ 0.85
- Recall @ 6h horizon ≥ 0.70
- Recall @ 24h horizon ≥ 0.50
- Precision @ alert budget ≥ 0.60 at 5 alerts/1000 GPU-days
- Brier score ≤ 0.10
- False alerts / 1000 GPU-days ≤ 10
- MTTR improvement: measured with vs without ML predictions
- Rules baseline comparison: ML must outperform rule engine alone

---

### 12. Kaggle / Colab Notebook Architecture & Fleet Evaluation Suite

The codebase packages map directly to **8 standalone notebooks**. Each notebook is self-contained and can run independently on Kaggle/Colab.

| Notebook | Title | Inputs | Outputs |
|---|---|---|---|
| NB 1 | Fleet Simulator | Config params | Parquet telemetry dataset |
| NB 2 | Failure Injection | Telemetry dataset | Labeled failure dataset (with ground-truth) |
| NB 3 | Feature Engineering | Labeled dataset | Feature matrix (EWMA, slopes, interactions) |
| NB 4 | Failure Prediction | Feature matrix | Trained XGBoost v1 model, ROC-AUC report |
| NB 5 | Anomaly Detection | Feature matrix | Isolation Forest / Autoencoder anomaly scores |
| NB 6 | RCA Agent | Telemetry dataset | Incident investigations, tool call traces |
| NB 7 | Autonomous Remediation | Incidents | End-to-end remediation simulation results |
| NB 8 | Fleet Evaluation & SLA | NB4 model + NB7 results | Evaluation protocol metrics (Recall horizons, Brier score, MTTR vs Rules Baseline) |

---

## CLI: `gpu-ctl` Commands

`gpu-ctl` is a separate lightweight HTTP client binary. It connects to the running server via REST API at `localhost:8080`.

- `cmd/server/main.go` → builds `gpu-autopilot` (Server)
- `cmd/cli/main.go` → builds `gpu-ctl` (Client)

```
gpu-ctl status                              # Fleet summary: total, healthy, degraded, critical, quarantined
gpu-ctl list [--status=DEGRADED]            # List GPUs with optional status filter
gpu-ctl inspect <gpu-id>                    # Detailed GPU health card with all signals
gpu-ctl inject <scenario> <gpu-id>          # Trigger failure scenario on a GPU
gpu-ctl jobs                                # List active training/serving jobs
gpu-ctl jobs create --gpus=8 --type=TRAINING # Submit a new job
gpu-ctl drain <node-id>                     # Drain all GPUs on a node
gpu-ctl quarantine <gpu-id>                 # Quarantine a GPU
gpu-ctl return <gpu-id>                     # Return a quarantined GPU to service
gpu-ctl export --records=1000000 --output=data/telemetry.parquet  # Export telemetry dataset
gpu-ctl predict <gpu-id>                    # Run ML prediction for a specific GPU
gpu-ctl incidents [--status=ACTIVE]         # List incidents
```

---

## File Layout & Organization

```
GPU-Fleet-Autopilot/
├── cmd/
│   ├── server/main.go               # Go binary entrypoint — starts everything
│   └── cli/main.go                  # `gpu-ctl` CLI utility
├── schema/
│   ├── telemetry_v1.json            # Canonical telemetry schema
│   └── model_v1.json                # Model export format spec
├── testdata/
│   └── golden/                      # Golden dataset and parity test fixtures
├── pkg/
│   ├── config/                      # Configuration loading & validation
│   ├── types/                       # Core data structures (GPU, Job, Incident)
│   ├── simulator/                   # Tick loop engine
│   ├── chaos/                       # Failure injection
│   ├── health/                      # Health scoring
│   ├── scheduler/                   # Workload scheduling (eviction, migration, spares)
│   ├── regression/                  # Performance regression detection
│   ├── agent/                       # AI Infrastructure Agent (Go-native rules)
│   ├── remediation/                 # Remediation controller
│   ├── ml/                          # Go-native ML inference
│   │   ├── evaluator.go             # XGBoost JSON tree walker
│   │   ├── features.go              # Feature extraction
│   │   └── models/                  # Pre-trained model artifacts
│   ├── telemetry/                   # Prometheus metrics & event logging
│   └── api/                         # REST API routes, WebSocket, Web Console
├── ml/                              # Python — OFFLINE ONLY
│   ├── dataset/
│   │   └── generator.py             # Parquet telemetry generator
│   └── model/
│       ├── feature_engineering.py   # Feature definitions
│       ├── train.py                 # XGBoost v1 training script
│       └── export.py                # Exports model to JSON
├── notebooks/                       # Kaggle/Colab notebooks
├── observability/                   # Prometheus and Grafana configs
├── scripts/                         # Setup and demo scripts
├── config.yaml                      # Fleet configuration
└── README.md                        # Documentation
```

---

## Phased Implementation Order

| Phase | Components | Why This Order |
|---|---|---|
| **Phase 0** | Contracts & Reproducibility | Defines schema/telemetry_v1.json, schema/model_v1.json, split protocol, and golden dataset. Base source of truth. |
| **Phase 1** | `pkg/types/`, `pkg/config/`, `pkg/simulator/`, `pkg/health/` | Foundation (Pure Go) — GPU struct, config, tick loop, health scoring. |
| **Phase 2** | `pkg/chaos/`, `pkg/telemetry/prometheus.go` | Simulation observable and testable via `/metrics`. |
| **Phase 3** | `pkg/api/`, `cmd/cli/main.go` | REST API + CLI (`gpu-ctl`) enables interaction. |
| **Phase 4** | `pkg/scheduler/`, `pkg/regression/` | Adds operational logic: job allocation, migration delays, stragglers. |
| **Phase 5** | `pkg/agent/`, `pkg/remediation/`, event log | AI agent + remediation logic. |
| **Phase 6** | `ml/` + `ml/model/train.py` | Offline Python generator + training pipeline + model export. Produces model artifact. |
| **Phase 7** | `pkg/ml/` | Go-native ML inference (loads model from Phase 6). |
| **Phase 8** | Web console + Grafana dashboard | Polish layer, observability UIs. |
| **Phase 9** | Kaggle/Colab notebooks | Packages components into NB1-NB8. |

> Phases 1–5 are **pure Go** with zero Python dependency. Phase 6 is Python offline. Phase 7 loads the Phase 6 artifact in Go.

---

## Verification Plan

### Unit Tests
| Package | Test | Expected |
|---|---|---|
| `pkg/health` | Score a GPU with known signal values | Exact health score, correct tier |
| `pkg/agent` | Match rules against known GPU states | Correct rule fires, correct actions |
| `pkg/ml` | Load exported model, predict on known input | Matches Python's prediction |
| `pkg/ml` | Golden-vector parity test | 100 Go/Python predictions match ≤ 1e-6 |
| `pkg/scheduler` | Allocate job, fail GPU, verify migration | Job MIGRATING, warm spare allocated, resumed |

### Benchmark Tests
```bash
go test -bench=BenchmarkTickLoop -benchtime=10s ./pkg/simulator/
```

### Integration Tests
1. **Failure Injection & Degradation**: Inject `GPU_THERMAL_FAILURE` on `gpu-00042` → verify temperature climbs, clock drops, health drops.
2. **Health Scorer Tiers**: Verify correct tier assignment.
3. **Performance Regression Detector**: Inject regression → verify incident created.
4. **Scheduler Rescheduling**: Allocate job → fail GPU → verify migration state and spare allocation.
5. **Prometheus Metrics Validation**: Query `/metrics` → verify canonical DCGM metrics.
6. **AI Agent End-to-End**: Trigger CRITICAL health → verify incident DETECTED, agent diagnoses, quarantines.
7. **ML Feature Parity**: Compute features in Go and Python → verify identical outputs.
8. **Evaluation Protocol Benchmarks**: Verify ROC-AUC ≥ 0.90, MTTR improvements, and rules baseline comparisons per the split protocol metrics.

### Lint & CI
```bash
golangci-lint run ./...
go vet ./...
ruff check ml/
pytest ml/tests/
```

### Manual / Operational Verification
1. Run `go build -o gpu-autopilot ./cmd/server && ./gpu-autopilot`.
2. Open Embedded Web Console (`http://localhost:8080`).
