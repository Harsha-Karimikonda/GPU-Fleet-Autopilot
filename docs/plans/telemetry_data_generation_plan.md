# High-Fidelity Physics-Based GPU Telemetry Generator & Validation Pipeline

Design and implementation plan for generating realistic, production-grade GPU telemetry datasets to train the failure prediction model (XGBoost / LightGBM) and validate that the simulated data reliably mirrors physical hardware dynamics. Similarity to public traces is plausibility validation — not proof of transfer to physical hardware. This simulation demonstrates predictive methodology, not production-ready failure prediction. It utilizes the canonical telemetry schema v1.0.

> **Important Disclaimer**: This telemetry generator produces synthetic data grounded in physics-based models and semiconductor failure statistics. Distribution similarity to public cluster traces (Alibaba, Google) validates the plausibility of the simulation's statistical properties. It does NOT constitute proof that models trained on this synthetic data will generalize to physical GPU hardware. Real-world deployment requires validation against production telemetry from actual GPU clusters.

---

## 1. Mathematical Modeling of Realism

### A. Thermodynamics & Dynamic Power

* **Power Draw Model:**
  $$P(t) = P_{\text{idle}} + P_{\text{dynamic}} \cdot u(t) + \mathcal{N}(0, \sigma_P^2)$$
  Where $u(t) \in [0, 1]$ simulates distributed training phases (forward/backward compute bursts $\sim 95\%$, AllReduce communication $\sim 45\%$, and checkpoint stalls $\sim 15\%$).

* **Thermal Inertia (Newton's Cooling Law):**
  Physical state variable $T_{phys}$ evolves WITHOUT noise using an Euler method integrator with a physics timestep of $dt = 2\text{s}$:
  $$\tau \frac{dT_{phys}}{dt} = -(T_{phys} - T_{ambient}) + R_{thermal} \cdot P(t)$$
  The observed temperature adds measurement noise ($\sigma_{sensor} = 1.5^\circ\text{C}$):
  $$T_{observed} = T_{phys} + \mathcal{N}(0, \sigma_{sensor}^2)$$
  Increasing $R_{thermal}$ (thermal resistance) means worse cooling $\to$ higher temperature for the same power input. During cooling degradation failure scenarios, $R_{thermal}$ increases to simulate thermal paste dryout or fan failure.

* **Thermal Throttling Hysteresis:**
  ```python
  if T >= 92:
      is_throttled = True
  elif T <= 83:
      is_throttled = False
  # else: is_throttled unchanged (hysteresis)

  if is_throttled:
      SM_clock = max(500, 1980 - 15 * (T - 83))  # MHz
  else:
      SM_clock = 1980  # MHz
  ```

### B. Degradation & Correlated Failure Injections

* **Ambient Temperature**: All GPUs in a rack share an ambient temperature. Rack ambient varies slowly (daily cycle + HVAC events). A rack cooling failure raises ambient for all 8 GPUs on every node in that rack.
* **PDU/Fan Faults**: A rack-level PDU partial failure degrades power delivery to all nodes $\to$ increased thermal resistance for all GPUs.
* **Job-Level Workload Correlation**: All GPUs in the same training job experience the same workload phase (forward/backward/AllReduce/checkpoint). An AllReduce stall affects all GPUs in the job simultaneously.
* **Network Degradation**: A ToR switch degradation causes packet loss for all GPUs on that switch's nodes.

### C. Failure Type Mapping

- **THERMAL**: $R_{thermal}$ increases, driving thermal creep, throttling, and a compute regression.
- **ECC_MEMORY**: Poisson arrival for soft errors, clustering into specific banks, culminating in an uncorrectable Double-Bit Error and `Xid 62`.
- **NVLINK**: CRC replay counter accumulation $\to$ dropped flits $\to$ link retraining failure (`Xid 92`).
- **XID_CRASH**: Sudden failure causing a specific active XID code.
- **STRAGGLER**: Performance ratio degrades steadily relative to peers.
- **NETWORK**: `network_errors_total` climbs, `performance_ratio` drops due to communication delays. Maps to `NETWORK_PACKET_LOSS`.
- **RDMA**: `network_errors_total` spikes, GPU appears stalled (utilization drops to near 0 periodically). Maps to `RDMA_FAILURE`.
- **STORAGE**: `dcgm_gpu_utilization` shows periodic drops to 0 (I/O wait), `performance_ratio` drops. Maps to `STORAGE_LATENCY`.
- **DRIVER**: GPU telemetry goes flat/zero (driver hung), then `dcgm_xid_errors` = 79 (fallen off bus). Maps to `DRIVER_CRASH`.
- **MEMORY_DEGRADATION**: `dcgm_mem_copy_utilization` anomalous, `performance_ratio` gradual decline. Maps to `GPU_MEMORY_DEGRADATION`.

---

## 2. Canonical Telemetry Schema v1.0

All fields use DCGM-canonical names. The same schema is used by the Go runtime, CSV/Parquet exports, and ML features.

### Canonical Fields
- `schema_version` (string): "1.0"
- `timestamp` (int64): epoch milliseconds
- `gpu_id` (string): "gpu-00001" .. "gpu-10000"
- `node_id` (string): "node-001" .. "node-1250"
- `rack_id` (string): "rack-01" .. "rack-XX"
- `cluster_id` (string): "us-east-cluster-1"
- `job_id` (string): allocated job ID, empty if idle
- `dcgm_gpu_temp` (float64, °C): GPU temperature. Gauge.
- `dcgm_power_usage` (float64, Watts): GPU power draw. Gauge.
- `dcgm_gpu_utilization` (float64, %): SM utilization 0-100. Gauge.
- `dcgm_mem_copy_utilization` (float64, %): HBM utilization 0-100. Gauge.
- `dcgm_sm_clock` (int, MHz): SM clock speed. Gauge.
- `dcgm_clock_throttle_reasons` (int, bitmask): 0=None, 1=Thermal, 2=Power, 4=Board. Gauge.
- `dcgm_ecc_sbe_volatile_total` (int): Cumulative SBE count. Monotonic counter — never decrements.
- `dcgm_ecc_dbe_volatile_total` (int): Cumulative DBE count. Monotonic counter.
- `dcgm_xid_errors` (int): Last active XID code (0, 31, 43, 62, 79, 92). Point-in-time.
- `dcgm_nvlink_error_count` (int): Cumulative NVLink CRC/replay errors. Monotonic counter.
- `network_errors_total` (int): Cumulative IB/RoCE packet drops. Monotonic counter.
- `performance_ratio` (float64): Actual throughput / baseline. 1.0 = 100%. Gauge.

### Dataset-Only Fields
- `failure_in_next_2h` (int, 0 or 1)
- `failure_in_next_6h` (int, 0 or 1)
- `failure_in_next_24h` (int, 0 or 1)
- `failure_type` (string): `NONE`, `THERMAL`, `ECC_MEMORY`, `NVLINK`, `XID_CRASH`, `STRAGGLER`, `NETWORK`, `RDMA`, `STORAGE`, `DRIVER`, `MEMORY_DEGRADATION`

### Time Scales
- Physics timestep: 2000ms (configurable). Internal ODE integration.
- Telemetry sampling interval: 30s (configurable). What gets persisted to dataset.
- Feature windows: 5m, 15m, 1h EWMA.
- Prediction target horizons: 2h, 6h, 24h.
- Dataset sizing: 10,000 GPUs × 24h × 1 sample/30s = 28.8M rows/day. 7-day dataset = ~200M rows. Format: Parquet, partitioned by day.

---

## 3. Label & Evaluation Protocol

### Label Definitions
- Failure unit: GPU-incident (transition to QUARANTINED/FAILED).
- 1.5% of GPU-hours in pre-failure window.
- Labels set retrospectively.
- Censoring: post-terminal records excluded.
- Node/network attribution: to the GPU experiencing the symptom.

### Leakage Controls
- Excluded from ML features: `dcgm_xid_errors`, `status`, `health_score`.
- Diagnostic accuracy report: with and without terminal indicators.

### Evaluation Protocol
- Time split: Train days 1-5, val day 6, test day 7.
- Entity split: GPUs 1-7000 train, 7001-8500 val, 8501-10000 test.
- Scenario holdout: 8/10 train, 2/10 test.
- OOD test: different workload, ambient temp, hardware profile.
- Metrics Target:
  - ROC-AUC $\ge$ 0.90
  - PR-AUC $\ge$ 0.70
  - recall@2h $\ge$ 0.85
  - recall@6h $\ge$ 0.70
  - recall@24h $\ge$ 0.50
  - precision@budget $\ge$ 0.60
  - Brier $\le$ 0.10
  - false alerts $\le$ 10/1000 GPU-days
  - MTTR improvement measured
  - rules baseline comparison

---

## 4. Sim-to-Real Statistical Validation Suite

Before any model training, the dataset must pass 7 quantitative validation gates:

1. **Power-Temperature Lagged Cross-Correlation**: Compute cross-correlation function CCF(P, T) at lags $\tau = 0, 1, 2, ..., 30$ samples. CCF peak $\ge 0.80$ AND peak lag $\tau \in [2, 10]$ seconds. Step-response test: exponential rise with $\tau \approx 5\text{s}$.
2. **Counter Monotonicity & Sensor Integrity**: Cumulative counters never decrement. Sensor noise $\sigma \le 1.5^\circ\text{C}$. SM clock only changes at throttle boundaries.
3. **Precursor Detection Lead-Time**: Degradation anomalies emerge 2-24h before terminal event. Validated per failure type.
4. **Class Imbalance Realism**: 0.5%-2.0% of GPU-hours in pre-failure window.
5. **Distribution Divergence vs. Public Traces**: Compare generated distributions against Alibaba GPU cluster traces (Ali-Cluster-GPU-2024) and Google cluster data. Metrics: KS test ($p > 0.05$ for temperature, power), Wasserstein distance, Population Stability Index (PSI < 0.25). *Explicit disclaimer: Distribution similarity to public traces validates plausibility of the simulation's statistical properties. It does NOT prove that models trained on this synthetic data will transfer to physical hardware.*
6. **Thermal Hysteresis Verification**: Verify enter/exit threshold asymmetry. A GPU that enters throttle at $92^\circ\text{C}$ should not exit until $\le 83^\circ\text{C}$.
7. **Correlated Failure Verification**: When a rack cooling fault is injected, verify all GPUs on that rack show correlated temperature increases. When a job-level AllReduce stall occurs, verify all GPUs in that job show synchronized utilization drops.

---

## 5. File Organization & Deliverables

```
GPU-Fleet-Autopilot/
├── docs/
│   └── plans/
│       ├── implementation_plan.md          # 10 Core Pillars of the overall system
│       └── telemetry_data_generation_plan.md# This document: Data physics & validation spec
├── schema/                                 # Canonical protocol buffers / schema definitions
├── ml/
│   ├── dataset/
│   │   ├── physics.py                      # Thermal ODE solver & power dynamics
│   │   ├── degradation.py                  # Weibull, Poisson, and XID failure curves
│   │   ├── generator.py                    # Multi-GPU batch Parquet generator
│   │   └── validate.py                     # Sim-to-Real statistical validation suite
│   ├── model/
│   │   ├── feature_engineering.py          # Rolling EWMA, slopes, interaction terms
│   │   ├── train.py                        # XGBoost v1 training pipeline
│   │   └── export.py                       # Export trained model artifacts
│   ├── testdata/
│   │   └── golden/                         # Golden datasets for validation testing
│   └── requirements.txt                    # numpy, scipy, pandas, xgboost, scikit-learn
```

---

## 6. Verification Plan

### Automated Tests
* **Golden Dataset Generation & Validation Suite:**
  ```bash
  python -m ml.dataset.generator --output testdata/golden/dataset.parquet
  python -m ml.dataset.validate --input testdata/golden/dataset.parquet
  ```
* **ML Model Performance Target & Split Protocol:**
  Ensure training follows the rigorous entity/time/scenario split protocol:
  ```bash
  python ml/model/train.py --data testdata/golden/dataset.parquet
  ```
  * Assert metrics against targets: $\text{ROC-AUC} \ge 0.90$, $\text{PR-AUC} \ge 0.70$, $\text{recall@2h} \ge 0.85$, $\text{recall@6h} \ge 0.70$, $\text{recall@24h} \ge 0.50$, $\text{precision@budget} \ge 0.60$, $\text{Brier} \le 0.10$, $\text{false alerts} \le 10/1000\text{ GPU-days}$.
