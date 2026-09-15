#!/usr/bin/env python3
"""
High-Fidelity Physics-Based GPU Telemetry Generator.
Generates realistic cluster telemetry adhering strictly to schema/telemetry_v1.json.
Uses only Python standard library (csv, json, math, random, time).
"""

import argparse
import csv
import math
import os
import random
import sys
import time

SCENARIOS = [
    "NONE",
    "GPU_THERMAL_FAILURE",
    "GPU_ECC_FAILURE",
    "GPU_XID_ERROR",
    "NVLINK_DEGRADATION",
    "PERFORMANCE_REGRESSION",
    "DRIVER_CRASH",
    "GPU_MEMORY_DEGRADATION",
    "NETWORK_PACKET_LOSS",
    "RDMA_FAILURE",
    "STORAGE_LATENCY",
]


class VirtualGPU:
    def __init__(self, gpu_id, node_id, rack_id, cluster_id="us-east-cluster-1", model="H100"):
        self.gpu_id = gpu_id
        self.node_id = node_id
        self.rack_id = rack_id
        self.cluster_id = cluster_id
        self.model = model

        self.ambient_temp = 23.0 + random.uniform(-1.5, 1.5)
        self.temp_phys = 64.0 + random.uniform(-2.0, 2.0)
        self.tau = 25.0
        self.r_thermal = 0.075
        self.is_throttled = False

        self.utilization = 10.0
        self.memory_utilization = 15.0
        self.power = 250.0
        self.sm_clock = 1980
        self.clock_throttle = 0
        self.performance = 1.0

        self.ecc_sbe_total = 0
        self.ecc_dbe_total = 0
        self.latest_xid = 0
        self.nvlink_errors = 0
        self.network_errors = 0

        self.allocated_job_id = None
        self.scenario = "NONE"
        self.ticks_in_scenario = 0

    def step(self, dt=2.0, is_busy=True):
        # 1. Base Workload
        if is_busy:
            base_u = random.gauss(91.0, 4.0)
        else:
            base_u = random.gauss(8.0, 2.0)
        self.utilization = max(0.0, min(100.0, base_u))
        self.memory_utilization = max(5.0, min(95.0, self.utilization * 0.85 + random.uniform(-3, 3)))

        # Power draw
        p_idle = 150.0
        p_dyn = 550.0
        p_noise = random.gauss(0.0, 6.0)
        self.power = max(100.0, min(750.0, p_idle + p_dyn * (self.utilization / 100.0) + p_noise))

        # 2. Chaos scenario degradation
        if self.scenario != "NONE":
            self.ticks_in_scenario += 1
            self._apply_scenario()
        else:
            self.performance = 1.0
            self.clock_throttle = 0

        # 3. Newton's cooling ODE
        d_temp = (dt / self.tau) * (-(self.temp_phys - self.ambient_temp) + self.r_thermal * self.power)
        self.temp_phys = max(15.0, min(115.0, self.temp_phys + d_temp))

        # 4. Thermal Throttling
        if self.temp_phys >= 92.0:
            self.is_throttled = True
            self.clock_throttle |= 1
        elif self.temp_phys <= 83.0 and self.scenario != "GPU_THERMAL_FAILURE":
            self.is_throttled = False
            self.clock_throttle &= ~1

        if self.is_throttled:
            throttled_clock = max(500, int(1980 - 15 * (self.temp_phys - 83.0)))
            self.sm_clock = throttled_clock
            self.performance = throttled_clock / 1980.0
        elif self.scenario == "NONE":
            self.sm_clock = 1980

        observed_temp = max(15.0, min(115.0, self.temp_phys + random.gauss(0.0, 1.1)))
        return observed_temp

    def _apply_scenario(self):
        if self.scenario == "GPU_THERMAL_FAILURE":
            self.r_thermal = min(0.24, self.r_thermal + 0.005)
        elif self.scenario == "GPU_ECC_FAILURE":
            self.ecc_sbe_total += random.randint(3, 7)
            if self.ticks_in_scenario > 15:
                self.ecc_dbe_total += 1
                self.latest_xid = 62
                self.performance = 0.0
        elif self.scenario == "GPU_XID_ERROR":
            self.latest_xid = random.choice([31, 43, 62, 79, 92])
            self.performance = 0.20
        elif self.scenario == "NVLINK_DEGRADATION":
            self.nvlink_errors += random.randint(5, 15)
            self.performance = max(0.40, self.performance - 0.05)
            if self.nvlink_errors > 40:
                self.latest_xid = 92
        elif self.scenario == "PERFORMANCE_REGRESSION":
            self.performance = 0.65  # -35% regression
        elif self.scenario == "DRIVER_CRASH":
            self.sm_clock = 0
            self.utilization = 0.0
            self.performance = 0.0
            self.latest_xid = 79
        elif self.scenario == "GPU_MEMORY_DEGRADATION":
            self.memory_utilization = min(100.0, self.memory_utilization + 5.0)
            self.performance = max(0.35, self.performance - 0.04)
        elif self.scenario == "NETWORK_PACKET_LOSS":
            self.network_errors += random.randint(2, 6)
            self.performance = max(0.50, self.performance - 0.03)
        elif self.scenario == "RDMA_FAILURE":
            self.network_errors += random.randint(8, 16)
            if random.random() < 0.5:
                self.utilization = 0.0
                self.performance = 0.10
        elif self.scenario == "STORAGE_LATENCY":
            self.utilization = 5.0
            self.performance = 0.30


def generate_dataset(output_path, num_gpus=128, ticks=100, dt=2.0):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    gpus = []
    for i in range(num_gpus):
        gpu_idx = i + 1
        node_idx = (i // 8) + 1
        rack_idx = ((node_idx - 1) // 16) + 1
        g = VirtualGPU(
            gpu_id=f"gpu-{gpu_idx:05d}",
            node_id=f"node-{node_idx:03d}",
            rack_id=f"rack-{rack_idx:02d}",
        )
        # Assign 80% to active jobs
        if random.random() < 0.8:
            job_idx = (i // 8) + 1
            g.allocated_job_id = f"job-{job_idx:05d}"
        gpus.append(g)

    # Pick ~10% of GPUs to inject failure scenarios
    failing_count = max(1, int(num_gpus * 0.10))
    failing_gpus = random.sample(gpus, failing_count)
    for idx, fg in enumerate(failing_gpus):
        fg.scenario = SCENARIOS[1 + (idx % (len(SCENARIOS) - 1))]

    fieldnames = [
        "schema_version",
        "timestamp",
        "gpu_id",
        "node_id",
        "rack_id",
        "cluster_id",
        "job_id",
        "dcgm_gpu_temp",
        "dcgm_power_usage",
        "dcgm_gpu_utilization",
        "dcgm_mem_copy_utilization",
        "dcgm_sm_clock",
        "dcgm_clock_throttle_reasons",
        "dcgm_ecc_sbe_volatile_total",
        "dcgm_ecc_dbe_volatile_total",
        "dcgm_xid_errors",
        "dcgm_nvlink_error_count",
        "network_errors_total",
        "performance_ratio",
        "failure_in_next_2h",
        "failure_in_next_6h",
        "failure_in_next_24h",
        "failure_type",
    ]

    base_time_ms = int(time.time() * 1000) - (ticks * int(dt * 1000))
    total_rows = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in range(ticks):
            timestamp = base_time_ms + int(t * dt * 1000)
            for g in gpus:
                temp = g.step(dt=dt, is_busy=(g.allocated_job_id is not None))
                is_fail = g.scenario != "NONE"

                row = {
                    "schema_version": "1.0",
                    "timestamp": timestamp,
                    "gpu_id": g.gpu_id,
                    "node_id": g.node_id,
                    "rack_id": g.rack_id,
                    "cluster_id": g.cluster_id,
                    "job_id": g.allocated_job_id or "",
                    "dcgm_gpu_temp": round(temp, 2),
                    "dcgm_power_usage": round(g.power, 2),
                    "dcgm_gpu_utilization": round(g.utilization, 2),
                    "dcgm_mem_copy_utilization": round(g.memory_utilization, 2),
                    "dcgm_sm_clock": g.sm_clock,
                    "dcgm_clock_throttle_reasons": g.clock_throttle,
                    "dcgm_ecc_sbe_volatile_total": g.ecc_sbe_total,
                    "dcgm_ecc_dbe_volatile_total": g.ecc_dbe_total,
                    "dcgm_xid_errors": g.latest_xid,
                    "dcgm_nvlink_error_count": g.nvlink_errors,
                    "network_errors_total": g.network_errors,
                    "performance_ratio": round(g.performance, 3),
                    "failure_in_next_2h": 1 if is_fail else 0,
                    "failure_in_next_6h": 1 if is_fail else 0,
                    "failure_in_next_24h": 1 if is_fail else 0,
                    "failure_type": g.scenario,
                }
                writer.writerow(row)
                total_rows += 1

    print(f"Generated {total_rows} records across {num_gpus} GPUs ({failing_count} degraded) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GPU telemetry dataset")
    parser.add_argument("--output", default="data/sample_telemetry.csv", help="Output CSV path")
    parser.add_argument("--gpus", type=int, default=128, help="Number of GPUs to simulate")
    parser.add_argument("--ticks", type=int, default=100, help="Number of time steps")
    args = parser.parse_args()

    generate_dataset(args.output, num_gpus=args.gpus, ticks=args.ticks)
