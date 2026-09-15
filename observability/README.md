# Observability Suite — GPU Fleet Autopilot

Turnkey observability stack providing production-grade Grafana dashboards and Prometheus metric scraping for GPU Fleet Autopilot.

---

## Architecture

```
┌───────────────────────────────┐
│  GPU Fleet Autopilot Server   │
│  http://localhost:8080        │
│  Exports: /metrics (DCGM fmt) │
└──────────────┬────────────────┘
               │ (Scrapes every 5s)
               ▼
┌───────────────────────────────┐
│          Prometheus           │
│  http://localhost:9090        │
└──────────────┬────────────────┘
               │ (Datasource)
               ▼
┌───────────────────────────────┐
│           Grafana             │
│  http://localhost:3000        │
│  Pre-provisioned dashboard:   │
│  "Fleet Overview"             │
└───────────────────────────────┘
```

---

## Quickstart with Docker Compose

1. Start your `gpu-autopilot` server in the background:
   ```bash
   cargo run --bin gpu-autopilot -- --config config.yaml
   ```

2. Start Prometheus & Grafana:
   ```bash
   docker compose -f observability/docker-compose.yml up -d
   ```

3. Open Grafana:
   - **URL**: [http://localhost:3000](http://localhost:3000)
   - **Login**: `admin` / `admin`
   - **Dashboard**: Navigate to **Dashboards > GPU Fleet > Fleet Overview**.

4. Open Prometheus:
   - **URL**: [http://localhost:9090](http://localhost:9090)
   - Query raw DCGM metrics (e.g., `dcgm_gpu_temp`, `fleet_availability_ratio`).

---

## Standalone Grafana Import

If you run an existing Grafana instance:
1. Ensure your Prometheus instance scrapes `http://<server-ip>:8080/metrics`.
2. In Grafana, click **Dashboards > New > Import**.
3. Upload [`observability/grafana/dashboards/fleet_overview.json`](./grafana/dashboards/fleet_overview.json).
4. Select your Prometheus data source and click **Import**.

