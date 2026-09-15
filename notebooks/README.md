# Kaggle & Colab Notebook Suite — GPU Fleet Autopilot

This directory contains the **8-part Jupyter/Colab notebook architecture** for research, dataset generation, failure prediction modeling, and SLA evaluation.

---

## Notebook Architecture Map

| Notebook | Title | Description | Primary Libraries |
|---|---|---|---|
| **[01_fleet_simulator.ipynb](./01_fleet_simulator.ipynb)** | Fleet Simulator | Physics ODE integration (Newton's cooling law, dynamic power, throttling) | `numpy`, `pandas`, `matplotlib` |
| **[02_failure_injection.ipynb](./02_failure_injection.ipynb)** | Failure Injection | Telemetry exploration across all 10 hardware failure degradation curves | `pandas`, `seaborn` |
| **[03_feature_engineering.ipynb](./03_feature_engineering.ipynb)** | Feature Engineering | Rolling EWMA (5m, 15m), slopes, interaction terms, leakage controls | `pandas`, `numpy` |
| **[04_failure_prediction_xgboost.ipynb](./04_failure_prediction_xgboost.ipynb)** | Failure Prediction | Trains XGBoost v1 model and exports to neutral JSON for Rust inference | `xgboost`, `scikit-learn` |
| **[05_anomaly_detection.ipynb](./05_anomaly_detection.ipynb)** | Anomaly Detection | Unsupervised multivariate anomaly detection (Isolation Forest) | `scikit-learn` |
| **[06_rca_agent.ipynb](./06_rca_agent.ipynb)** | RCA Agent | Automated diagnostic tool traces & root cause determination rule engine | `pandas`, `numpy` |
| **[07_autonomous_remediation.ipynb](./07_autonomous_remediation.ipynb)** | Autonomous Remediation | Closed-loop job eviction, warm spare failover, MTTR comparison | `pandas`, `seaborn` |
| **[08_fleet_evaluation_sla.ipynb](./08_fleet_evaluation_sla.ipynb)** | Fleet Evaluation & SLA | Rigorous SLA verification protocol (ROC-AUC, Brier score, MTTR) | `tabulate`, `pandas` |

---

## Running Locally

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost tabulate jupyterlab
   ```

2. Generate the sample dataset:
   ```bash
   python3 ml/dataset/generator.py --output data/sample_telemetry.csv --gpus 128 --ticks 100
   ```

3. Launch JupyterLab:
   ```bash
   jupyter lab notebooks/
   ```

---

## Running on Google Colab / Kaggle

Upload any `.ipynb` file directly to [Google Colab](https://colab.research.google.com) or create a Kaggle Notebook. In the first cell, mount or generate the synthetic data:
```python
!python -c "import urllib.request; ..." # or run generator directly
```
