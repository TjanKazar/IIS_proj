"""
dashboard.py
────────────
Simple admin dashboard for MLP + LSTM metrics.
Reads the latest MLflow run from DagsHub.
"""

import os
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient

# ── Config ────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/TjanKazar/IIS_proj.mlflow"
)
EXPERIMENT_NAME = "traffic_mlp_train"

st.set_page_config(page_title="Traffic Model Dashboard", layout="wide")
st.title("🚦 Traffic Model Dashboard")

# ── Connect to MLflow ─────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    st.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
    st.stop()

# Get latest run
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)
if not runs:
    st.warning("No runs found yet.")
    st.stop()

run = runs[0]
metrics = run.data.metrics
params = run.data.params

# ── Display run info ──────────────────────────────────────────────────────
st.subheader(f" Latest Run: `{run.info.run_name}`")
st.caption(f"Run ID: `{run.info.run_id}`  |  {run.info.start_time}")

# ── MLP Classification Metrics ────────────────────────────────────────────
st.header(" MLP Classifier Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
col2.metric("Macro F1", f"{metrics.get('test_macro_f1', 0):.3f}")
col3.metric("Weighted F1", f"{metrics.get('test_weighted_f1', 0):.3f}")
col4.metric("Best Val Loss", f"{metrics.get('best_val_loss', 0):.4f}")

# Per‑class metrics
st.subheader("Per‑Class Metrics")
class_metrics = []
for key, val in metrics.items():
    if key.endswith("_f1") and not key.startswith("test_"):
        class_name = key.replace("_f1", "").replace("_", " ")
        precision = metrics.get(f"{key.replace('_f1', '')}_precision", 0)
        recall = metrics.get(f"{key.replace('_f1', '')}_recall", 0)
        class_metrics.append({
            "Class": class_name,
            "Precision": precision,
            "Recall": recall,
            "F1": val
        })

if class_metrics:
    df_class = pd.DataFrame(class_metrics)
    st.dataframe(df_class, use_container_width=True)
    
    fig = px.bar(df_class.melt(id_vars="Class", var_name="Metric", value_name="Value"),
                 x="Class", y="Value", color="Metric", barmode="group",
                 title="Per‑Class Precision, Recall, F1")
    st.plotly_chart(fig, use_container_width=True)

# ── LSTM Forecast Metrics ─────────────────────────────────────────────────
st.header(" LSTM Forecast MAE")

# Map of expected MAE metric names (adjust to what you log)
forecast_metrics_map = {
    "Volume": "mae_volume",
    "Speed": "mae_speed",
    "Headway": "mae_headway",
    "Occupancy": "mae_occupancy",
}

mae_data = {}
for label, key in forecast_metrics_map.items():
    mae_data[label] = metrics.get(key, None)

# If you haven't logged these yet, show a hint
if all(v is None for v in mae_data.values()):
    st.info("MAE metrics not found in this run. Add them to your training script with `mlflow.log_metrics({'mae_volume': ...})`.")
else:
    cols = st.columns(len(mae_data))
    for col, (label, mae) in zip(cols, mae_data.items()):
        if mae is not None:
            col.metric(f"{label} MAE", f"{mae:.2f}")
        else:
            col.metric(f"{label} MAE", "N/A")

# ── Hyperparameters ───────────────────────────────────────────────────────
st.header("⚙️ Hyperparameters")
important_params = ["hidden_dims", "dropout_rate", "batch_norm", "learning_rate",
                    "batch_size", "max_epochs", "patience", "forecast_model"]
param_dict = {p: params.get(p, "N/A") for p in important_params}
st.json(param_dict)

# ── Raw metrics dump ──────────────────────────────────────────────────────
with st.expander("🔍 Full Metrics & Params"):
    st.json({"metrics": metrics, "params": params})