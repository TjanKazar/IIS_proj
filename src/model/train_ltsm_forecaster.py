"""
train_lstm_forecaster.py
────────────────────────
Offline training script – run ONCE to create the .pth model files.
Reads configuration from params.yaml.

Usage:
    python train_lstm_forecaster.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "src", "model"))

from model.lstm_utils import LSTMQuantileForecaster, pinball_loss, QUANTILE_LEVELS


def prepare_sequences(series: np.ndarray, context_len: int, pred_len: int):
    X, y = [], []
    for i in range(len(series) - context_len - pred_len + 1):
        X.append(series[i : i + context_len])
        y.append(series[i + context_len : i + context_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    params_path = os.path.join(project_root, "params.yaml")
    with open(params_path) as f:
        all_params = yaml.safe_load(f)
    
    params = all_params["preprocess"]
    train_params = all_params["train"]
    forecast_params = train_params.get("chronos", {})  # or "forecast"
    
    INPUT_PATH     = params["input"]
    CONTEXT_LEN    = forecast_params.get("context_len", 12)
    PRED_LEN       = forecast_params.get("prediction_len", 6)
    TARGET_METRICS = forecast_params.get(
        "target_numeric_cols",
        ["Št. vozil [N/h]", "Hitrost [km/h]", "Razmik [s]", "Zasedenost [%]"],
    )
    
    EPOCHS     = forecast_params.get("epochs", 100)
    BATCH_SIZE = forecast_params.get("batch_size", 256)
    LR         = forecast_params.get("learning_rate", 1e-3)
    
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    name_map = {
        "Št. vozil [N/h]": "volume_fc.pth",
        "St. vozil [N/h]": "volume_fc.pth",
        "Hitrost [km/h]":  "speed_fc.pth",
        "Razmik [s]":      "headway_fc.pth",
        "Zasedenost [%]":  "occupancy_fc.pth",
    }
    
    csv_path = os.path.join(project_root, INPUT_PATH)
    print(f"[train_lstm] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[train_lstm] Columns: {list(df.columns)}")
    
    for metric in TARGET_METRICS:
        if metric not in df.columns:
            print(f"[train_lstm] WARNING: '{metric}' not in CSV, skipping.")
            continue
        
        series = df[metric].dropna().values.astype(np.float32)
        print(f"\n[train_lstm] Training '{metric}' — {len(series)} points")
        
        X, y = prepare_sequences(series, CONTEXT_LEN, PRED_LEN)
        print(f"               Samples: {len(X)}")
        
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        model = LSTMQuantileForecaster(CONTEXT_LEN, PRED_LEN)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        q_tensor = torch.tensor(QUANTILE_LEVELS)
        
        best_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            total_loss = 0.0
            for xb, yb in loader:
                y_pred = model(xb)
                loss = pinball_loss(yb, y_pred, q_tensor)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item() * len(xb)
            
            avg_loss = total_loss / len(dataset)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"               Epoch {epoch:3d}  loss {avg_loss:.6f}")
        
        model.load_state_dict(best_state)
        out_path = os.path.join(models_dir, name_map[metric])
        torch.save(model.state_dict(), out_path)
        print(f"               Saved: {out_path}  (loss: {best_loss:.6f})")
    
    print(f"\n[train_lstm] Done. Models saved to {models_dir}/")


if __name__ == "__main__":
    main()