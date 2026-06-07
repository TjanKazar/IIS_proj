"""
lstm_utils.py
─────────────
Lightweight LSTM quantile forecaster that replaces Amazon Chronos.
- LSTM with a two‑layer head (64 → 32 neurons)
- Pinball loss during training
- Wrapper interface for drop‑in compatibility

Memory: <100 KB per model on disk, ~0.2 MB at inference time.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Keep the same quantiles as your original Chronos setup
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


# ── Model definition ─────────────────────────────────────────────────────────
class LSTMQuantileForecaster(nn.Module):
    """
    LSTM encoder with a two‑hidden‑layer MLP head (64 → 32 neurons) as requested.

    Args:
        context_len: number of historical time steps
        pred_len: number of future steps to predict
        lstm_hidden: size of the LSTM hidden state
    """
    def __init__(self, context_len: int, pred_len: int, lstm_hidden: int = 32):
        super().__init__()
        self.pred_len = pred_len
        self.lstm_hidden = lstm_hidden

        # Single‑layer LSTM (input is 1‑D per time step)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # MLP head: LSTM hidden → 64 → 32 → output
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len * len(QUANTILE_LEVELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, context_len) or (context_len,)
        Returns:
            tensor of shape (batch, pred_len, n_quantiles)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)               # (1, context_len)
        x = x.unsqueeze(-1)                  # (batch, context_len, 1)

        out, _ = self.lstm(x)                # (batch, context_len, lstm_hidden)
        last_hidden = out[:, -1, :]          # take last time step
        pred = self.head(last_hidden)        # (batch, pred_len * n_quantiles)
        return pred.view(-1, self.pred_len, len(QUANTILE_LEVELS))


# ── Wrapper that mimics the Chronos pipeline interface ────────────────────────
class LSTMForecasterWrapper:
    """
    Replaces a Chronos pipeline with a trained LSTM model.
    Exposes the exact same predict() method as ChronosBoltPipeline.
    """
    def __init__(self, model_path: str, context_len: int, pred_len: int):
        self.model = LSTMQuantileForecaster(context_len, pred_len)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, ctx_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            ctx_tensor: 1-D torch.Tensor of shape (context_len,)
        Returns:
            mean: (pred_len,)  numpy array
            q10:  (pred_len,)  numpy array
            q90:  (pred_len,)  numpy array
        """
        with torch.no_grad():
            q_pred = self.model(ctx_tensor)       # (1, pred_len, 3)
        q_np = q_pred.squeeze(0).numpy()
        mean = q_np[:, 1]       # median = quantile 0.5
        q10  = q_np[:, 0]
        q90  = q_np[:, 2]
        return mean, q10, q90


# ── Loading helpers ──────────────────────────────────────────────────────────
def load_lstm_forecaster(metric_name: str, context_len: int = 12,
                         pred_len: int = 6, models_dir: str = "models") -> LSTMForecasterWrapper:
    """
    Load the appropriate .pth file for a given traffic metric.
    """
    name_map = {
        "Št. vozil [N/h]":   "volume_fc.pth",
        "St. vozil [N/h]":   "volume_fc.pth",
        "Hitrost [km/h]":    "speed_fc.pth",
        "Razmik [s]":        "headway_fc.pth",
        "Zasedenost [%]":    "occupancy_fc.pth",
    }
    
    if metric_name not in name_map:
        raise KeyError(
            f"Unknown metric '{metric_name}'. "
            f"Available metrics: {list(name_map.keys())}"
        )
    
    path = os.path.join(models_dir, name_map[metric_name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Did you run train_lstm_forecaster.py for this metric?"
        )
    
    return LSTMForecasterWrapper(path, context_len, pred_len)


def load_all_forecasters(metrics: List[str], context_len: int = 12,
                         pred_len: int = 6, models_dir: str = "models") -> dict:
    """Convenience: load forecasters for a list of metric names."""
    return {col: load_lstm_forecaster(col, context_len, pred_len, models_dir)
            for col in metrics}


# ── Pinball loss (for training) ──────────────────────────────────────────────
def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                 quantiles: torch.Tensor) -> torch.Tensor:
    """
    y_true: (batch, pred_len)
    y_pred: (batch, pred_len, n_quantiles)
    """
    errors = y_true.unsqueeze(-1) - y_pred
    loss = torch.max((quantiles - 1) * errors, quantiles * errors).mean()
    return loss