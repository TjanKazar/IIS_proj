"""
chronos_utils.py
────────────────
Utilities for loading and running Amazon Chronos time-series forecasting.

Install the dependency with:
    pip install chronos-forecasting
or: pip install git+https://github.com/amazon-science/chronos-forecasting.git

Memory guide for the GTX 1650 (4 GB VRAM):
    chronos-t5-tiny  : ~400 MB  — fits easily, very fast
    chronos-t5-mini  : ~900 MB  — fits easily
    chronos-t5-small : ~1.8 GB  — fits comfortably  ← recommended
    chronos-t5-base  : ~3.5 GB  — tight, may OOM
    chronos-t5-large : ~6+ GB   — will OOM, don't use

dtype=torch.bfloat16 cuts VRAM roughly in half vs float32.
"""

import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error


# ── Model loading ─────────────────────────────────────────────────────────────

def load_chronos_pipeline(model_id: str, device: torch.device):
    """
    Load a Chronos pipeline from HuggingFace.
    Falls back to CPU automatically if the GPU runs out of memory.
    """
    from chronos import ChronosPipeline  # lazy import — script loads without it

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[chronos] Loading '{model_id}' on {device} with dtype={dtype}")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=str(device),
            torch_dtype=dtype,
        )
        print("[chronos] Model loaded on GPU.")
    except RuntimeError as e:
        print(f"[chronos] GPU OOM ({e}), falling back to CPU.")
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    return pipeline


# ── Context preparation ───────────────────────────────────────────────────────

def prepare_chronos_context(
    raw_df: pd.DataFrame,
    location_key_cols: list,
    timestamp_col: str,
    target_numeric_cols: list,
    context_len: int,
) -> dict:
    """
    For each (location, direction) group, extract the last `context_len`
    readings of each numeric column as a context tensor.

    Returns:
        dict mapping location_key → {col_name: torch.Tensor of shape (1, T)}
    """
    groups = {}
    for key, grp in raw_df.groupby(location_key_cols, sort=False):
        grp = grp.sort_values(timestamp_col)
        col_tensors = {}
        for col in target_numeric_cols:
            if col not in grp.columns:
                continue
            series = grp[col].dropna().values[-context_len:]  # last N readings
            if len(series) < 2:
                continue  # not enough history
            col_tensors[col] = torch.tensor(series, dtype=torch.float32).unsqueeze(0)  # (1, T)
        if col_tensors:
            groups[key] = col_tensors
    return groups


# ── Forecasting ───────────────────────────────────────────────────────────────

def run_chronos_forecast(
    pipeline,
    context_groups: dict,
    prediction_len: int,
    num_samples: int = 20,
) -> pd.DataFrame:
    """
    Run Chronos on each location group and return a tidy DataFrame of forecasts.

    Each row = one location × one column × one future step.
    Columns: location_key, series_col, step, mean, q10, q90

    `num_samples` controls the Monte Carlo sample count for uncertainty bounds.
    Lower = faster; 20 is a good balance for a 4 GB GPU.
    """
    rows = []
    for location_key, col_tensors in context_groups.items():
        for col_name, context_tensor in col_tensors.items():
            ctx_1d   = context_tensor.squeeze(0)  # (1, T) → (T,)
            forecast = pipeline.predict(ctx_1d, prediction_len, num_samples=num_samples)
            # shape: (1, num_samples, prediction_len)

            forecast_np = forecast.squeeze(0).numpy()       # (num_samples, prediction_len)
            mean_fc     = forecast_np.mean(axis=0)           # (prediction_len,)
            q10_fc      = np.percentile(forecast_np, 10, axis=0)
            q90_fc      = np.percentile(forecast_np, 90, axis=0)

            for step_idx in range(prediction_len):
                rows.append({
                    "location_key": str(location_key),
                    "series_col":   col_name,
                    "step":         step_idx + 1,           # 1-indexed
                    "mean":         mean_fc[step_idx],
                    "q10":          q10_fc[step_idx],
                    "q90":          q90_fc[step_idx],
                })

    return pd.DataFrame(rows)


# ── MAE evaluation ────────────────────────────────────────────────────────────

def evaluate_chronos_mae(
    raw_df: pd.DataFrame,
    location_key_cols: list,
    timestamp_col: str,
    target_numeric_cols: list,
    context_len: int,
    prediction_len: int,
    pipeline,
    n_eval_groups: int = 10,
) -> dict:
    """
    Quick MAE evaluation: for each of `n_eval_groups` random location groups,
    use the first (N - prediction_len) readings as context and the last
    `prediction_len` readings as ground truth.

    Returns a dict {col_name: mean_absolute_error}.
    """
    all_groups = list(raw_df.groupby(location_key_cols, sort=False))
    sampled    = random.sample(all_groups, min(n_eval_groups, len(all_groups)))

    col_errors = {col: [] for col in target_numeric_cols}

    for _, grp in sampled:
        grp = grp.sort_values(timestamp_col)
        for col in target_numeric_cols:
            if col not in grp.columns:
                continue
            series = grp[col].dropna().values
            needed = context_len + prediction_len
            if len(series) < needed:
                continue

            context_vals = series[-(context_len + prediction_len) : -prediction_len]
            ground_truth = series[-prediction_len:]

            ctx_tensor = torch.tensor(context_vals, dtype=torch.float32)  # 1D
            forecast   = pipeline.predict(
                ctx_tensor, prediction_length=prediction_len, num_samples=20
            )
            pred_mean = forecast.squeeze(0).numpy().mean(axis=0)

            col_errors[col].append(mean_absolute_error(ground_truth, pred_mean))

    return {col: float(np.mean(errs)) for col, errs in col_errors.items() if errs}