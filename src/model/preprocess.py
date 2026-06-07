"""
preprocess.py
─────────────
Main entry point: preprocessing → MLP training → LSTM 30-min forecast.

Depends on:
    transforms.py      — sklearn preprocessing transformers
    lstm_utils.py      — LSTM model loading / forecasting / evaluation
"""

import os
import random

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from lstm_utils import (
    load_all_forecasters,
    QUANTILE_LEVELS,
)
from transforms import (
    CategoricalEncoder,
    DatePreprocessor,
    LagFeatureTransformer,
    LeakageRemover,
    LocationForwardFiller,
    NoDataFilter,
    TargetEncoder,
    TimeDiffTransformer,
)

# ── Load params ───────────────────────────────────────────────────────────────

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
params_path  = os.path.join(project_root, "params.yaml")
all_params   = yaml.safe_load(open(params_path))
params       = all_params["preprocess"]
train_params = all_params["train"]

INPUT_PATH         = params["input"]
OUTPUT_PATH        = params["output"]
RANDOM_STATE       = params["random_state"]
TARGET_COL         = params["target_col"]
TIMESTAMP_COL      = params["timestamp_col"]
LOCATION_COL       = params["location_col"]
DIRECTION_COL      = params["direction_col"]
ROAD_COL           = params["road_col"]
TIME_DIFF_UNIT     = params["time_diff_unit"]
LOCATION_KEY_COLS  = params["location_key_cols"]
CREATE_TIME_DIFF   = params["create_time_diff"]
CREATE_TIME_FEAT   = params["create_time_features"]
REMOVE_LEAKAGE     = params["remove_location_leakage"]
FILTER_NO_DATA     = params["filter_no_data"]
NO_DATA_LABEL      = params["no_data_label"]
NO_DATA_OUTPUT     = params["no_data_output"]

TEST_SIZE      = train_params["test_size"]
USE_TIME_SPLIT = train_params["use_time_split"]

mlp_params   = train_params.get("mlp", {})
HIDDEN_DIMS  = mlp_params.get("hidden_dims",   [256, 128, 64])
DROPOUT_RATE = mlp_params.get("dropout_rate",  0.3)
BATCH_NORM   = mlp_params.get("batch_norm",    True)
LEARNING_RATE = mlp_params.get("learning_rate", 1e-3)
WEIGHT_DECAY  = mlp_params.get("weight_decay",  1e-4)
BATCH_SIZE   = mlp_params.get("batch_size",    512)
MAX_EPOCHS   = mlp_params.get("max_epochs",    100)
PATIENCE     = mlp_params.get("patience",      10)

# ── LSTM forecast parameters (replaces Chronos) ──────────────────────────────
forecast_params      = train_params.get("forecast", {})
FORECAST_CONTEXT_LEN = forecast_params.get("context_len",    12)
FORECAST_PRED_LEN    = forecast_params.get("prediction_len", 6)
FORECAST_TARGET_COLS = forecast_params.get(
    "target_numeric_cols",
    ["Št. vozil [N/h]", "Hitrost [km/h]", "Razmik [s]", "Zasedenost [%]"],
)
FORECAST_LOCATION = forecast_params.get("forecast_location", None)
RUN_FORECAST = forecast_params.get("enabled", True)

MLFLOW_TRACKING_URI = train_params.get(
    "mlflow_tracking_uri",
    "https://dagshub.com/TjanKazar/IIS_proj.mlflow",
)
MLFLOW_EXPERIMENT = train_params.get("mlflow_experiment", "traffic_mlp_train")

# ── Column name constants ─────────────────────────────────────────────────────

COL_VOLUME    = "Št. vozil [N/h]"
COL_SPEED     = "Hitrost [km/h]"
COL_HEADWAY   = "Razmik [s]"
COL_OCCUPANCY = "Zasedenost [%]"

# ── Reproducibility ───────────────────────────────────────────────────────────

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Using device: {DEVICE}")


# ── MLP model definition ──────────────────────────────────────────────────────

class TrafficMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int,
                 dropout_rate: float = 0.3, batch_norm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DataLoader helpers ────────────────────────────────────────────────────────

def make_loaders(X_train, y_train, X_test, y_test, batch_size: int):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss    += loss.item() * len(y_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += len(y_batch)
    return total_loss / total_samples, total_correct / total_samples


# ── Simple MAE evaluator ──────────────────────────────────────────────────────

def evaluate_mae(series_dict, context_len, pred_len, forecasters, n_eval_windows=5):
    errors = {col: [] for col in series_dict}
    for col, ser in series_dict.items():
        if len(ser) < context_len + pred_len:
            continue
        for _ in range(n_eval_windows):
            start = np.random.randint(0, len(ser) - context_len - pred_len)
            ctx = ser[start : start + context_len]
            true = ser[start + context_len : start + context_len + pred_len]
            ctx_tensor = torch.tensor(ctx, dtype=torch.float32)
            mean_pred, _, _ = forecasters[col].predict(ctx_tensor)
            errors[col].append(np.mean(np.abs(true - mean_pred)))
    return {col: float(np.mean(errs)) for col, errs in errors.items() if errs}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print(f"[preprocess] Reading  : {INPUT_PATH}")
    raw_df = pd.read_csv(os.path.join(project_root, INPUT_PATH))
    df     = raw_df.copy()
    print(f"[preprocess] Rows in  : {len(df)}")

    filler = LocationForwardFiller(road_col=ROAD_COL, location_col=LOCATION_COL)
    df     = filler.fit_transform(df)

    df_with_timestamps = df.copy()

    date_proc = DatePreprocessor(timestamp_col=TIMESTAMP_COL, create_time_features=CREATE_TIME_FEAT)
    df = date_proc.fit_transform(df)
    df_with_timestamps = date_proc.fit_transform(df_with_timestamps)

    if FILTER_NO_DATA:
        no_data_filter = NoDataFilter(target_col=TARGET_COL, no_data_label=NO_DATA_LABEL)
        df = no_data_filter.fit_transform(df)
        n_removed = len(no_data_filter.no_data_df_)
        print(f"[preprocess] Removed  : {n_removed} '{NO_DATA_LABEL}' rows "
              f"({100 * n_removed / (len(df) + n_removed):.1f}% of total)")
        no_data_path = os.path.join(project_root, NO_DATA_OUTPUT)
        os.makedirs(os.path.dirname(no_data_path), exist_ok=True)
        df_with_timestamps = df_with_timestamps[
            df_with_timestamps[TARGET_COL].astype(str) != NO_DATA_LABEL
        ].copy()

    target_enc = TargetEncoder(target_col=TARGET_COL)
    df = target_enc.fit_transform(df)
    print(f"[preprocess] Target classes   : {target_enc.class_map_}")

    if CREATE_TIME_DIFF:
        time_diff = TimeDiffTransformer(
            timestamp_col=TIMESTAMP_COL,
            location_key_cols=LOCATION_KEY_COLS,
            unit=TIME_DIFF_UNIT,
        )
        df = time_diff.fit_transform(df)

        time_diff_col = f"{TIME_DIFF_UNIT}_since_last_reading"
        if time_diff_col in df.columns:
            n_before = len(df)
            df = df[df[time_diff_col].between(4.5, 5.5)].copy()
            n_removed_gap = n_before - len(df)
            print(f"[preprocess] Interval filter : removed {n_removed_gap} rows "
                  f"({100 * n_removed_gap / n_before:.1f}%) where gap ≠ 5 min")
            print(f"[preprocess] Rows remaining  : {len(df)}")
            if time_diff_col in df_with_timestamps.columns:
                df_with_timestamps = df_with_timestamps[
                    df_with_timestamps[time_diff_col].between(4.5, 5.5)
                ].copy()

    lag_tf = LagFeatureTransformer(
        location_key_cols=LOCATION_KEY_COLS,
        timestamp_col=TIMESTAMP_COL,
        numeric_cols=[COL_VOLUME, COL_SPEED, COL_HEADWAY, COL_OCCUPANCY],
        target_col=TARGET_COL,
    )
    df = lag_tf.fit_transform(df)

    cat_enc = CategoricalEncoder(cols=[ROAD_COL, DIRECTION_COL, LOCATION_COL])
    df = cat_enc.fit_transform(df)

    if REMOVE_LEAKAGE:
        leakage_remover = LeakageRemover(
            cols_to_drop=[ROAD_COL, DIRECTION_COL, LOCATION_COL, TARGET_COL]
        )
        df = leakage_remover.fit_transform(df)

    df = df.drop(columns=[TIMESTAMP_COL], errors="ignore")

    output_path = os.path.join(project_root, OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Rows out : {len(df)}")
    print(f"[preprocess] Columns  : {list(df.columns)}")
    print(f"[preprocess] Saved to : {output_path}")

    # ── Train / test split ─────────────────────────────────────────────────────
    TARGET_ENC_COL = f"{TARGET_COL}_encoded"
    feature_cols = [c for c in df.columns if c != TARGET_ENC_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_ENC_COL].values.astype(np.int64)

    if USE_TIME_SPLIT:
        split_idx = int(len(df) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        split_strategy = "time_based"
        print(f"\n[train] Time-based split at index {split_idx}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        split_strategy = "random_stratified"
        print(f"\n[train] Random stratified split")

    print(f"[train] Train size : {len(X_train)}")
    print(f"[train] Test size  : {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Build & train MLP ──────────────────────────────────────────────────────
    num_classes = len(target_enc.class_map_)
    input_dim = X_train.shape[1]

    model = TrafficMLP(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        batch_norm=BATCH_NORM,
    ).to(DEVICE)

    print(f"\n[train] Model architecture:\n{model}")
    print(f"[train] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="train_traffic_mlp_lstm"):

        mlflow.log_params({
            "input_path": INPUT_PATH,
            "target_col": TARGET_COL,
            "filter_no_data": FILTER_NO_DATA,
            "create_time_diff": CREATE_TIME_DIFF,
            "create_time_feat": CREATE_TIME_FEAT,
            "remove_leakage": REMOVE_LEAKAGE,
            "time_diff_unit": TIME_DIFF_UNIT,
            "test_size": TEST_SIZE,
            "use_time_split": USE_TIME_SPLIT,
            "split_strategy": split_strategy,
            "random_state": RANDOM_STATE,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": input_dim,
            "target_classes": str(target_enc.class_map_),
            "hidden_dims": str(HIDDEN_DIMS),
            "dropout_rate": DROPOUT_RATE,
            "batch_norm": BATCH_NORM,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "device": str(DEVICE),
            "forecast_model": "LSTM_quantile",
        })

        best_val_loss, best_epoch, epochs_no_improve = float("inf"), 0, 0
        best_state_dict = None

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer=None)
            scheduler.step(val_loss)

            mlflow.log_metrics(
                {"train_loss": train_loss, "train_acc": train_acc,
                 "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"\n[train] Early stopping at epoch {epoch} "
                          f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})")
                    break

        model.load_state_dict(best_state_dict)
        mlflow.log_params({"best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 6)})

        # Final evaluation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                logits = model(X_batch.to(DEVICE))
                all_preds.append(logits.argmax(dim=1).cpu().numpy())

        y_pred = np.concatenate(all_preds)
        acc = accuracy_score(y_test, y_pred)
        report_dict = classification_report(
            y_test, y_pred,
            target_names=[target_enc.class_map_[i] for i in sorted(target_enc.class_map_)],
            output_dict=True,
        )

        print(f"\n[train] Test accuracy : {acc:.4f}")
        print("\n[train] Classification report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[target_enc.class_map_[i] for i in sorted(target_enc.class_map_)],
        ))

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_macro_f1": report_dict["macro avg"]["f1-score"],
            "test_macro_precision": report_dict["macro avg"]["precision"],
            "test_macro_recall": report_dict["macro avg"]["recall"],
            "test_weighted_f1": report_dict["weighted avg"]["f1-score"],
            "test_weighted_precision": report_dict["weighted avg"]["precision"],
            "test_weighted_recall": report_dict["weighted avg"]["recall"],
        })
        for class_name, class_metrics in report_dict.items():
            if isinstance(class_metrics, dict):
                safe_name = class_name.replace(" ", "_").replace("/", "_")
                mlflow.log_metrics({
                    f"{safe_name}_f1": class_metrics["f1-score"],
                    f"{safe_name}_precision": class_metrics["precision"],
                    f"{safe_name}_recall": class_metrics["recall"],
                })

        # Save MLP artifacts
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, "mlp_traffic.pt")
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        features_path = os.path.join(models_dir, "feature_cols.pkl")

        torch.save(model.state_dict(), model_path)
        joblib.dump(target_enc, encoder_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_cols, features_path)

        for path in [model_path, encoder_path, scaler_path, features_path, output_path]:
            mlflow.log_artifact(path)
        mlflow.pytorch.log_model(model, name="mlp_model")

        print(f"\n[train] Model saved     : {model_path}")
        print(f"[train] Encoder saved   : {encoder_path}")
        print(f"[train] Scaler saved    : {scaler_path}")
        print(f"[train] Features saved  : {features_path}")

    # ── LSTM forecast ─────────────────────────────────────────────────────────
    if RUN_FORECAST:
        print("\n" + "=" * 70)
        print("[forecast] Starting 30-minute-ahead LSTM forecast")
        print(f"[forecast] Context length : {FORECAST_CONTEXT_LEN}")
        print(f"[forecast] Predict steps  : {FORECAST_PRED_LEN}")
        print(f"[forecast] Metrics        : {FORECAST_TARGET_COLS}")
        print(f"[forecast] Location       : {FORECAST_LOCATION}")
        print("=" * 70)

        if FORECAST_LOCATION is None:
            print("[forecast] No forecast_location set, skipping forecast.")
        else:
            loc_key = tuple(FORECAST_LOCATION)
            available_keys = set(
                tuple(k)
                for k in df_with_timestamps[LOCATION_KEY_COLS].drop_duplicates().values.tolist()
            )
            if loc_key not in available_keys:
                print(f"[forecast] WARNING: location {loc_key} not found in data. Skipping forecast.")
            else:
                # Load LSTM forecasters
                forecasters = load_all_forecasters(
                    FORECAST_TARGET_COLS,
                    context_len=FORECAST_CONTEXT_LEN,
                    pred_len=FORECAST_PRED_LEN,
                    models_dir=models_dir,
                )

                # Filter data to requested location
                loc_filter = pd.Series([True] * len(df_with_timestamps), index=df_with_timestamps.index)
                for col, val in zip(LOCATION_KEY_COLS, FORECAST_LOCATION):
                    loc_filter &= df_with_timestamps[col].astype(str) == str(val)
                loc_df = df_with_timestamps[loc_filter].sort_values(TIMESTAMP_COL)

                print(f"[forecast] Rows for this location : {len(loc_df)}")

                # Forecast each metric
                forecasts = {}
                for col in FORECAST_TARGET_COLS:
                    if col not in loc_df.columns:
                        continue
                    series = loc_df[col].dropna().values[-FORECAST_CONTEXT_LEN:]
                    if len(series) < FORECAST_CONTEXT_LEN:
                        continue
                    ctx_tensor = torch.tensor(series, dtype=torch.float32)
                    mean_fc, q10_fc, q90_fc = forecasters[col].predict(ctx_tensor)
                    forecasts[col] = (mean_fc, q10_fc, q90_fc)

                # Display forecasts
                for col_name, (mean_fc, q10_fc, q90_fc) in forecasts.items():
                    print(f"\n[forecast] {col_name}:")
                    unit_map = {
                        COL_VOLUME: "vehicles/h",
                        COL_SPEED: "km/h",
                        COL_HEADWAY: "sec",
                        COL_OCCUPANCY: "%",
                    }
                    unit = unit_map.get(col_name, "")
                    for i, (m, lo, hi) in enumerate(zip(mean_fc, q10_fc, q90_fc), 1):
                        print(f"  t+{i*5:2d} min : {m:.1f} {unit}  [q10={lo:.1f}, q90={hi:.1f}]")

                # Get 30-min forecast values
                v = forecasts[COL_VOLUME][0][-1] if COL_VOLUME in forecasts else None
                s = forecasts[COL_SPEED][0][-1] if COL_SPEED in forecasts else None
                h = forecasts[COL_HEADWAY][0][-1] if COL_HEADWAY in forecasts else None
                o = forecasts[COL_OCCUPANCY][0][-1] if COL_OCCUPANCY in forecasts else None

                # Patch into MLP and classify
                mlp_loc_df = df[df.index.isin(loc_df.index)]
                if len(mlp_loc_df) > 0:
                    mlp_row = mlp_loc_df.iloc[-1][feature_cols].values.astype(np.float32).copy()

                    for col, val in [(COL_VOLUME, v), (COL_SPEED, s), (COL_HEADWAY, h), (COL_OCCUPANCY, o)]:
                        if col in feature_cols and val is not None:
                            mlp_row[feature_cols.index(col)] = val

                    mlp_row_scaled = scaler.transform(mlp_row.reshape(1, -1))
                    mlp_input = torch.tensor(mlp_row_scaled, dtype=torch.float32).to(DEVICE)

                    model.eval()
                    with torch.no_grad():
                        logits = model(mlp_input)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_class = int(logits.argmax(dim=1).cpu().item())

                    predicted_stanje = target_enc.class_map_[pred_class]
                    print(f"\n[forecast→MLP] Predicted Stanje at t+30min : '{predicted_stanje}'")
                    print("[forecast→MLP] Class probabilities:")
                    for cls_idx, cls_name in sorted(target_enc.class_map_.items()):
                        print(f"  {cls_name:20s}: {probs[cls_idx]:.3f}")

                del forecasters

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[train] MLflow run logged to: {MLFLOW_TRACKING_URI}")
    print(df[TARGET_ENC_COL].value_counts(normalize=True))