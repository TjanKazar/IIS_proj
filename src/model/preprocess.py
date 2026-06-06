import os
import random
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -- Load params from project root ---------------------------------------------
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

TEST_SIZE          = train_params["test_size"]
USE_TIME_SPLIT     = train_params["use_time_split"]

# MLP hyperparams — read from params.yaml under train.mlp, else use defaults
mlp_params = train_params.get("mlp", {})
HIDDEN_DIMS     = mlp_params.get("hidden_dims",     [256, 128, 64])  # layer widths
DROPOUT_RATE    = mlp_params.get("dropout_rate",    0.3)
BATCH_NORM      = mlp_params.get("batch_norm",      True)
LEARNING_RATE   = mlp_params.get("learning_rate",   1e-3)
WEIGHT_DECAY    = mlp_params.get("weight_decay",    1e-4)  # L2 regularisation
BATCH_SIZE      = mlp_params.get("batch_size",      512)
MAX_EPOCHS      = mlp_params.get("max_epochs",      100)
PATIENCE        = mlp_params.get("patience",        10)    # early stopping

MLFLOW_TRACKING_URI = train_params.get(
    "mlflow_tracking_uri",
    "https://dagshub.com/TjanKazar/IIS_proj.mlflow"
)
MLFLOW_EXPERIMENT = train_params.get("mlflow_experiment", "traffic_mlp_train")

# -- Reproducibility -----------------------------------------------------------
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Using device: {DEVICE}")

# -- Column name constants -----------------------------------------------------
COL_VOLUME    = "St. vozil [N/h]"
COL_SPEED     = "Hitrost [km/h]"
COL_HEADWAY   = "Razmik [s]"
COL_OCCUPANCY = "Zasedenost [%]"


# ==============================================================================
# Transformers (unchanged from original)
# ==============================================================================

class DatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_col: str, create_time_features: bool = True):
        self.timestamp_col = timestamp_col
        self.create_time_features = create_time_features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.timestamp_col] = pd.to_datetime(
            X[self.timestamp_col], format="%d.%m.%Y %H:%M:%S", errors="coerce"
        )
        if self.create_time_features:
            ts = X[self.timestamp_col]
            X["hour"]        = ts.dt.hour
            X["day_of_week"] = ts.dt.dayofweek
            X["month"]       = ts.dt.month
            X["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
            X["is_night"]    = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype(int)
        return X


class LocationForwardFiller(BaseEstimator, TransformerMixin):
    def __init__(self, road_col: str, location_col: str):
        self.road_col     = road_col
        self.location_col = location_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.location_col] = (
            X.groupby(self.road_col, sort=False)[self.location_col]
            .transform(lambda s: s.replace("", np.nan).ffill())
        )
        return X


class TimeDiffTransformer(BaseEstimator, TransformerMixin):
    _DIVISORS = {"seconds": 1, "minutes": 60, "hours": 3600}

    def __init__(self, timestamp_col: str, location_key_cols: list, unit: str = "minutes"):
        self.timestamp_col     = timestamp_col
        self.location_key_cols = location_key_cols
        self.unit              = unit

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        divisor   = self._DIVISORS.get(self.unit, 60)
        feat_name = f"{self.unit}_since_last_reading"
        X[feat_name] = (
            X.groupby(self.location_key_cols, sort=False)[self.timestamp_col]
            .transform(lambda s: s.sort_values().diff().dt.total_seconds() / divisor)
        )
        median_gap    = X[feat_name].median()
        X[feat_name]  = X[feat_name].fillna(median_gap)
        return X


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, location_key_cols: list, timestamp_col: str,
                 numeric_cols: list, target_col: str):
        self.location_key_cols = location_key_cols
        self.timestamp_col     = timestamp_col
        self.numeric_cols      = numeric_cols
        self.target_col        = target_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.sort_values(self.location_key_cols + [self.timestamp_col])
        for col in self.numeric_cols:
            if col in X.columns:
                lag_name = "prev_" + col.split("[")[0].strip().lower().replace(" ", "_").replace(".", "")
                X[lag_name] = X.groupby(self.location_key_cols, sort=False)[col].transform(lambda s: s.shift(1))
        enc_target = f"{self.target_col}_encoded"
        if enc_target in X.columns:
            X["prev_state"] = X.groupby(self.location_key_cols, sort=False)[enc_target].transform(lambda s: s.shift(1))
        lag_cols      = [c for c in X.columns if c.startswith("prev_")]
        X[lag_cols]   = X[lag_cols].fillna(0)
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.le_        = LabelEncoder()
        self.class_map_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.le_.fit(X[self.target_col].astype(str))
        self.class_map_ = dict(enumerate(self.le_.classes_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[f"{self.target_col}_encoded"] = self.le_.transform(X[self.target_col].astype(str))
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list):
        self.cols      = cols
        self.encoders_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.cols:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders_.items():
            known  = set(le.classes_)
            X[col] = X[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            X[f"{col}_encoded"] = le.transform(X[col])
        return X


class LeakageRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop: list):
        self.cols_to_drop = cols_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X    = X.copy()
        drop = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=drop)


class NoDataFilter(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str, no_data_label: str):
        self.target_col    = target_col
        self.no_data_label = no_data_label
        self.no_data_df_   = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mask             = X[self.target_col].astype(str) == self.no_data_label
        self.no_data_df_ = X[mask].copy()
        return X[~mask].copy()


# ==============================================================================
# MLP Model Definition
# ==============================================================================

class TrafficMLP(nn.Module):
    """
    Feedforward MLP for traffic state classification.

    Regularisation strategy:
      - BatchNorm before each activation  → stabilises training, acts as mild regulariser
      - Dropout after each activation     → primary regulariser, combats overfitting
      - Weight decay (L2) on optimiser    → penalises large weights globally

    The time-delta feature (minutes_since_last_reading) is treated as a regular
    input — no special handling needed since MLP makes no temporal assumptions.
    """

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

        layers.append(nn.Linear(in_dim, num_classes))  # logits out, no softmax (CrossEntropyLoss handles it)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# Training helpers
# ==============================================================================

def make_loaders(X_train, y_train, X_test, y_test, batch_size: int):
    """Wrap numpy arrays in PyTorch DataLoaders."""
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None):
    """Run one epoch; if optimizer is None, runs in eval mode."""
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


# ==============================================================================
# Main: preprocess + train with MLflow tracking
# ==============================================================================

if __name__ == "__main__":

    # ── Preprocessing ──────────────────────────────────────────────────────────

    print(f"[preprocess] Reading  : {INPUT_PATH}")
    df = pd.read_csv(os.path.join(project_root, INPUT_PATH))
    print(f"[preprocess] Rows in  : {len(df)}")

    filler = LocationForwardFiller(road_col=ROAD_COL, location_col=LOCATION_COL)
    df = filler.fit_transform(df)

    date_proc = DatePreprocessor(timestamp_col=TIMESTAMP_COL, create_time_features=CREATE_TIME_FEAT)
    df = date_proc.fit_transform(df)

    if FILTER_NO_DATA:
        no_data_filter = NoDataFilter(target_col=TARGET_COL, no_data_label=NO_DATA_LABEL)
        df = no_data_filter.fit_transform(df)
        n_removed = len(no_data_filter.no_data_df_)
        print(f"[preprocess] Removed  : {n_removed} '{NO_DATA_LABEL}' rows "
              f"({100 * n_removed / (len(df) + n_removed):.1f}% of total)")
        no_data_path = os.path.join(project_root, NO_DATA_OUTPUT)
        os.makedirs(os.path.dirname(no_data_path), exist_ok=True)

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

    # ── Train/test split ───────────────────────────────────────────────────────

    TARGET_ENC_COL = f"{TARGET_COL}_encoded"
    feature_cols   = [c for c in df.columns if c != TARGET_ENC_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_ENC_COL].values.astype(np.int64)

    if USE_TIME_SPLIT:
        split_idx = int(len(df) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        split_strategy = "time_based"
        print(f"\n[train] Time-based split at index {split_idx}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        split_strategy = "random_stratified"
        print(f"\n[train] Random stratified split")

    print(f"[train] Train size : {len(X_train)}")
    print(f"[train] Test size  : {len(X_test)}")

    # ── Feature scaling ────────────────────────────────────────────────────────
    # Critical for MLP — unlike tree-based models, MLPs are sensitive to scale.
    # Fit only on train set to prevent data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Build model ────────────────────────────────────────────────────────────

    num_classes = len(target_enc.class_map_)
    input_dim   = X_train.shape[1]

    model = TrafficMLP(
        input_dim    = input_dim,
        hidden_dims  = HIDDEN_DIMS,
        num_classes  = num_classes,
        dropout_rate = DROPOUT_RATE,
        batch_norm   = BATCH_NORM,
    ).to(DEVICE)

    print(f"\n[train] Model architecture:\n{model}")
    print(f"[train] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,   # L2 regularisation
    )
    # Reduce LR when validation loss plateaus — helps squeeze out last gains
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # ── MLflow experiment ──────────────────────────────────────────────────────

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="train_traffic_mlp"):

        # Log preprocessing params
        mlflow.log_param("input_path",         INPUT_PATH)
        mlflow.log_param("target_col",         TARGET_COL)
        mlflow.log_param("filter_no_data",     FILTER_NO_DATA)
        mlflow.log_param("create_time_diff",   CREATE_TIME_DIFF)
        mlflow.log_param("create_time_feat",   CREATE_TIME_FEAT)
        mlflow.log_param("remove_leakage",     REMOVE_LEAKAGE)
        mlflow.log_param("time_diff_unit",     TIME_DIFF_UNIT)

        # Log split params
        mlflow.log_param("test_size",          TEST_SIZE)
        mlflow.log_param("use_time_split",     USE_TIME_SPLIT)
        mlflow.log_param("split_strategy",     split_strategy)
        mlflow.log_param("random_state",       RANDOM_STATE)
        mlflow.log_param("n_train_samples",    len(X_train))
        mlflow.log_param("n_test_samples",     len(X_test))
        mlflow.log_param("n_features",         input_dim)
        mlflow.log_param("target_classes",     str(target_enc.class_map_))

        # Log MLP hyperparams
        mlflow.log_param("hidden_dims",        str(HIDDEN_DIMS))
        mlflow.log_param("dropout_rate",       DROPOUT_RATE)
        mlflow.log_param("batch_norm",         BATCH_NORM)
        mlflow.log_param("learning_rate",      LEARNING_RATE)
        mlflow.log_param("weight_decay",       WEIGHT_DECAY)
        mlflow.log_param("batch_size",         BATCH_SIZE)
        mlflow.log_param("max_epochs",         MAX_EPOCHS)
        mlflow.log_param("patience",           PATIENCE)
        mlflow.log_param("device",             str(DEVICE))

        # ── Training loop with early stopping ──────────────────────────────────

        best_val_loss   = float("inf")
        best_epoch      = 0
        epochs_no_improve = 0
        best_state_dict = None

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
            val_loss,   val_acc   = run_epoch(model, test_loader,  criterion, optimizer=None)

            scheduler.step(val_loss)

            # Log per-epoch metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc",  train_acc,  step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            mlflow.log_metric("val_acc",    val_acc,    step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Early stopping: track best val loss, restore best weights at end
            if val_loss < best_val_loss - 1e-5:
                best_val_loss    = val_loss
                best_epoch       = epoch
                epochs_no_improve = 0
                best_state_dict  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"\n[train] Early stopping at epoch {epoch} "
                          f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})")
                    break

        # Restore best weights
        model.load_state_dict(best_state_dict)
        mlflow.log_param("best_epoch",      best_epoch)
        mlflow.log_param("best_val_loss",   round(best_val_loss, 6))

        # ── Final evaluation ───────────────────────────────────────────────────

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                logits = model(X_batch.to(DEVICE))
                all_preds.append(logits.argmax(dim=1).cpu().numpy())

        y_pred = np.concatenate(all_preds)
        acc    = accuracy_score(y_test, y_pred)

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

        # Log aggregate metrics
        mlflow.log_metric("test_accuracy",           acc)
        mlflow.log_metric("test_macro_f1",           report_dict["macro avg"]["f1-score"])
        mlflow.log_metric("test_macro_precision",    report_dict["macro avg"]["precision"])
        mlflow.log_metric("test_macro_recall",       report_dict["macro avg"]["recall"])
        mlflow.log_metric("test_weighted_f1",        report_dict["weighted avg"]["f1-score"])
        mlflow.log_metric("test_weighted_precision", report_dict["weighted avg"]["precision"])
        mlflow.log_metric("test_weighted_recall",    report_dict["weighted avg"]["recall"])

        # Log per-class metrics
        for class_name, class_metrics in report_dict.items():
            if isinstance(class_metrics, dict):
                safe_name = class_name.replace(" ", "_").replace("/", "_")
                mlflow.log_metric(f"{safe_name}_f1",        class_metrics["f1-score"])
                mlflow.log_metric(f"{safe_name}_precision", class_metrics["precision"])
                mlflow.log_metric(f"{safe_name}_recall",    class_metrics["recall"])

        # ── Save & log artifacts ───────────────────────────────────────────────

        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path    = os.path.join(models_dir, "mlp_traffic.pt")
        encoder_path  = os.path.join(models_dir, "label_encoder.pkl")
        scaler_path   = os.path.join(models_dir, "scaler.pkl")
        features_path = os.path.join(models_dir, "feature_cols.pkl")

        torch.save(model.state_dict(), model_path)
        joblib.dump(target_enc,   encoder_path)
        joblib.dump(scaler,       scaler_path)
        joblib.dump(feature_cols, features_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(features_path)
        mlflow.log_artifact(output_path)   # preprocessed CSV

        # Log model natively via mlflow.pytorch
        mlflow.pytorch.log_model(model, artifact_path="mlp_model")

        print(f"\n[train] Model saved     : {model_path}")
        print(f"[train] Encoder saved   : {encoder_path}")
        print(f"[train] Scaler saved    : {scaler_path}")
        print(f"[train] Features saved  : {features_path}")
        print(f"[train] MLflow run logged to: {MLFLOW_TRACKING_URI}")

        print(df[TARGET_ENC_COL].value_counts(normalize=True))