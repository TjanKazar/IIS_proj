import os
import random
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import dagshub

dagshub.init(repo_owner="TjanKazar", repo_name="IIS_proj", mlflow=True)

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

# XGBoost hyperparams — read from params.yaml if present, else use defaults
xgb_params = train_params.get("xgboost", {})
N_ESTIMATORS      = xgb_params.get("n_estimators", 500)
MAX_DEPTH         = xgb_params.get("max_depth", 6)
LEARNING_RATE     = xgb_params.get("learning_rate", 0.05)
SUBSAMPLE         = xgb_params.get("subsample", 0.8)
COLSAMPLE_BYTREE  = xgb_params.get("colsample_bytree", 0.8)
EARLY_STOPPING    = xgb_params.get("early_stopping_rounds", 20)

# MLflow tracking URI — override in params.yaml under train.mlflow_tracking_uri
MLFLOW_TRACKING_URI = train_params.get(
    "mlflow_tracking_uri",
    "https://dagshub.com/TjanKazar/IIS_proj.mlflow"   # replace with your URI
)
MLFLOW_EXPERIMENT   = train_params.get("mlflow_experiment", "traffic_xgboost_train")

# -- Reproducibility -----------------------------------------------------------
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
    X = df[feature_cols].values
    y = df[TARGET_ENC_COL].values

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

    # ── MLflow experiment ──────────────────────────────────────────────────────

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="train_traffic_xgboost"):

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
        mlflow.log_param("n_features",         len(feature_cols))
        mlflow.log_param("target_classes",     str(target_enc.class_map_))

        # Log XGBoost hyperparams
        mlflow.log_param("n_estimators",       N_ESTIMATORS)
        mlflow.log_param("max_depth",          MAX_DEPTH)
        mlflow.log_param("learning_rate",      LEARNING_RATE)
        mlflow.log_param("subsample",          SUBSAMPLE)
        mlflow.log_param("colsample_bytree",   COLSAMPLE_BYTREE)
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING)

        # ── Train XGBoost ──────────────────────────────────────────────────────

        model = XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            early_stopping_rounds=EARLY_STOPPING,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Log best iteration
        mlflow.log_param("best_iteration", model.best_iteration)

        # ── Evaluate ───────────────────────────────────────────────────────────

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        # Per-class precision, recall, F1
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
        mlflow.log_metric("test_accuracy",          acc)
        mlflow.log_metric("test_macro_f1",          report_dict["macro avg"]["f1-score"])
        mlflow.log_metric("test_macro_precision",   report_dict["macro avg"]["precision"])
        mlflow.log_metric("test_macro_recall",      report_dict["macro avg"]["recall"])
        mlflow.log_metric("test_weighted_f1",       report_dict["weighted avg"]["f1-score"])
        mlflow.log_metric("test_weighted_precision",report_dict["weighted avg"]["precision"])
        mlflow.log_metric("test_weighted_recall",   report_dict["weighted avg"]["recall"])

        # Log per-class metrics (useful for imbalanced traffic-state classes)
        for class_name, class_metrics in report_dict.items():
            if isinstance(class_metrics, dict):  # skip "accuracy" scalar entry
                safe_name = class_name.replace(" ", "_").replace("/", "_")
                mlflow.log_metric(f"{safe_name}_f1",        class_metrics["f1-score"])
                mlflow.log_metric(f"{safe_name}_precision", class_metrics["precision"])
                mlflow.log_metric(f"{safe_name}_recall",    class_metrics["recall"])

        # ── Save & log artifacts ───────────────────────────────────────────────

        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path    = os.path.join(models_dir, "xgboost_traffic.pkl")
        encoder_path  = os.path.join(models_dir, "label_encoder.pkl")
        features_path = os.path.join(models_dir, "feature_cols.pkl")

        joblib.dump(model,        model_path)
        joblib.dump(target_enc,   encoder_path)
        joblib.dump(feature_cols, features_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(features_path)
        mlflow.log_artifact(output_path)   # the preprocessed CSV

        # Also log the model natively via mlflow.sklearn for the model registry
        mlflow.sklearn.log_model(model, artifact_path="xgboost_model")

        print(f"\n[train] Model saved   : {model_path}")
        print(f"[train] Encoder saved : {encoder_path}")
        print(f"[train] Features saved: {features_path}")
        print(f"[train] MLflow run logged to: {MLFLOW_TRACKING_URI}")

        print(df[TARGET_ENC_COL].value_counts(normalize=True))