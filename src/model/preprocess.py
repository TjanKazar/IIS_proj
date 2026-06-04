"""
Traffic data preprocessing + training pipeline.

Preprocessing features:
  - Lag features        : prev_state, prev_volume, prev_speed, prev_occupancy
  - Time-gap feature    : minutes_since_last_reading  (key for irregular series)
  - Calendar features   : hour, day_of_week, month, is_weekend, is_night
  - Target encoding     : Stanje -> integer label
  - Direction encoding  : label-encoded
  - Road encoding       : label-encoded

Training:
  - XGBoost classifier
  - Time-based train/test split (no shuffle — respects temporal order)
  - Saves model + label encoder via joblib
"""

import os
import random
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# -- Load params from project root ---------------------------------------------
# Resolved relative to this script's location, not the working directory,
# so it works regardless of which directory you call uv run from.
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
params_path  = os.path.join(project_root, "params.yaml")

all_params   = yaml.safe_load(open(params_path))
params       = all_params["preprocess"]
train_params = all_params["train"]

INPUT_PATH         = params["input"]
OUTPUT_PATH        = params["output"]
RANDOM_STATE       = params["random_state"]
TARGET_COL         = params["target_col"]          # "Stanje"
TIMESTAMP_COL      = params["timestamp_col"]       # "Cas"
LOCATION_COL       = params["location_col"]        # "Lokacija"
DIRECTION_COL      = params["direction_col"]       # "Smer"
ROAD_COL           = params["road_col"]            # "Cesta"
TIME_DIFF_UNIT     = params["time_diff_unit"]      # "minutes"
LOCATION_KEY_COLS  = params["location_key_cols"]   # ["Lokacija", "Smer"]
CREATE_TIME_DIFF   = params["create_time_diff"]
CREATE_TIME_FEAT   = params["create_time_features"]
REMOVE_LEAKAGE     = params["remove_location_leakage"]
FILTER_NO_DATA     = params["filter_no_data"]
NO_DATA_LABEL      = params["no_data_label"]       # "Ni podatka"
NO_DATA_OUTPUT     = params["no_data_output"]

TEST_SIZE          = train_params["test_size"]     # fraction, e.g. 0.2
USE_TIME_SPLIT     = train_params["use_time_split"]

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
# 1. DatePreprocessor
# ==============================================================================
class DatePreprocessor(BaseEstimator, TransformerMixin):
    """
    Parse timestamp and optionally extract calendar features.

    Parameters
    ----------
    timestamp_col : str
        Name of the raw timestamp column (e.g. "Cas").
    create_time_features : bool
        If True, add hour, day_of_week, month, is_weekend, is_night columns.
    """

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


# ==============================================================================
# 2. LocationForwardFiller
# ==============================================================================
class LocationForwardFiller(BaseEstimator, TransformerMixin):
    """
    Forward-fill the Lokacija column within each road (Cesta) group,
    because the raw data leaves it blank for the second direction row.
    """

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


# ==============================================================================
# 3. TimeDiffTransformer
# ==============================================================================
class TimeDiffTransformer(BaseEstimator, TransformerMixin):
    """
    Add a `minutes_since_last_reading` column per location-direction group.

    A 5-min gap vs. a 10-hour gap carry very different information — this
    single feature is what lets XGBoost handle irregular time series.
    """

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


# ==============================================================================
# 4. LagFeatureTransformer
# ==============================================================================
class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create lag-1 features per location-direction group.
    Lagged: volume, speed, headway, occupancy, and the encoded target.
    """

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


# ==============================================================================
# 5. TargetEncoder
# ==============================================================================
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Label-encode the target column (Stanje) and store the class mapping.
    The original column is kept; a new `<target>_encoded` column is added.
    """

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


# ==============================================================================
# 6. CategoricalEncoder
# ==============================================================================
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Label-encode categorical string columns (Cesta, Smer, Lokacija).
    Encoders are fit on training data and stored for reuse at inference.
    """

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
            known    = set(le.classes_)
            X[col]   = X[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            X[f"{col}_encoded"] = le.transform(X[col])
        return X


# ==============================================================================
# 7. LeakageRemover
# ==============================================================================
class LeakageRemover(BaseEstimator, TransformerMixin):
    """Drop raw string columns after encoding to avoid identity leakage."""

    def __init__(self, cols_to_drop: list):
        self.cols_to_drop = cols_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X    = X.copy()
        drop = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=drop)


# ==============================================================================
# 8. NoDataFilter
# ==============================================================================
class NoDataFilter(BaseEstimator, TransformerMixin):
    """
    Remove rows where Stanje == no_data_label (e.g. "Ni podatka").

    These rows represent sensor downtime, not a traffic state. Filtering them
    out before feature engineering ensures lag and gap features are only
    computed over real traffic readings.

    Filtered rows are stored in .no_data_df_ for separate inspection.
    """

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
# Run: preprocess + train
# ==============================================================================
if __name__ == "__main__":

    # ── Preprocessing ──────────────────────────────────────────────────────────

    print(f"[preprocess] Reading  : {INPUT_PATH}")
    df = pd.read_csv(os.path.join(project_root, INPUT_PATH))
    print(f"[preprocess] Rows in  : {len(df)}")

    # Step 1: Forward-fill Lokacija
    filler = LocationForwardFiller(road_col=ROAD_COL, location_col=LOCATION_COL)
    df = filler.fit_transform(df)

    # Step 2: Parse timestamp + calendar features
    date_proc = DatePreprocessor(timestamp_col=TIMESTAMP_COL, create_time_features=CREATE_TIME_FEAT)
    df = date_proc.fit_transform(df)

    # Step 3: Filter no-data rows BEFORE any feature engineering
    if FILTER_NO_DATA:
        no_data_filter = NoDataFilter(target_col=TARGET_COL, no_data_label=NO_DATA_LABEL)
        df = no_data_filter.fit_transform(df)
        n_removed = len(no_data_filter.no_data_df_)
        print(f"[preprocess] Removed  : {n_removed} '{NO_DATA_LABEL}' rows "
              f"({100 * n_removed / (len(df) + n_removed):.1f}% of total)")
        no_data_path = os.path.join(project_root, NO_DATA_OUTPUT)
        os.makedirs(os.path.dirname(no_data_path), exist_ok=True)
        no_data_filter.no_data_df_.to_csv(no_data_path, index=False)
        print(f"[preprocess] No-data saved to : {no_data_path}")

    # Step 4: Label-encode target BEFORE lag features
    target_enc = TargetEncoder(target_col=TARGET_COL)
    df = target_enc.fit_transform(df)
    print(f"[preprocess] Target classes   : {target_enc.class_map_}")

    # Step 5: Time-gap feature
    if CREATE_TIME_DIFF:
        time_diff = TimeDiffTransformer(
            timestamp_col=TIMESTAMP_COL,
            location_key_cols=LOCATION_KEY_COLS,
            unit=TIME_DIFF_UNIT,
        )
        df = time_diff.fit_transform(df)

    # Step 6: Lag features
    lag_tf = LagFeatureTransformer(
        location_key_cols=LOCATION_KEY_COLS,
        timestamp_col=TIMESTAMP_COL,
        numeric_cols=[COL_VOLUME, COL_SPEED, COL_HEADWAY, COL_OCCUPANCY],
        target_col=TARGET_COL,
    )
    df = lag_tf.fit_transform(df)

    # Step 7: Encode categoricals
    cat_enc = CategoricalEncoder(cols=[ROAD_COL, DIRECTION_COL, LOCATION_COL])
    df = cat_enc.fit_transform(df)

    # Step 8: Drop raw string columns
    if REMOVE_LEAKAGE:
        leakage_remover = LeakageRemover(
            cols_to_drop=[ROAD_COL, DIRECTION_COL, LOCATION_COL, TARGET_COL]
        )
        df = leakage_remover.fit_transform(df)

    # Step 9: Drop raw timestamp
    df = df.drop(columns=[TIMESTAMP_COL], errors="ignore")

    # Save preprocessed CSV
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
        # Temporal split — no shuffle, preserves order
        split_idx = int(len(df) * (1 - TEST_SIZE))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"\n[train] Time-based split at index {split_idx}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"\n[train] Random stratified split")

    print(f"[train] Train size : {len(X_train)}")
    print(f"[train] Test size  : {len(X_test)}")

    # ── Train XGBoost ──────────────────────────────────────────────────────────

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[train] Test accuracy : {acc:.4f}")
    print("\n[train] Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[target_enc.class_map_[i] for i in sorted(target_enc.class_map_)],
    ))

    # ── Save model + encoder ───────────────────────────────────────────────────

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path   = os.path.join(models_dir, "xgboost_traffic.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    features_path = os.path.join(models_dir, "feature_cols.pkl")

    joblib.dump(model,        model_path)
    joblib.dump(target_enc,   encoder_path)
    joblib.dump(feature_cols, features_path)

    print(f"\n[train] Model saved   : {model_path}")
    print(f"[train] Encoder saved : {encoder_path}")
    print(f"[train] Features saved: {features_path}")
    print(df[TARGET_ENC_COL].value_counts(normalize=True))