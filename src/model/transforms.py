"""
transforms.py
─────────────
Sklearn-compatible preprocessing transformers for the traffic pipeline.
All classes follow the BaseEstimator / TransformerMixin interface so they
can be dropped into a sklearn Pipeline if needed.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


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
        median_gap   = X[feat_name].median()
        X[feat_name] = X[feat_name].fillna(median_gap)
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
                lag_name = (
                    "prev_"
                    + col.split("[")[0].strip().lower().replace(" ", "_").replace(".", "")
                )
                X[lag_name] = (
                    X.groupby(self.location_key_cols, sort=False)[col]
                    .transform(lambda s: s.shift(1))
                )
        enc_target = f"{self.target_col}_encoded"
        if enc_target in X.columns:
            X["prev_state"] = (
                X.groupby(self.location_key_cols, sort=False)[enc_target]
                .transform(lambda s: s.shift(1))
            )
        lag_cols     = [c for c in X.columns if c.startswith("prev_")]
        X[lag_cols]  = X[lag_cols].fillna(0)
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