
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import model_config as cfg


class PersistenceRegressor(BaseEstimator, RegressorMixin):
    """Baseline: the speed at the horizon equals the speed now."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["speed_now"].to_numpy(dtype=float)


class SeasonalMeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline: average TRAINING speed for the same road, local hour and weekday/weekend."""
    _keys = ["segment_id", "target_hour", "target_is_weekend"]

    def fit(self, X, y):
        d = X[self._keys].copy()
        d["y"] = np.asarray(y, dtype=float)
        self.table_ = d.groupby(self._keys)["y"].mean()
        self.seg_ = d.groupby("segment_id")["y"].mean()
        self.global_ = float(d["y"].mean())
        return self

    def predict(self, X):
        idx = pd.MultiIndex.from_frame(X[self._keys])
        out = self.table_.reindex(idx).to_numpy()
        fallback = X["segment_id"].map(
            self.seg_).fillna(self.global_).to_numpy()
        return np.where(np.isnan(out), fallback, out)


def candidate_regressors(fast=False):
    """name -> (estimator, needs_scaling)."""
    n = 40 if fast else 200
    rs = cfg.RANDOM_STATE
    return {
        "Linear Regression": (LinearRegression(), True),
        "Random Forest": (RandomForestRegressor(n_estimators=n, min_samples_leaf=5, n_jobs=-1, random_state=rs), False),
        # early_stopping=False for the same reason as the classifier: no random hold-out inside time series.
        "Gradient Boosting": (HistGradientBoostingRegressor(max_iter=60 if fast else 200, learning_rate=0.06,
                                                            early_stopping=False, random_state=rs), False),
    }


def build_pipeline(estimator, numeric_cols, scale=False):
    """Preprocessing + model in ONE object, so training and serving cannot drift apart."""
    num = Pipeline([("impute", SimpleImputer(strategy="median"))] +
                   ([("scale", StandardScaler())] if scale else []))
    pre = ColumnTransformer([("num", num, list(numeric_cols)),
                             ("seg", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["segment_id"])])
    return Pipeline([("pre", pre), ("model", estimator)])
