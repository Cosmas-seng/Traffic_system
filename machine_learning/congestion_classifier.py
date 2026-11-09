
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import model_config as cfg


def label_ratio(ratio, low=cfg.LOW_RATIO, heavy=cfg.HEAVY_RATIO):
    r = np.asarray(ratio, dtype=float)
    return np.where(r >= low, "low", np.where(r >= heavy, "moderate", "heavy"))


class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """Baseline: the future class equals the class right now (computed from speed_ratio_now)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return label_ratio(X["speed_ratio_now"].to_numpy())


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Baseline: always predict the most common training class."""

    def fit(self, X, y):
        vals, counts = np.unique(np.asarray(y), return_counts=True)
        self.majority_ = vals[counts.argmax()]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)


def candidate_classifiers(fast=False):
    """name -> (estimator, needs_scaling). Trees do not need scaling."""
    n = 40 if fast else 200
    rs = cfg.RANDOM_STATE
    return {
        "Decision Tree": (DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced",
                                                 random_state=rs), False),
        "Random Forest": (RandomForestClassifier(n_estimators=n, min_samples_leaf=5, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=rs), False),
        # early_stopping=False: sklearn's automatic early stopping holds out a RANDOM slice of the rows,
        # which would mix future and past observations and break the chronological discipline.
        "Gradient Boosting": (HistGradientBoostingClassifier(max_iter=60 if fast else 200, learning_rate=0.06,
                                                             class_weight="balanced", early_stopping=False,
                                                             random_state=rs), False),
    }
