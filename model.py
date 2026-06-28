"""
model.py
========
ML ensemble for the directional predictor.

Pipeline:
  * LightGBM gradient-boosted classifier on the engineered features.
  * Isotonic / Platt calibration layer (sklearn) so the output probability is
    *calibrated* — essential because we compare model prob vs market implied
    prob to compute EV. An over-confident model would bleed money.
  * Graceful fallback to a logistic-regression model, then to "rules only"
    when there isn't enough data or LightGBM isn't installed.

The class is deliberately small and serialisable so the same artifact is used
by the backtest, paper, and live paths.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from features import ML_FEATURE_COLUMNS

log = logging.getLogger("model")


@dataclass
class ModelArtifacts:
    booster: object = None          # lightgbm.Booster or sklearn estimator
    calibrator: object = None       # sklearn calibration model on raw scores
    kind: str = "rules_only"        # lightgbm | logistic | rules_only
    feature_columns: tuple = tuple(ML_FEATURE_COLUMNS)
    n_train: int = 0


class DirectionModel:
    """Predicts P(up) from a feature matrix."""

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("model", {})
        self.art = ModelArtifacts()

    # ---------------------------------------------------------------- training
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelArtifacts:
        """Train on X (n, n_features) with binary y (1=up, 0=down)."""
        n = len(y)
        min_rows = self.cfg.get("min_training_rows", 2000)
        if n < min_rows:
            log.warning("Only %d training rows (<%d). Using rules_only.", n, min_rows)
            self.art = ModelArtifacts(kind="rules_only", n_train=n)
            return self.art

        kind = self.cfg.get("type", "lightgbm")
        try:
            if kind == "lightgbm":
                self._train_lightgbm(X, y)
            else:
                self._train_logistic(X, y)
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            log.warning("Primary model (%s) failed: %s. Falling back to logistic.", kind, exc)
            try:
                self._train_logistic(X, y)
            except Exception as exc2:
                log.error("Logistic fallback failed: %s. rules_only.", exc2)
                self.art = ModelArtifacts(kind="rules_only", n_train=n)
        self.art.n_train = n
        return self.art

    def _split(self, X, y, frac=0.8):
        idx = int(len(y) * frac)            # time-ordered split (no shuffle / leakage)
        return X[:idx], y[:idx], X[idx:], y[idx:]

    def _train_lightgbm(self, X, y):
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression

        Xtr, ytr, Xval, yval = self._split(X, y)
        train_set = lgb.Dataset(Xtr, label=ytr)
        valid_set = lgb.Dataset(Xval, label=yval, reference=train_set)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 50,
            "verbose": -1,
        }
        booster = lgb.train(
            params, train_set, num_boost_round=600,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        raw_val = booster.predict(Xval)
        cal = IsotonicRegression(out_of_bounds="clip")
        # guard against degenerate single-class validation slices
        if len(np.unique(yval)) > 1:
            cal.fit(raw_val, yval)
        else:
            cal = None
        self.art = ModelArtifacts(booster=booster, calibrator=cal, kind="lightgbm")
        log.info("Trained LightGBM on %d rows (val logloss-monitored).", len(ytr))

    def _train_logistic(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        Xtr, ytr, Xval, yval = self._split(X, y)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        pipe.fit(Xtr, ytr)
        self.art = ModelArtifacts(booster=pipe, calibrator=None, kind="logistic")
        log.info("Trained logistic regression on %d rows.", len(ytr))

    # -------------------------------------------------------------- prediction
    def predict_proba_up(self, x_row: np.ndarray) -> Optional[float]:
        """Return calibrated P(up) for one feature row, or None if rules_only."""
        if self.art.kind == "rules_only" or self.art.booster is None:
            return None
        x = x_row.reshape(1, -1)
        try:
            if self.art.kind == "lightgbm":
                raw = float(self.art.booster.predict(x)[0])
                if self.art.calibrator is not None:
                    raw = float(self.art.calibrator.predict([raw])[0])
                return float(np.clip(raw, 0.001, 0.999))
            else:  # logistic pipeline
                return float(np.clip(self.art.booster.predict_proba(x)[0, 1], 0.001, 0.999))
        except Exception as exc:  # pragma: no cover
            log.debug("predict failed: %s", exc)
            return None

    # ------------------------------------------------------------------- io
    def save(self, model_path: str, calibrator_path: str):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        if self.art.kind == "lightgbm" and self.art.booster is not None:
            self.art.booster.save_model(model_path)
            with open(calibrator_path, "wb") as fh:
                pickle.dump({"calibrator": self.art.calibrator,
                             "kind": self.art.kind, "n_train": self.art.n_train}, fh)
        elif self.art.booster is not None:
            with open(model_path, "wb") as fh:
                pickle.dump(self.art.booster, fh)
            with open(calibrator_path, "wb") as fh:
                pickle.dump({"calibrator": None, "kind": self.art.kind,
                             "n_train": self.art.n_train}, fh)
        log.info("Saved %s model to %s", self.art.kind, model_path)

    def load(self, model_path: str, calibrator_path: str) -> bool:
        if not (os.path.exists(model_path) and os.path.exists(calibrator_path)):
            log.info("No model artifacts found; running rules_only.")
            self.art = ModelArtifacts(kind="rules_only")
            return False
        try:
            with open(calibrator_path, "rb") as fh:
                meta = pickle.load(fh)
            kind = meta.get("kind", "lightgbm")
            if kind == "lightgbm":
                import lightgbm as lgb
                booster = lgb.Booster(model_file=model_path)
                self.art = ModelArtifacts(booster=booster, calibrator=meta.get("calibrator"),
                                          kind="lightgbm", n_train=meta.get("n_train", 0))
            else:
                with open(model_path, "rb") as fh:
                    booster = pickle.load(fh)
                self.art = ModelArtifacts(booster=booster, calibrator=None,
                                          kind=kind, n_train=meta.get("n_train", 0))
            log.info("Loaded %s model (%d train rows).", self.art.kind, self.art.n_train)
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to load model (%s); rules_only.", exc)
            self.art = ModelArtifacts(kind="rules_only")
            return False
