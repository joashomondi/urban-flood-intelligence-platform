"""
modeling.py
===========
Flood Susceptibility Modelling engine.

Trains and evaluates interpretable classifiers that predict per-cell flood
occurrence from the engineered indicators. Emphasis is on **operational
usefulness and interpretability**, not model exotica:

    * Logistic Regression  - transparent, calibrated baseline
    * Random Forest        - non-linear, ships feature importances
    * XGBoost              - optional gradient-boosted challenger

Outputs: fitted model, hold-out metrics, ROC/PR points, confusion matrix,
feature importances and a persisted model artefact (joblib).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import utils
from .feature_engineering import INDICATOR_NAMES
from .utils import MODELS_DIR, get_logger

log = get_logger("ufip.model")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    import pickle
    _HAS_JOBLIB = False


FEATURE_COLUMNS: List[str] = INDICATOR_NAMES


def _build_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       class_weight="balanced",
                                       random_state=utils.RANDOM_SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=8,
            class_weight="balanced", n_jobs=-1,
            random_state=utils.RANDOM_SEED),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.2,
            eval_metric="logloss", n_jobs=-1,
            random_state=utils.RANDOM_SEED)
    return models


def _feature_importance(name: str, model, feature_names: List[str]) -> Dict[str, float]:
    """Extract a comparable feature-importance mapping across model types."""
    if name == "logistic_regression":
        clf = model.named_steps["clf"]
        imp = np.abs(clf.coef_[0])
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    else:
        imp = np.ones(len(feature_names))
    imp = imp / (imp.sum() + 1e-12)
    return {f: float(v) for f, v in zip(feature_names, imp)}


def train_and_evaluate(table: pd.DataFrame,
                       feature_columns: List[str] | None = None,
                       test_size: float = 0.25) -> dict:
    """Train every model, pick the best by ROC-AUC, and return full diagnostics."""
    log.info("=== STEP 4: FLOOD RISK MODELLING ===")
    feats = feature_columns or FEATURE_COLUMNS
    X = table[feats].to_numpy()
    y = table["flood_label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y,
        random_state=utils.RANDOM_SEED)
    log.info("Train=%d  Test=%d  positive rate=%.1f%%",
             len(y_tr), len(y_te), 100 * y.mean())

    results: Dict[str, dict] = {}
    trained: Dict[str, object] = {}
    for name, model in _build_models().items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_te, proba)
        fpr, tpr, _ = roc_curve(y_te, proba)
        prec_c, rec_c, _ = precision_recall_curve(y_te, proba)
        metrics = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "roc_auc": float(auc),
            "avg_precision": float(average_precision_score(y_te, proba)),
        }
        results[name] = {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr": {"precision": prec_c.tolist(), "recall": rec_c.tolist()},
            "feature_importance": _feature_importance(name, model, feats),
            "classification_report": classification_report(
                y_te, pred, output_dict=True, zero_division=0),
        }
        trained[name] = model
        log.info("  %-20s AUC=%.3f  F1=%.3f  Acc=%.3f",
                 name, auc, metrics["f1"], metrics["accuracy"])

    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    log.info("Best model: %s (AUC=%.3f)",
             best_name, results[best_name]["metrics"]["roc_auc"])

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": trained[best_name],
        "trained_models": trained,
        "feature_columns": feats,
        "test_data": {"X_test": X_te, "y_test": y_te},
    }


def save_model(model, name: str, feature_columns: List[str],
               path: Path | None = None) -> Path:
    """Persist a fitted model + its feature schema for the Streamlit app."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (MODELS_DIR / "flood_model.joblib")
    payload = {"model": model, "name": name, "feature_columns": feature_columns}
    if _HAS_JOBLIB:
        joblib.dump(payload, path)
    else:  # pragma: no cover
        path = path.with_suffix(".pkl")
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
    log.info("Saved model artefact -> %s", path)
    return path


def load_model(path: Path | None = None) -> dict:
    """Load a persisted model payload."""
    path = path or (MODELS_DIR / "flood_model.joblib")
    if _HAS_JOBLIB:
        return joblib.load(path)
    import pickle  # pragma: no cover
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict_surface(model, indicators: Dict[str, np.ndarray],
                    feature_columns: List[str]) -> np.ndarray:
    """Apply a fitted model to the full raster stack -> probability surface."""
    h, w = indicators[feature_columns[0]].shape
    X = np.column_stack([indicators[c].ravel() for c in feature_columns])
    proba = model.predict_proba(X)[:, 1]
    return proba.reshape(h, w).astype("float32")
