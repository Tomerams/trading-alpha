# ─────────────────────────────────────────────────────────────────────────
# models/model_meta_trainer.py   –  refactor June‑2025 (v3‑multiclass‑fix)
# ─────────────────────────────────────────────────────────────────────────
"""
Meta‑learner trainer
====================
• Loads **meta_dataset.pkl** (created by model_prediction_trainer)
• Derives `Action` labels on‑the‑fly (BUY / HOLD / SELL *or* binary)  
  via quantiles, unless the column already exists.
• Trains a stack‑of‑models pipeline:
      – Base learners   : LightGBM (optionally XGB / RF ‑ yet to add)
      – Stacking layer  : Logistic‑Regression (weighted, L2)
• Computes validation metrics that work for both **binary** and **multi‑class**
    (ROC‑AUC, weighted‑F1, confusion‑matrix).
• Persists the full pipeline + metrics under ***META_PARAMS['meta_model_path']***

> **Backward‑compat NOTE**  
> Added `load_meta_model()` so existing code (e.g. *backtester.py*) can still
> `from models.model_meta_trainer import load_meta_model` without changes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config.meta_data_config import META_PARAMS
from routers.routers_entities import UpdateIndicatorsData

# ─────────────────────────────────────────────────────────────────────────
# logging setup
# -----------------------------------------------------------------------
log = logging.getLogger("model_meta_trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────
# paths / constants
# -----------------------------------------------------------------------
META_PKL = Path("files/datasets/meta_dataset.pkl")

# ─────────────────────────────────────────────────────────────────────────
# helpers – data loading & label engineering
# -----------------------------------------------------------------------

def _load_dataset() -> pd.DataFrame:
    if not META_PKL.exists():
        raise FileNotFoundError(
            "meta_dataset.pkl missing – run /api/train first to create it"
        )
    df = pd.read_pickle(META_PKL)
    log.info("Loaded meta‑dataset: %s  shape=%s", META_PKL, df.shape)
    return df


def _derive_action(df: pd.DataFrame) -> pd.Series:
    """Return BUY/HOLD/SELL(0/1/2) or binary BUY(1)/REST(0)."""
    if META_PARAMS.get("binary_labels", False):
        return (df["Return_3d"] > 0).astype(int)

    thr_buy, thr_sell = df["Return_3d"].quantile([0.67, 0.33])
    return df["Return_3d"].apply(
        lambda r: 2 if r >= thr_buy else (0 if r <= thr_sell else 1)
    ).astype(int)

# ─────────────────────────────────────────────────────────────────────────
# helpers – models
# -----------------------------------------------------------------------

def _train_lgbm(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
) -> LGBMClassifier:
    params: Dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": META_PARAMS.get("random_state", 42),
        **META_PARAMS.get("lgbm_params", {}),
    }
    model = LGBMClassifier(**params).fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        callbacks=[
            early_stopping(50, verbose=False),
            log_evaluation(100),
        ],
    )
    return model


def _stack_preds(base: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Stack probability column(s) of each base learner."""
    stacks: List[np.ndarray] = []
    for name, mdl in base.items():
        proba = mdl.predict_proba(X)
        if proba.shape[1] == 2:
            stacks.append(proba[:, 1:2])  # keep 2‑D shape (N,1)
        else:
            stacks.append(proba)  # (N, C)
    return np.concatenate(stacks, axis=1)

# ─────────────────────────────────────────────────────────────────────────
# helpers – evaluation
# -----------------------------------------------------------------------

def _eval_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute ROC‑AUC (binary or multiclass), weighted‑F1, CM."""
    if y_prob.shape[1] == 2:  # binary
        roc = roc_auc_score(y_true, y_prob[:, 1])
    else:  # multi‑class
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    return {
        "roc_auc": float(roc),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }

# ─────────────────────────────────────────────────────────────────────────
# main trainer
# -----------------------------------------------------------------------

def _train_meta_model(req: UpdateIndicatorsData) -> Dict[str, Any]:
    log.info("▶️  Meta‑training start for %s", req.stock_ticker)

    df = _load_dataset()
    if "Action" not in df.columns:
        log.warning("Action column was missing – derived from Return_3d")
        df["Action"] = _derive_action(df)

    feat_cols = [c for c in df.columns if c.startswith("Pred_")]
    X, y = df[feat_cols].to_numpy(dtype=np.float32), df["Action"].to_numpy()

    Xtr, Xva, ytr, yva = train_test_split(
        X,
        y,
        test_size=META_PARAMS.get("val_size", 0.2),
        stratify=y,
        shuffle=True,
        random_state=META_PARAMS.get("random_state", 42),
    )

    log.info(
        "Split done  train=%s  val=%s  classes=%s",
        Xtr.shape,
        Xva.shape,
        np.unique(y, return_counts=True),
    )

    # ── Train base learners ────────────────────────────────────────────
    base: Dict[str, Any] = {
        "lgbm": _train_lgbm(Xtr, ytr, Xva, yva),
    }

    # ── Train stacking (meta) layer ────────────────────────────────────
    meta_Xtr = _stack_preds(base, Xtr)
    meta_Xva = _stack_preds(base, Xva)

    meta = LogisticRegression(
        C=META_PARAMS.get("meta_C", 1.0),
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    ).fit(meta_Xtr, ytr)

    # ── Evaluate ───────────────────────────────────────────────────────
    meta_prob = meta.predict_proba(meta_Xva)
    meta_pred = meta_prob.argmax(axis=1)
    metrics = _eval_metrics(yva, meta_prob, meta_pred)
    log.info("Validation ROC‑AUC=%.4f  F1=%.4f", metrics["roc_auc"], metrics["f1"])

    # ── Persist all artefacts ─────────────────────────────────────────
    model_path = Path(META_PARAMS["meta_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"base": base, "meta": meta, "metrics": metrics}, model_path)
    log.info("✅ meta‑model saved → %s", model_path)

    # ── JSON‑ready response ───────────────────────────────────────────
    return {
        "status": "success",
        "model_path": str(model_path),
        "metrics": metrics,
    }

# -----------------------------------------------------------------------
# FastAPI wrapper
# -----------------------------------------------------------------------

def train_meta_model_from_request(req: UpdateIndicatorsData):
    """Endpoint entry‑point (used in routers)."""
    return _train_meta_model(req)

# -----------------------------------------------------------------------
# backward‑compat utility: load_meta_model
# -----------------------------------------------------------------------

def load_meta_model(path: str | Path | None = None):
    """Load the persisted meta‑model bundle for inference / backtesting."""
    if path is None:
        path = META_PARAMS["meta_model_path"]
    bundle = joblib.load(path)
    return bundle
