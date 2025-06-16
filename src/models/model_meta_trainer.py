# ─────────────────────────────────────────────────────────────────────────
# models/model_meta_trainer.py      Rev-Pareto-Scalar   2025-06-16
# ─────────────────────────────────────────────────────────────────────────
"""
Meta-Model Trainer – “80-20” (גרסת Scalar)
──────────────────────────────────────────
• לוקחים **עמודה בודדת** (scalar) מכל אחד מ-11 המודלים-הבסיסיים:
  11 פיצ'רים →  מאמנים LightGBM-Classifier  על Rolling-TimeSeries-CV.

• שומרים חבילה self-contained:
    {
        "meta_model"     : sklearn Pipeline,
        "feature_columns": ["V_Target_Tomorrow", …],
        "metrics"        : {...},
        "base_targets"   : [...],
    }

תלות חיצונית יחידה:  lightgbm   (pip install lightgbm)
"""

from __future__ import annotations

# ── OPENMP settings – מונע קריסות lightgbm ב-macOS/WSL ────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"          # אל תפתח יותר מדי ת’רדים
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # דרוש בחלק מהסביבות

# ── stdlib / third-party ───────────────────────────────────────────────
import logging, traceback
from pathlib import Path
from typing import Dict, List

import joblib, numpy as np, pandas as pd, torch
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── project imports ────────────────────────────────────────────────────
from config.meta_data_config import META_PARAMS
from config.model_trainer_config import TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

# ─────────────────────────── logger ────────────────────────────────────
log = logging.getLogger("model_meta_trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ───────────────────────── constants ───────────────────────────────────
TICKER          = "QQQ"
BASE_TARGETS: List[str] = META_PARAMS["base_targets"]      # 11 targets
SEQ_LEN         = META_PARAMS.get("seq_len", 60)
MODEL_DIR       = Path("files/models")
META_PKL        = Path("files/datasets/meta_dataset_scalar.pkl")
META_MODEL_PATH = MODEL_DIR / "meta_action_model.pkl"

# ───────────────────────── helpers ─────────────────────────────────────
def _load_artifacts(target: str):
    """
    Loads checkpoint, scaler, feat-list and returns:
        model (TransformerTCN out=11),
        scaler, feats, idx_of_target
    """
    stem        = MODEL_DIR / f"{TICKER}_{target}"
    state_path  = stem.with_suffix(".pt")
    scaler_path = stem.parent / f"{stem.name}_scaler.pkl"
    feats_path  = stem.parent / f"{stem.name}_features.pkl"

    if not (state_path.exists() and scaler_path.exists() and feats_path.exists()):
        raise FileNotFoundError(f"Missing artifacts for target={target}")

    chkpt   = torch.load(state_path, map_location="cpu")
    out_dim = chkpt["net.head.weight"].shape[0]

    scaler  = joblib.load(scaler_path)
    feats   = joblib.load(feats_path)
    model   = get_model(len(feats), "TransformerTCN", out_dim)
    model.load_state_dict(chkpt)
    model.eval()

    tgt_cols = TRAIN_TARGETS_PARAMS["target_cols"]
    try:
        idx = tgt_cols.index(target)
    except ValueError:
        idx = BASE_TARGETS.index(target)  # fallback

    return model, scaler, feats, idx


def _predict_value(target: str, df: pd.DataFrame) -> np.ndarray:
    """Returns a single float prediction per row for given base-model target."""
    model, scaler, feats, idx = _load_artifacts(target)

    X = np.lib.stride_tricks.sliding_window_view(
        df[feats].values, (SEQ_LEN, len(feats))
    )[:-1]
    X = X.reshape(X.shape[0], SEQ_LEN, -1)
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()

    return preds[:, idx]  # (n,)


def _build_meta_dataset(req: UpdateIndicatorsData) -> pd.DataFrame:
    log.info("▶️  Generating meta_dataset (11 scalar preds)")
    df_raw = get_indicators_data(req).dropna().reset_index(drop=True)

    mats, cols = [], []
    for tgt in BASE_TARGETS:
        mats.append(_predict_value(tgt, df_raw))       # (n,)
        cols.append(f"V_{tgt}")

    meta_df = pd.DataFrame(np.vstack(mats).T, columns=cols)
    meta_df["Return_3d"] = df_raw["Target_3_Days"].iloc[SEQ_LEN:].values

    META_PKL.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_pickle(META_PKL)
    log.info("meta_dataset saved → %s  shape=%s", META_PKL, meta_df.shape)
    return meta_df


def _derive_action(y: np.ndarray) -> np.ndarray:
    q_hi, q_lo = np.quantile(y, [0.67, 0.33])
    return np.where(y >= q_hi, 2, np.where(y <= q_lo, 0, 1)).astype(int)


def _metrics(y_true, proba, pred) -> Dict:
    roc = roc_auc_score(y_true, proba, multi_class="ovr")
    return {
        "roc_auc": float(roc),
        "f1": float(f1_score(y_true, pred, average="weighted")),
        "cm": confusion_matrix(y_true, pred).tolist(),
    }

# ───────────────────────── core ────────────────────────────────────────
def train_meta_model(req: UpdateIndicatorsData):
    """FastAPI entry – builds dataset, trains LightGBM meta-model, saves bundle."""
    try:
        df = _build_meta_dataset(req)
        X  = df.filter(like="V_").values          # (n, 11)
        y  = _derive_action(df["Return_3d"].values)

        scaler = StandardScaler()
        lgbm   = LGBMClassifier(
            n_estimators      = 600,
            learning_rate     = 0.03,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            max_depth         = -1,
            objective         = "multiclass",
            num_class         = 3,
            class_weight      = "balanced",
            n_jobs            = 1,      # ← חוסך קריסות OpenMP
            random_state      = META_PARAMS.get("random_state", 42),
        )

        tscv   = TimeSeriesSplit(n_splits=5)
        scores = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            lgbm.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[va_idx], y[va_idx])],
                eval_metric="multi_logloss",
                callbacks=[lambda env: None],   # silence
            )
            proba = lgbm.predict_proba(X[va_idx])
            pred  = proba.argmax(1)
            s     = _metrics(y[va_idx], proba, pred)
            scores.append(s)
            log.info("Fold %d  ROC-AUC=%.3f  F1=%.3f", fold + 1, s["roc_auc"], s["f1"])

        pipe = Pipeline([("sc", scaler), ("lgbm", lgbm)]).fit(X, y)

        avg_roc = float(np.mean([s["roc_auc"] for s in scores]))
        avg_f1  = float(np.mean([s["f1"]     for s in scores]))
        bundle  = {
            "meta_model"     : pipe,
            "feature_columns": list(df.filter(like="V_").columns),
            "metrics"        : {"roc_auc": avg_roc, "f1": avg_f1},
            "base_targets"   : BASE_TARGETS,
        }

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, META_MODEL_PATH)
        log.info("✅ meta-model saved → %s  |  ROC-AUC=%.3f  F1=%.3f",
                 META_MODEL_PATH, avg_roc, avg_f1)
        return {"status": "success", "metrics": bundle["metrics"]}

    except Exception:
        log.error("Meta-training failed:\n%s", traceback.format_exc())
        raise


# ───────────────────────── backward-compat API ─────────────────────────
def train_meta_model_from_request(req: UpdateIndicatorsData):
    """Alias kept for existing routers."""
    return train_meta_model(req)
