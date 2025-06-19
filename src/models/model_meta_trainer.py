# ─────────────────────────────────────────────────────────────────────────
# models/model_meta_trainer.py      Rev-Pareto-OOF+Optuna   2025-06-19
# ─────────────────────────────────────────────────────────────────────────
"""
Meta-Model Trainer – Walk-Forward + Engineered Features + Optuna
────────────────────────────────────────────────────────────────
1. תחזיות OOF מכל אחד מ-11 המודלים → 11 ערכים.
2. פיצ'רים על תחזיות: מומנטום + SMA-5 → 33 תכונות סה"כ.
3. אימון LightGBM עם אופטונא.

תלות חיצונית:
    pip install lightgbm optuna
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging, traceback, json, warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib, numpy as np, pandas as pd, torch, optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.meta_data_config import META_PARAMS
from config.model_trainer_config import TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── logger ────────────────────────────────────
log = logging.getLogger("meta_trainer_action")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ───────────────────────── constants ───────────────────────────────────
TICKER          = "QQQ"
BASE_TARGETS: List[str] = META_PARAMS["base_targets"]
SEQ_LEN         = META_PARAMS.get("seq_len", 60)
MODEL_DIR       = Path("files/models")
META_PKL        = Path("files/datasets/meta_dataset_scalar.pkl")
META_MODEL_PATH = MODEL_DIR / "meta_action_model.pkl"

# ───────────────────────── helpers ─────────────────────────────────────
def _load_artifacts(target: str):
    stem        = MODEL_DIR / f"{TICKER}_{target}"
    state_path  = stem.with_suffix(".pt")
    scaler_path = stem.parent / f"{stem.name}_scaler.pkl"
    feats_path  = stem.parent / f"{stem.name}_features.pkl"
    hp_path     = stem.with_suffix(".json")

    if not (state_path.exists() and scaler_path.exists() and feats_path.exists()):
        raise FileNotFoundError(f"Missing artifacts for target={target}")

    chkpt   = torch.load(state_path, map_location="cpu")
    out_dim = chkpt["net.head.weight"].shape[0]

    if hp_path.exists():
        hyper = json.loads(hp_path.read_text())
    else:
        hidden_guess = chkpt["net.tcn.network.0.conv1.bias"].shape[0]
        hyper = {"hidden_size": hidden_guess}
    hyper.setdefault("num_layers", len([k for k in chkpt if k.startswith("net.tcn.network.")])//2)

    scaler = joblib.load(scaler_path)
    feats  = joblib.load(feats_path)
    model  = get_model(len(feats), "TransformerTCN", out_dim, **hyper)
    model.load_state_dict(chkpt, strict=False)
    model.eval()

    tgt_cols = TRAIN_TARGETS_PARAMS["target_cols"]
    idx = tgt_cols.index(target) if target in tgt_cols else BASE_TARGETS.index(target)
    return model, scaler, feats, idx

def _rolling_oof_pred(model: torch.nn.Module, X_windows: np.ndarray, idx: int | None = None) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor(X_windows, dtype=torch.float32)).cpu().numpy()
    if out.ndim == 1 or out.shape[1] == 1:
        return out.ravel()
    return out[:, idx]

def _predict_value(target: str, df: pd.DataFrame) -> np.ndarray:
    model, scaler, feats, idx = _load_artifacts(target)

    # שלב מניעת התרסקות: מסיר פיצ'רים שלא קיימים
    missing_feats   = [f for f in feats if f not in df.columns]

    if missing_feats:
        log.warning("⚠️  Missing features for %s: %s", target, missing_feats)

    all_feats = pd.DataFrame(index=df.index)

    for f in feats:
        if f in df.columns:
            all_feats[f] = df[f]
        else:
            all_feats[f] = 0.0  # אפשר גם np.nan אם תרצה ש־scaler יתתמודד לבד

    raw = all_feats.values
    win = np.lib.stride_tricks.sliding_window_view(raw, (SEQ_LEN, len(feats)))[:-1]
    win = win.reshape(win.shape[0], SEQ_LEN, -1)
    win = scaler.transform(win.reshape(-1, win.shape[-1])).reshape(win.shape)

    return _rolling_oof_pred(model, win, idx)

def _build_meta_dataset(req: UpdateIndicatorsData) -> pd.DataFrame:
    log.info("▶️  Generating meta_dataset (OOF, engineered features)")
    df_raw = get_indicators_data(req).dropna().reset_index(drop=True)

    mats, feat_names = [], []
    for tgt in BASE_TARGETS:
        vec = _predict_value(tgt, df_raw)
        mats.append(vec)
        feat_names.append(f"V_{tgt}")

    meta = pd.DataFrame(np.vstack(mats).T, columns=feat_names)

    for c in feat_names:
        meta[f"{c}_d1"]   = meta[c].diff()
        meta[f"{c}_sma5"] = meta[c].rolling(5).mean()

    meta.dropna(inplace=True)
    meta["Return_3d"] = df_raw["Target_3_Days"] \
        .iloc[SEQ_LEN + (len(df_raw) - len(meta)) :].values

    META_PKL.parent.mkdir(parents=True, exist_ok=True)
    meta.to_pickle(META_PKL)
    log.info("meta_dataset saved → %s  shape=%s", META_PKL, meta.shape)
    return meta

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

@lru_cache(maxsize=1)
def _best_lgbm_params(X: np.ndarray, y: np.ndarray) -> dict:
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate"    : trial.suggest_float("lr", 0.01, 0.2, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 10),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child", 10, 60),
            "objective"        : "multiclass",
            "num_class"        : 3,
            "n_jobs"           : 1,
        }
        cv = TimeSeriesSplit(n_splits=4)
        roc = []
        for tr, va in cv.split(X):
            clf = LGBMClassifier(**params)
            clf.fit(X[tr], y[tr])
            roc.append(roc_auc_score(y[va], clf.predict_proba(X[va]), multi_class="ovr"))
        return 1 - np.mean(roc)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    return study.best_params

# ───────────────────────── core ────────────────────────────────────────
def train_meta_model(req: UpdateIndicatorsData):
    try:
        df = _build_meta_dataset(req)
        X  = df.filter(like="V_").values
        y  = _derive_action(df["Return_3d"].values)

        best = _best_lgbm_params(X, y)
        log.info("⚙️  Best LGBM params: %s", best)
        lgbm = LGBMClassifier(**best)

        scaler = StandardScaler()
        tscv   = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            lgbm.fit(
                X[tr_idx], y[tr_idx],
                eval_set=[(X[va_idx], y[va_idx])],
                eval_metric="multi_logloss",
                callbacks=[lambda env: None],
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
        log.info("✅ saved → %s  |  ROC=%.3f  F1=%.3f", META_MODEL_PATH, avg_roc, avg_f1)
        return {"status": "success", "metrics": bundle["metrics"]}

    except Exception:
        log.error("Meta-training failed:\n%s", traceback.format_exc())
        raise

def train_meta_model_from_request(req: UpdateIndicatorsData):
    return train_meta_model(req)