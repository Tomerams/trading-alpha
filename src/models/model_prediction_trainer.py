"""
Model Prediction Trainer (v2‑pareto – June 2025)
───────────────────────────────────────────────
* Trains a single DL **base model** or, via `train_all_base_models`, a full batch of targets.
* Distinguishes `arch` (network architecture → sent to `get_model`) from `target` (label / file‑stem).
* Saves `{TICKER}_{TARGET}.pt | _scaler.pkl | _features.pkl` under `files/models/`.
* During batch mode we skip rebuilding `meta_dataset.pkl` each iteration.
"""
from __future__ import annotations

import logging, time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config.meta_data_config import META_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

# ────────────────────────── logging
log = logging.getLogger("model_prediction_trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

def _save_artifacts(model: torch.nn.Module, scaler: StandardScaler, feats: List[str], *, ticker: str, target: str) -> None:
    out = Path("files/models"); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),          out / f"{ticker}_{target}.pt")
    joblib.dump(scaler,                    out / f"{ticker}_{target}_scaler.pkl")
    joblib.dump(feats,                     out / f"{ticker}_{target}_features.pkl")
    log.info("Artifacts saved → %s/*", out)


def _make_sequences(df: pd.DataFrame, feats: List[str], targets: List[str], L: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df) - L
    X = np.stack([df[feats].iloc[i:i+L].values for i in range(n)])
    y = df[targets].iloc[L:].values.astype(np.float32)
    return X, y

# ═══════════════════════════════════════════════════════════════════════
# single‑target trainer
# ═══════════════════════════════════════════════════════════════════════

def train_single(req: UpdateIndicatorsData, *, persist_meta: bool = True):
    t0 = time.time()
    ticker = req.stock_ticker.upper()

    # split target vs. architecture  ------------------------------------
    target = (req.model_type or TRAIN_TARGETS_PARAMS["default_target"]).strip()
    arch   = MODEL_TRAINER_PARAMS["model_type"].strip()       # e.g. "TransformerTCN"
    seq_len = MODEL_TRAINER_PARAMS["seq_len"]

    # 1) data -----------------------------------------------------------
    df = get_indicators_data(req).dropna().reset_index(drop=True)
    target_cols  = TRAIN_TARGETS_PARAMS["target_cols"]
    feature_cols = [c for c in df.columns if c not in set(target_cols) | {"Date"}]

    # 2) split ----------------------------------------------------------
    split_idx = int(len(df) * 0.8)
    tr_df, va_df = df.iloc[:split_idx], df.iloc[split_idx:]
    X_tr, y_tr = _make_sequences(tr_df, feature_cols, target_cols, seq_len)
    X_va, y_va = _make_sequences(va_df, feature_cols, target_cols, seq_len)

    # 3) scale ----------------------------------------------------------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va = scaler.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)

    # 4) build model ----------------------------------------------------
    model = get_model(len(feature_cols), arch, len(target_cols), **MODEL_TRAINER_PARAMS.get("model_kwargs", {}))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()

    epochs = MODEL_TRAINER_PARAMS.get("epochs", 10)
    for ep in range(1, epochs + 1):
        model.train(); optim.zero_grad()
        loss_tr = loss_fn(model(torch.tensor(X_tr, dtype=torch.float32)), torch.tensor(y_tr))
        loss_tr.backward(); optim.step()
        with torch.no_grad():
            loss_va = loss_fn(model(torch.tensor(X_va, dtype=torch.float32)), torch.tensor(y_va))
        log.info("[%s→%s] Epoch %d/%d  Train %.4f  Val %.4f", arch, target, ep, epochs, loss_tr.item(), loss_va.item())

    # 5) save -----------------------------------------------------------
    _save_artifacts(model, scaler, feature_cols, ticker=ticker, target=target)

    # 6) optional meta‑dataset -----------------------------------------
    if persist_meta:
        X_full, _ = _make_sequences(df, feature_cols, target_cols, seq_len)
        X_full = scaler.transform(X_full.reshape(-1, X_full.shape[-1])).reshape(X_full.shape)
        with torch.no_grad():
            preds_full = model(torch.tensor(X_full, dtype=torch.float32)).cpu().numpy()
        meta_df = pd.DataFrame(preds_full, columns=[f"Pred_{c}" for c in target_cols])
        meta_df["Return_3d"] = df["Target_3_Days"].iloc[seq_len:].astype(float)
        Path("files/datasets").mkdir(parents=True, exist_ok=True)
        meta_df.to_pickle("files/datasets/meta_dataset.pkl")
        log.info("Meta‑dataset saved → files/datasets/meta_dataset.pkl  shape=%s", meta_df.shape)

    log.info("✅ %s→%s finished in %.1fs", arch, target, time.time() - t0)
    return {
        "target": target,
        "architecture": arch,
        "train_loss": float(loss_tr.item()),
        "val_loss": float(loss_va.item()),
    }

# ═══════════════════════════════════════════════════════════════════════
# batch (ALL targets)
# ═══════════════════════════════════════════════════════════════════════

def train_all_base_models(req: UpdateIndicatorsData):
    results = {}
    for tgt in META_PARAMS["base_targets"]:
        clone = req.model_copy(update={"model_type": tgt})
        results[tgt] = train_single(clone, persist_meta=False)
    return results

# ═══════════════════════════════════════════════════════════════════════
# FastAPI wrapper
# ═══════════════════════════════════════════════════════════════════════


def train_model(request: UpdateIndicatorsData):
    if request.model_type and request.model_type.upper() == "ALL":
        return {"status": "batch", "results": train_all_base_models(request)}
    return {"status": "single", **train_single(request)}
