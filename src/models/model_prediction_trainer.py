"""
Model Prediction Trainer – v3-fast (June 2025)

• מריץ מודל בודד (train_single) או אצווה (train_all_base_models) במהירות משופרת.
• יוצר רצפים פעם אחת, מבצע סקלינג פעם אחת ומשתף בין כל ה-targets.
• מאפשר הרצה מקבילית בטראדים (GPU-safe) או בתהליכים (CPU-safe).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.meta_data_config import META_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

log = logging.getLogger("model_prediction_trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


def _save_artifacts(
    model: torch.nn.Module,
    scaler: StandardScaler,
    feats: List[str],
    *,
    ticker: str,
    target: str,
) -> None:
    out = Path("files/models")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{ticker}_{target}.pt")
    joblib.dump(scaler, out / f"{ticker}_{target}_scaler.pkl")
    joblib.dump(feats, out / f"{ticker}_{target}_features.pkl")


def _prepare_shared_tensors(
    req: UpdateIndicatorsData,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    StandardScaler,
    pd.DataFrame,
]:
    df = get_indicators_data(req).dropna().reset_index(drop=True)
    target_cols = TRAIN_TARGETS_PARAMS["target_cols"]
    feature_cols = [c for c in df.columns if c not in set(target_cols) | {"Date"}]
    seq_len = MODEL_TRAINER_PARAMS["seq_len"]

    X_full = sliding_window_view(df[feature_cols].values, (seq_len, len(feature_cols)))[
        :-1
    ]
    X_full = X_full.reshape(X_full.shape[0], seq_len, -1)

    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full.reshape(-1, X_full.shape[-1])).reshape(
        X_full.shape
    )

    split_idx = int(len(X_full) * 0.8)
    X_tr, X_va = X_full[:split_idx], X_full[split_idx:]
    y_full = df[target_cols].iloc[seq_len:].values.astype(np.float32)
    y_tr, y_va = y_full[:split_idx], y_full[split_idx:]
    return X_tr, y_tr, X_va, y_va, feature_cols, scaler, df


def _fit_model(
    arch: str,
    feature_count: int,
    target_count: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> Tuple[torch.nn.Module, float, float]:
    model = get_model(
        feature_count,
        arch,
        target_count,
        **MODEL_TRAINER_PARAMS.get("model_kwargs", {}),
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    if torch.cuda.is_available():
        model.cuda()
    batch_size = MODEL_TRAINER_PARAMS.get("batch_size", 256)
    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
    )
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs = MODEL_TRAINER_PARAMS.get("epochs", 10)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            if torch.cuda.is_available():
                xb, yb = xb.cuda(), yb.cuda()
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = loss_fn(model(xb.float()), yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optim)
            scaler_amp.update()
        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(
                model(
                    torch.tensor(X_tr).float().cuda()
                    if torch.cuda.is_available()
                    else torch.tensor(X_tr).float()
                ),
                torch.tensor(y_tr).cuda()
                if torch.cuda.is_available()
                else torch.tensor(y_tr),
            ).item()
            val_loss = loss_fn(
                model(
                    torch.tensor(X_va).float().cuda()
                    if torch.cuda.is_available()
                    else torch.tensor(X_va).float()
                ),
                torch.tensor(y_va).cuda()
                if torch.cuda.is_available()
                else torch.tensor(y_va),
            ).item()
        log.info(
            "[%s] epoch %d/%d  train %.4f  val %.4f",
            arch,
            ep,
            epochs,
            tr_loss,
            val_loss,
        )
    return model, tr_loss, val_loss


def train_single(req: UpdateIndicatorsData, *, persist_meta: bool = True) -> Dict:
    t0 = time.time()
    arch = MODEL_TRAINER_PARAMS["model_type"].strip()
    ticker = req.stock_ticker.upper()
    X_tr, y_tr, X_va, y_va, feats, scaler, df = _prepare_shared_tensors(req)
    model, tr_loss, val_loss = _fit_model(
        arch, len(feats), y_tr.shape[1], X_tr, y_tr, X_va, y_va
    )
    target = req.model_type or TRAIN_TARGETS_PARAMS["default_target"]
    _save_artifacts(model, scaler, feats, ticker=ticker, target=target)

    if persist_meta:
        with torch.no_grad():
            preds = (
                model(
                    torch.tensor(
                        X_tr
                        if torch.cuda.is_available()
                        else np.concatenate([X_tr, X_va])
                    )
                    .float()
                    .cuda()
                    if torch.cuda.is_available()
                    else torch.tensor(np.concatenate([X_tr, X_va])).float()
                )
                .cpu()
                .numpy()
            )
        meta_df = pd.DataFrame(
            preds, columns=[f"Pred_{c}" for c in TRAIN_TARGETS_PARAMS["target_cols"]]
        )
        meta_df["Return_3d"] = df["Target_3_Days"].iloc[
            MODEL_TRAINER_PARAMS["seq_len"] :
        ]
        Path("files/datasets").mkdir(parents=True, exist_ok=True)
        meta_df.to_pickle("files/datasets/meta_dataset.pkl")
        log.info("meta_dataset.pkl saved  shape=%s", meta_df.shape)

    log.info("✅ %s finished in %.1fs", target, time.time() - t0)
    return dict(
        target=target,
        architecture=arch,
        train_loss=float(tr_loss),
        val_loss=float(val_loss),
    )


def _train_wrapper(
    target: str,
    req: UpdateIndicatorsData,
    shared_X_tr: np.ndarray,
    shared_y_tr: np.ndarray,
    shared_X_va: np.ndarray,
    shared_y_va: np.ndarray,
    feats: List[str],
    scaler: StandardScaler,
) -> Dict:
    local_req = req.model_copy(update={"model_type": target})
    arch = MODEL_TRAINER_PARAMS["model_type"].strip()
    model, tr_loss, val_loss = _fit_model(
        arch,
        len(feats),
        shared_y_tr.shape[1],
        shared_X_tr,
        shared_y_tr,
        shared_X_va,
        shared_y_va,
    )
    _save_artifacts(
        model, scaler, feats, ticker=req.stock_ticker.upper(), target=target
    )
    return dict(
        train_loss=float(tr_loss),
        val_loss=float(val_loss),
        architecture=arch,
        target=target,
    )


def train_all_base_models(req: UpdateIndicatorsData) -> Dict[str, Dict]:
    X_tr, y_tr, X_va, y_va, feats, scaler, _ = _prepare_shared_tensors(req)
    results = {}
    max_workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_wrapper, tgt, req, X_tr, y_tr, X_va, y_va, feats, scaler
            ): tgt
            for tgt in META_PARAMS["base_targets"]
        }
        for fut in futures:
            tgt = futures[fut]
            try:
                results[tgt] = fut.result()
            except Exception as e:
                log.exception("training failed for target=%s", tgt)
                results[tgt] = {"error": str(e)}
    return results


def train_model(request: UpdateIndicatorsData):
    if request.model_type and request.model_type.upper() == "ALL":
        return {"status": "batch", "results": train_all_base_models(request)}
    return {"status": "single", **train_single(request)}
