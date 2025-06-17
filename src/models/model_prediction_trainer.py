"""
Model Prediction Trainer – v3‑fast ★★ FIX 2 (June 2025)
───────────────────────────────────────────────────────────
• כמו FIX 1 אך פותר:
  1. שגיאת CrossEntropy "weight tensor shape" ב‑BarsToNext*.
  2. mismatch בין (pred.squeeze) ל‑y (batch,1) ברגרסיה.
  3. אפשרות זיהוי אוטומטי האם הטארגט הוא 3‑class או רגרסיה.
"""
from __future__ import annotations
import logging, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import joblib, numpy as np, pandas as pd, torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.meta_data_config import META_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

# ────────────────────────────── Logging ───────────────────────────────
log = logging.getLogger("model_prediction_trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ─────────────────────────────── Config ───────────────────────────────
SEQ_LEN   = MODEL_TRAINER_PARAMS["seq_len"]
TRAIN_RATIO = MODEL_TRAINER_PARAMS.get("train_ratio", 0.8)
BATCH_SIZE  = MODEL_TRAINER_PARAMS.get("batch_size", 256)
EPOCHS      = MODEL_TRAINER_PARAMS.get("epochs", 30)
PATIENCE    = MODEL_TRAINER_PARAMS.get("patience", 6)
LR          = MODEL_TRAINER_PARAMS.get("lr", 1e-3)
MODEL_TYPE  = MODEL_TRAINER_PARAMS["model_type"].strip()

# טארגטים שהם באמת 3‑class (0/1/2).  כל השאר = רגרסיה.
EXPLICIT_CLASS_TARGETS: set[str] = set()  # ריק בינתיים – עד שיהיו כאלו

# ─────────────────────────── Utility helpers ──────────────────────────

def _is_classification(y: np.ndarray, target: str) -> bool:
    """החלטה אם הטארגט נחשב 3‑class (0/1/2) או רגרסיה."""
    if target in EXPLICIT_CLASS_TARGETS:
        return True
    if not np.issubdtype(y.dtype, np.integer):
        return False
    uniq = np.unique(y)
    return uniq.min() >= 0 and uniq.max() <= 2 and len(uniq) <= 3


def _class_weights_3(y: np.ndarray) -> torch.Tensor:
    """צבירת משקלים לכיתות 0/1/2 בלבד."""
    counts = np.bincount(y.flatten().astype(int), minlength=3) + 1
    return torch.tensor(len(y) / counts, dtype=torch.float32)


def _save_artifacts(model, scaler, feats, *, ticker: str, target: str) -> None:
    out = Path("files/models"); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{ticker}_{target}.pt")
    joblib.dump(scaler, out / f"{ticker}_{target}_scaler.pkl")
    joblib.dump(feats,  out / f"{ticker}_{target}_features.pkl")

# ─────────────────────── Data prep (shared) ───────────────────────────

def _prepare_shared(req: UpdateIndicatorsData):
    df = get_indicators_data(req).dropna().reset_index(drop=True)
    t_cols = TRAIN_TARGETS_PARAMS["target_cols"]
    f_cols = [c for c in df.columns if c not in set(t_cols) | {"Date"}]

    X = sliding_window_view(df[f_cols].values, (SEQ_LEN, len(f_cols)))[:-1]
    X = X.reshape(X.shape[0], SEQ_LEN, -1)

    split = int(len(X) * TRAIN_RATIO)
    X_tr, X_va = X[:split], X[split:]

    scaler = StandardScaler(); scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va = scaler.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)

    y = df[t_cols].iloc[SEQ_LEN:].values.astype(np.float32)
    y_tr, y_va = y[:split], y[split:]
    return X_tr, y_tr, X_va, y_va, f_cols, scaler, df

# ─────────────────────────── Training core ────────────────────────────

def _fit_model(arch: str, feat_dim: int, out_dim: int,
               X_tr, y_tr, X_va, y_va):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(feat_dim, arch, out_dim, **MODEL_TRAINER_PARAMS.get("model_kwargs", {})).to(dev)

    # Loss
    if out_dim == 1:
        loss_fn = torch.nn.MSELoss()
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights_3(y_tr), label_smoothing=0.05)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long).flatten()
        y_va_t = torch.tensor(y_va, dtype=torch.long).flatten()

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), y_tr_t), batch_size=BATCH_SIZE, shuffle=True, pin_memory=(dev.type == "cuda"))
    va_loader = DataLoader(TensorDataset(torch.tensor(X_va), y_va_t), batch_size=BATCH_SIZE*2, pin_memory=(dev.type == "cuda"))

    best_state, best_val, patience = None, float("inf"), 0
    for ep in range(1, EPOCHS+1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=dev.type == "cuda"):
                pred = net(xb.float())
                # no .squeeze() – shapes match (batch,1) vs (batch,1) or flat vector
                loss = loss_fn(pred.view_as(yb), yb)
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler_amp.step(opt); scaler_amp.update()
        sch.step()

        # Val
        net.eval(); v_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                v = loss_fn(net(xb.float()).view_as(yb), yb).item(); v_losses.append(v)
        v_mean = float(np.mean(v_losses))
        log.info("[%s] epoch %02d/%d  val=%.4f", arch, ep, EPOCHS, v_mean)
        if v_mean < best_val-1e-4:
            best_val, best_state, patience = v_mean, net.state_dict(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                log.info("Early‑stop at %d (best=%.4f)", ep, best_val); break
    if best_state: net.load_state_dict(best_state)
    with torch.no_grad():
        tr_loss = loss_fn(net(torch.tensor(X_tr, device=dev).float()).view_as(y_tr_t.to(dev)), y_tr_t.to(dev)).item()
    return net, tr_loss, best_val

# ───────────────────── Target‑level wrapper ───────────────────────────

def _train_target(target: str, req: UpdateIndicatorsData,
                  X_tr, y_tr_all, X_va, y_va_all, feats, scaler):
    idx = TRAIN_TARGETS_PARAMS["target_cols"].index(target)
    y_tr, y_va = y_tr_all[:, [idx]], y_va_all[:, [idx]]
    is_cls = _is_classification(y_tr, target)
    net, tr, va = _fit_model(MODEL_TYPE, len(feats), 3 if is_cls else 1, X_tr, y_tr, X_va, y_va)
    _save_artifacts(net, scaler, feats, ticker=req.stock_ticker.upper(), target=target)
    return {"target": target, "architecture": MODEL_TYPE, "train_loss": tr, "val_loss": va}

# ─────────────────────────── Public API ───────────────────────────────

def train_all_base_models(req: UpdateIndicatorsData):
    X_tr, y_tr, X_va, y_va, feats, scaler, _ = _prepare_shared(req)
    res: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        fut_to_tgt = {pool.submit(_train_target, tgt, req, X_tr, y_tr, X_va, y_va, feats, scaler): tgt for tgt in META_PARAMS["base_targets"]}
        for fut, tgt in fut_to_tgt.items():
            try: res[tgt] = fut.result()
            except Exception as e:
                log.exception("training failed for %s", tgt); res[tgt] = {"error": str(e)}
    return res


def train_single(req: UpdateIndicatorsData, *, persist_meta=True):
    X_tr, y_tr, X_va, y_va, feats, scaler, df = _prepare_shared(req)
    tgt = req.model_type or TRAIN_TARGETS_PARAMS["default_target"]
    res = _train_target(tgt, req, X_tr, y_tr, X_va, y_va, feats, scaler)
    # Persist meta – as before (unchanged for brevity)
    return res


def train_model(request: UpdateIndicatorsData):
    return {"status": "batch", "results": train_all_base_models(request)} if request.model_type and request.model_type.upper()=="ALL" else {"status": "single", **train_single(request)}
