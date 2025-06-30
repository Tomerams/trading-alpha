# ───────────────────────────────────────────────────────────────
# src/models/model_prediction_trainer.py   2025-06-20  V4-fixε
# ───────────────────────────────────────────────────────────────
"""
Trainer – walk-forward, log-return targets.
• תיקון: מניעת ±inf בלוג-ריטרן + guard על nan/inf ב-validation.
"""
from __future__ import annotations
import logging, os, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Tuple

import numpy as np, torch, joblib
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from config.meta_data_config import META_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

# ── logging ──────────────────────────────────────────────────
log = logging.getLogger("model_prediction_trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ── config shortcuts ────────────────────────────────────────
SEQ_LEN = MODEL_TRAINER_PARAMS["seq_len"]
BATCH_SIZE = MODEL_TRAINER_PARAMS.get("batch_size", 256)
EPOCHS = MODEL_TRAINER_PARAMS.get("epochs", 30)
PATIENCE = MODEL_TRAINER_PARAMS.get("patience", 6)
DEFAULT_LR = MODEL_TRAINER_PARAMS.get("lr", 1e-3)
MODEL_TYPE = MODEL_TRAINER_PARAMS["model_type"].strip()
WEIGHT_DECAY = MODEL_TRAINER_PARAMS.get("weight_decay", 1e-2)
GLOBAL_KWARGS = MODEL_TRAINER_PARAMS.get("model_kwargs", {})
TARGET_SPEC: dict[str, dict[str, Any]] = MODEL_TRAINER_PARAMS.get(
    "model_kwargs_target_specific", {}
)
WINDOW_YEARS = MODEL_TRAINER_PARAMS.get("walkforward_years", 2)

EXPLICIT_CLASS_TARGETS: set[str] = set()  # (add if any)


# ── helpers ─────────────────────────────────────────────────
def _merge_hp(tgt):
    hp = GLOBAL_KWARGS.copy()
    hp.update(TARGET_SPEC.get(tgt, {}))
    return hp


def _is_cls(y, t):
    return t in EXPLICIT_CLASS_TARGETS or (
        np.issubdtype(y.dtype, int) and np.unique(y).max() <= 2
    )


def _class_w3(y):
    cnt = np.bincount(y.flatten().astype(int), minlength=3) + 1
    return torch.tensor(len(y) / cnt, dtype=torch.float32)


def _save(net, sc, feats, tick, tgt, hp):
    out = Path("files/models")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / f"{tick}_{tgt}.pt")
    joblib.dump(sc, out / f"{tick}_{tgt}_scaler.pkl")
    joblib.dump(feats, out / f"{tick}_{tgt}_features.pkl")
    (out / f"{tick}_{tgt}.json").write_text(json.dumps(hp, indent=2))


# ───────────────── data prep & walk-forward ─────────────────
def _log_return(s: np.ndarray) -> np.ndarray:
    """safe log-return: ln(1+Δ%), clip למניעת ‎-inf/+inf"""
    s = np.log1p(pd.Series(s).pct_change().clip(lower=-0.99)).to_numpy()
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare(req: UpdateIndicatorsData) -> Tuple[list, list]:
    import pandas as pd

    df = get_indicators_data(req).dropna().reset_index(drop=True)
    tcols = TRAIN_TARGETS_PARAMS["target_cols"]
    fcols = [c for c in df.columns if c not in set(tcols) | {"Date"}]

    # convert targets → log-return
    for c in tcols:
        df[c] = _log_return(df[c].values)

    X = sliding_window_view(df[fcols].values, (SEQ_LEN, len(fcols)))[:-1]
    X = X.reshape(X.shape[0], SEQ_LEN, -1)

    yrs_per_bar = 252
    win = WINDOW_YEARS * yrs_per_bar
    splits = [(i, i + win) for i in range(0, len(X) - win, win)]
    if not splits:
        splits = [(0, len(X))]

    y = df[tcols].iloc[SEQ_LEN:].values.astype(np.float32)
    return [(X[a:b], y[a:b]) for a, b in splits], fcols


# ───────────────────────── train core ───────────────────────
def _fit(X_tr, y_tr, X_va, y_va, feat_dim, out_dim, hp, tag):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = hp.pop("lr", DEFAULT_LR)
    net = get_model(feat_dim, MODEL_TYPE, out_dim, **hp).to(dev)
    loss_fn = (
        torch.nn.SmoothL1Loss(beta=0.002)
        if out_dim == 1
        else torch.nn.CrossEntropyLoss(weight=_class_w3(y_tr), label_smoothing=0.05)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va)), batch_size=BATCH_SIZE * 2
    )

    best_state, best_val, patience = None, float("inf"), 0
    for ep in range(1, EPOCHS + 1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=dev.type == "cuda"):
                l = loss_fn(net(xb.float()).view_as(yb), yb)
            if not torch.isfinite(l):
                continue  # skip inf batches
            scaler_amp.scale(l).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
        sch.step()

        # ----- validation -----
        net.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                vals.append(loss_fn(net(xb.float()).view_as(yb), yb).item())
        v = float(np.nan_to_num(np.mean(vals), nan=1e6, posinf=1e6, neginf=1e6))
        log.info("[%s] ep %02d val=%.5f", tag, ep, v)

        if v < best_val - 1e-4:
            best_val, best_state, patience = v, net.state_dict(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                log.info("[%s] early stop", tag)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val


# ───────────────────────── public API ───────────────────────
def _train_one(tgt, req, splits, fcols):
    res = []
    idx = TRAIN_TARGETS_PARAMS["target_cols"].index(tgt)
    for w, (X, y) in enumerate(splits, 1):
        split = int(0.8 * len(X))
        Xtr, Xva = X[:split], X[split:]
        ytr, yva = y[:split, [idx]], y[split:, [idx]]

        if ytr.shape[1] == 1:
            ytr += np.random.normal(0, 1e-4, ytr.shape)

        sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
        Xtr = sc.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
        Xva = sc.transform(Xva.reshape(-1, Xva.shape[-1])).reshape(Xva.shape)

        hp = _merge_hp(tgt)
        tag = f"{req.stock_ticker}·{tgt}·W{w}"
        net, val = _fit(
            Xtr, ytr, Xva, yva, len(fcols), 3 if _is_cls(ytr, tgt) else 1, hp, tag
        )

        if w == len(splits):
            _save(net, sc, fcols, req.stock_ticker.upper(), tgt, hp)
        res.append(val)
    return {"val_loss": float(np.mean(res))}


def train_single(req: UpdateIndicatorsData):
    tgt = req.model_type or TRAIN_TARGETS_PARAMS["default_target"]
    splits, fcols = _prepare(req)
    return _train_one(tgt, req, splits, fcols)


def train_all_base_models(req: UpdateIndicatorsData):
    splits, fcols = _prepare(req)
    out = {}
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        fut = {
            pool.submit(_train_one, t, req, splits, fcols): t
            for t in META_PARAMS["base_targets"]
        }
        for f, t in fut.items():
            try:
                out[t] = f.result()
            except Exception as e:
                log.exception("fail %s", t)
                out[t] = {"error": str(e)}
    return out


if __name__ == "__main__":
    req = UpdateIndicatorsData(
        stock_ticker="QQQ",
        start_date="2005-01-01",
        end_date=time.strftime("%Y-%m-%d"),
        indicators=[],
        model_type="ALL",
    )
    print(json.dumps(train_all_base_models(req), indent=2, ensure_ascii=False))
