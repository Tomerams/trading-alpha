from __future__ import annotations
import json, logging, os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib, numpy as np, pandas as pd, torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.meta_data_config import META_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

log = logging.getLogger("trainer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

SEQ_LEN_DEFAULT = MODEL_TRAINER_PARAMS["seq_len"]
SEQ_LEN_MAP = MODEL_TRAINER_PARAMS["seq_len_map"]
EPOCHS = MODEL_TRAINER_PARAMS["epochs"]
BATCH = MODEL_TRAINER_PARAMS["batch_size"]
VAL_RATIO = MODEL_TRAINER_PARAMS["val_ratio"]
PATIENCE = MODEL_TRAINER_PARAMS["early_stopping_patience"]
WARMUP = MODEL_TRAINER_PARAMS["warmup_pct"]
WD = MODEL_TRAINER_PARAMS["weight_decay"]
GLOBAL_HP = MODEL_TRAINER_PARAMS["model_kwargs"]
MODEL_KIND = MODEL_TRAINER_PARAMS["model_type"]

def seq_len_for_target(t: str) -> int:
    if "Tomorrow" in t:
        return SEQ_LEN_MAP[1]
    m = re.search(r"(\d+)_Days", t)
    if m:
        d = int(m.group(1))
        return SEQ_LEN_MAP.get(d, SEQ_LEN_DEFAULT)
    return SEQ_LEN_DEFAULT

def make_log_return(a: np.ndarray) -> np.ndarray:
    s = np.log1p(pd.Series(a).pct_change().clip(lower=-0.99)).values
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

def prepare_data(req: UpdateIndicatorsData, tgt: str):
    df = get_indicators_data(req).dropna().reset_index(drop=True)
    df[tgt] = make_log_return(df[tgt].values)
    feat_cols = [c for c in df.columns if c not in TRAIN_TARGETS_PARAMS["target_cols"] + ["Date"]]
    win = seq_len_for_target(tgt)
    X = sliding_window_view(df[feat_cols].values, (win, len(feat_cols)))[:-1]
    X = X.reshape(X.shape[0], win, -1)
    y = df[tgt].iloc[win:].values.astype(np.float32).reshape(-1, 1)
    split = int((1 - VAL_RATIO) * len(X))
    return (X[:split], y[:split]), (X[split:], y[split:]), feat_cols, win

def fit_model(Xtr, ytr, Xva, yva, feat_dim, hp):
    lr = hp.pop("lr", 1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_model(feat_dim, MODEL_KIND, 1, **hp).to(device)
    loss_fn = torch.nn.SmoothL1Loss(beta=1.0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD)
    steps_per_epoch = int(np.ceil(len(Xtr) / BATCH))
    total_steps = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps, pct_start=WARMUP)
    best_state, best_val, patience = None, float("inf"), 0
    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(yva)), batch_size=BATCH * 2)
    for ep in range(1, EPOCHS + 1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = loss_fn(net(xb.float()).view_as(yb), yb)
            l.backward()
            opt.step()
            scheduler.step()
        net.eval()
        with torch.no_grad():
            v = np.mean([loss_fn(net(xb.to(device).float()).view_as(yb.to(device)), yb.to(device)).item() for xb, yb in va_loader])
        if v < best_val - 1e-4:
            best_val, best_state, patience = v, net.state_dict(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val

def save_artifacts(net, scaler, feats, tick, tgt, hp):
    root = Path("files/models")
    root.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), root / f"{tick}_{tgt}.pt")
    joblib.dump(scaler, root / f"{tick}_{tgt}_scaler.pkl")
    joblib.dump(feats, root / f"{tick}_{tgt}_features.pkl")
    (root / f"{tick}_{tgt}.json").write_text(json.dumps(hp))

def train_target(tgt: str, req: UpdateIndicatorsData):
    (Xtr, ytr), (Xva, yva), feat_cols, win = prepare_data(req, tgt)
    scaler = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    Xtr = scaler.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xva = scaler.transform(Xva.reshape(-1, Xva.shape[-1])).reshape(Xva.shape)
    hp = GLOBAL_HP.copy()
    net, val = fit_model(Xtr, ytr, Xva, yva, len(feat_cols), hp)
    save_artifacts(net, scaler, feat_cols, req.stock_ticker.upper(), tgt, hp)
    return {"val_loss": float(val)}

def train_all(req: UpdateIndicatorsData):
    out = {}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        fut = {pool.submit(train_target, t, req): t for t in TRAIN_TARGETS_PARAMS["target_cols"]}
        for f in fut:
            t = fut[f]
            try:
                out[t] = f.result()
            except Exception as e:
                log.exception("fail %s", t)
                out[t] = {"error": str(e)}
    return out

if __name__ == "__main__":
    r = UpdateIndicatorsData(stock_ticker="QQQ", start_date="2005-01-01", end_date=time.strftime("%Y-%m-%d"), indicators=[], model_type="ALL")
    print(json.dumps(train_all(r), indent=2, ensure_ascii=False))