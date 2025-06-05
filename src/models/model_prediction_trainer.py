# ──────────────────────────────────────────────────────────────────────────────
# models/model_prediction_trainer.py  –  FULL FILE (refactored June 2025)
# ──────────────────────────────────────────────────────────────────────────────
"""
אימון מודל‑הבסיס + שמירת scaler/model/features _וגם_ יצירת
meta_dataset.pkl (תחזיות‑בסיס + action_label) – אפשרות A.

• train_single(request_data) – פונקציה עיקרית המופעלת ע״י /api/train
• קובצי‑יציאה:
    files/models/{TICKER}_{TYPE}.pt              – state_dict
    files/models/{TICKER}_{TYPE}_scaler.pkl      – StandardScaler
    files/models/{TICKER}_{TYPE}_features.pkl    – List[str]
    files/datasets/meta_dataset.pkl              – (X_meta, y_meta)
"""

from __future__ import annotations
import logging, time, joblib, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model  # ← removed missing import
from routers.routers_entities import UpdateIndicatorsData
from data.action_labels import make_action_label_quantile

# ────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("model_prediction_trainer")
logger.setLevel(logging.INFO)

# ────────────────────────────────────────────────────────────────────────
# helper 0 – save artifacts (local implementation)
# ────────────────────────────────────────────────────────────────────────

def save_model_artifacts(model: torch.nn.Module,
                         scaler: StandardScaler,
                         feature_cols: list[str],
                         ticker: str,
                         model_type: str) -> None:
    """Save model state_dict + scaler + features to files/models/"""
    base = Path("files/models")
    base.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base / f"{ticker}_{model_type}.pt")
    joblib.dump(scaler,            base / f"{ticker}_{model_type}_scaler.pkl")
    joblib.dump(feature_cols,      base / f"{ticker}_{model_type}_features.pkl")
    logger.info("Artifacts saved: %s, %s, %s",
                base / f"{ticker}_{model_type}.pt",
                base / f"{ticker}_{model_type}_scaler.pkl",
                base / f"{ticker}_{model_type}_features.pkl")

# ────────────────────────────────────────────────────────────────────────
# helper 1 – build sequences (Torch expects 3‑D tensor)
# ────────────────────────────────────────────────────────────────────────

def create_sequences(df: pd.DataFrame,
                     feature_cols: list[str],
                     target_cols: list[str],
                     seq_len: int):
    """Return: X_seq (N,L,F), y (N,T)"""
    n = len(df) - seq_len
    X = np.stack([df[feature_cols].iloc[i:i+seq_len].values for i in range(n)])
    y = df[target_cols].iloc[seq_len:].values.astype(np.float32)
    return X, y

# ────────────────────────────────────────────────────────────────────────
# main entry
# ────────────────────────────────────────────────────────────────────────

def train_single(request_data: UpdateIndicatorsData):
    t0 = time.time()
    ticker   = request_data.stock_ticker
    model_ty = MODEL_TRAINER_PARAMS["model_type"]
    seq_len  = MODEL_TRAINER_PARAMS["seq_len"]

    # ── 1) load data ───────────────────────────────────────────────────
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    df["action_label"] = make_action_label_quantile(df, "Target_3_Days", q=0.20)

    target_cols   = TRAIN_TARGETS_PARAMS["target_cols"]
    non_features  = set(target_cols) | {"action_label", "Date"}
    feature_cols  = [c for c in df.columns if c not in non_features]

    # ── 2) train/val split (chronological) ─────────────────────────────
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    val_df    = df.iloc[split_idx:].reset_index(drop=True)

    X_tr, y_tr = create_sequences(train_df, feature_cols, target_cols, seq_len)
    X_va, y_va = create_sequences(val_df,   feature_cols, target_cols, seq_len)

    # ── 3) scale features ──────────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr     = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va     = scaler.transform( X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)

    # ── 4) build and train model ───────────────────────────────────────
    model = get_model(
                input_size=len(feature_cols),
                model_type=model_ty,
                output_size=len(target_cols),
                **MODEL_TRAINER_PARAMS.get("model_kwargs", {})
            ).to(torch.device("cpu"))

    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()

    EPOCHS = 10
    for epoch in range(1, EPOCHS+1):
        model.train(); optim.zero_grad()
        pred_tr = model(torch.tensor(X_tr, dtype=torch.float32))
        loss_tr = loss_fn(pred_tr, torch.tensor(y_tr)); loss_tr.backward(); optim.step()

        model.eval()
        with torch.no_grad():
            loss_va = loss_fn(model(torch.tensor(X_va, dtype=torch.float32)),
                              torch.tensor(y_va))
        logger.info("Epoch %d/%d – Train: %.4f , Val: %.4f",
                    epoch, EPOCHS, loss_tr.item(), loss_va.item())

    # ── 5) save artifacts ───────────────────────────────────────────────
    save_model_artifacts(model, scaler, feature_cols, ticker, model_ty)

    # ── 6) generate full‑dataset sequences & predictions ───────────────
    X_full, _ = create_sequences(df, feature_cols, target_cols, seq_len)
    X_full    = scaler.transform(X_full.reshape(-1, X_full.shape[-1])).reshape(X_full.shape)

    with torch.no_grad():
        base_preds_full = model(torch.tensor(X_full, dtype=torch.float32)).cpu().numpy()

    y_meta = df["action_label"].iloc[seq_len:].values.astype(int)
    Path("files/datasets").mkdir(parents=True, exist_ok=True)
    joblib.dump((base_preds_full, y_meta), "files/datasets/meta_dataset.pkl")
    logger.info("Meta‑dataset saved → files/datasets/meta_dataset.pkl  shape=%s",
                base_preds_full.shape)

    logger.info("✅ train_single done in %.1fs", time.time() - t0)
    return {
        "Train_Loss": float(loss_tr.item()),
        "Val_Loss":   float(loss_va.item()),
        "Meta_Shape": base_preds_full.shape,
    }

# helper for FastAPI route --------------------------------------------------

def train_model(request_data: UpdateIndicatorsData):
    return train_single(request_data)
