#PYTHONPATH=./src python3 src/analyze/evaluate_base_models.py
"""
Evaluate every base-model *against its own target column*.
Prints regression or classification metrics per target.
"""
from datetime import date, timedelta
from routers.routers_entities import UpdateIndicatorsData

import numpy as np, pandas as pd, joblib, torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model

TICKER       = "QQQ"
BASE_TARGETS = META_PARAMS["base_targets"]
SEQ_LEN      = META_PARAMS.get("seq_len", 60)
MODEL_DIR    = Path("src/files/models")

req = UpdateIndicatorsData(
    ticker="QQQ",
    start_date=(date.today() - timedelta(days=365*20)).isoformat(),
    end_date=date.today().isoformat(),
    indicators=[]          # השאר ריק אם אינך צריך חישוב נוסף
)

df_raw = get_indicators_data(req).dropna().reset_index(drop=True)

def load_predict(target: str, df: pd.DataFrame) -> np.ndarray:
    """Return model predictions (scalar or Δprob BUY-SELL) for given target."""
    stem       = MODEL_DIR / f"{TICKER}_{target}"
    chkpt      = torch.load(stem.with_suffix(".pt"), map_location="cpu")
    feats      = joblib.load(stem.parent / f"{stem.name}_features.pkl")
    scaler     = joblib.load(stem.parent / f"{stem.name}_scaler.pkl")
    out_dim    = chkpt["net.head.weight"].shape[0]

    model = get_model(len(feats), "TransformerTCN", out_dim)
    model.load_state_dict(chkpt)
    model.eval()

    X = np.lib.stride_tricks.sliding_window_view(
        df[feats].values, (SEQ_LEN, len(feats))
    )[:-1]
    X = X.reshape(X.shape[0], SEQ_LEN, -1)
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32)).numpy()  # (n, out_dim)

    # אם המודל הוא 3-class – נחזיר Δprobability (BUY-SELL); אחרת scalar
    if out.shape[1] == 3:
        return out[:, 2] - out[:, 0]      # BUY – SELL
    return out[:, 0]                      # scalar יחיד

print("\nTarget evaluation")
print("-" * 60)
for tgt in BASE_TARGETS:
    y_true = df_raw[tgt].iloc[SEQ_LEN:].values
    y_pred = load_predict(tgt, df_raw)

    # Regression or classification?
    if np.issubdtype(y_true.dtype, np.floating):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2   = r2_score(y_true, y_pred)
        print(f"{tgt:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.3f}")
    else:  # assume int labels 0/1/2
        auc  = roc_auc_score((y_true==2).astype(int), y_pred)  # BUY-OVR
        f1   = f1_score(y_true, (y_pred>0).astype(int), average="weighted")
        acc  = accuracy_score(y_true, (y_pred>0).astype(int))
        print(f"{tgt:<22}  AUC={auc:.3f}  F1={f1:.3f}  Acc={acc:.3f}")
