#  PYTHONPATH=./src  python3  src/analyze/evaluate_base_models.py
"""
Evaluate every base-model *against its own target column*.
מדפיס MAE / RMSE / R² לכל טארגט +  Dir-Acc (סימן) לטווחי 1-5 ימים.
"""
from datetime import date, timedelta
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score)

from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData

# ── קבועים ────────────────────────────────────────────────────────────
TICKER       = "QQQ"
BASE_TARGETS = META_PARAMS["base_targets"]           # 11 targets
SEQ_LEN      = META_PARAMS.get("seq_len", 60)
MODEL_DIR    = Path("src/files/models")

# ──  DF עם כל הפיצ'רים והטארגטים ─────────────────────────────────────
req = UpdateIndicatorsData(
    ticker      = TICKER,
    start_date  = (date.today() - timedelta(days=365*20)).isoformat(),
    end_date    = date.today().isoformat(),
    indicators  = [],
)
df_raw = get_indicators_data(req).dropna().reset_index(drop=True)

# ───────────────────────────────────────────────────────────────────────
def _guess_arch_params(chkpt: dict) -> dict:
    """Fallback – שולף hidden_size / num_layers מה-state-dict כאשר
       קובץ-הייפרים *.json* לא נמצא."""
    hidden = chkpt["net.tcn.network.0.conv1.bias"].shape[0]
    # כל בלוק TCN מוסיף שני קונבולוציות; נחפש כמה Layers מופיעים
    n_layers = max(int(k.split('.')[3]) for k in chkpt if k.startswith("net.tcn.network.")) + 1
    return {"hidden_size": hidden, "num_layers": n_layers}

# ───────────────────────────────────────────────────────────────────────
def _load_predict(target: str, df: pd.DataFrame) -> np.ndarray:
    """חזוי scalar אחד לכל שורה – עבור טארגט מסוים."""
    stem   = MODEL_DIR / f"{TICKER}_{target}"
    chkpt  = torch.load(stem.with_suffix(".pt"), map_location="cpu")
    feats  = joblib.load(stem.parent / f"{stem.name}_features.pkl")
    scaler = joblib.load(stem.parent / f"{stem.name}_scaler.pkl")
    out_dim = chkpt["net.head.weight"].shape[0]

    # -------- hyper-params (json או Fallback) -------------------------
    hp = {}
    hp_file = stem.with_suffix(".json")
    if hp_file.exists():
        hp = json.loads(hp_file.read_text())
        hp.pop("lr", None)                       # לא דרוש לאינפרנס
        # map legacy names → get_model kwargs
        if "hidden"   in hp: hp["hidden_size"] = hp.pop("hidden")
        if "n_layers" in hp: hp["num_layers"]  = hp.pop("n_layers")
    else:
        hp = _guess_arch_params(chkpt)

    # -------- מודל -----------------------------------------------------
    model = get_model(len(feats), "TransformerTCN", out_dim, **hp)
    model.load_state_dict(chkpt, strict=False)
    model.eval()

    # -------- build tensor --------------------------------------------
    X = np.lib.stride_tricks.sliding_window_view(
            df[feats].values, (SEQ_LEN, len(feats))
        )[:-1]                                   # (n, 1, 1, 60, F)
    X = X.reshape(X.shape[0], SEQ_LEN, -1)       # (n, 60, F)
    X = scaler.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(-1, SEQ_LEN, len(feats)).astype(np.float32)

    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()    # (n, out_dim)

    return pred.squeeze() if pred.shape[1] == 1 else pred[:, 0]

# ──  metric helpers ───────────────────────────────────────────────────
def _rmse(y, yhat):
    try:
        return mean_squared_error(y, yhat, squared=False)
    except TypeError:                            # sklearn<0.22
        return mean_squared_error(y, yhat) ** 0.5

def dir_acc(y, yhat, thr=0.0):
    return accuracy_score(np.sign(y), np.sign(yhat - thr))

# ──  Run evaluation ───────────────────────────────────────────────────
print("\nTarget evaluation\n" + "-"*60)
for tgt in BASE_TARGETS:
    y_true = df_raw[tgt].iloc[SEQ_LEN:].values
    y_pred = _load_predict(tgt, df_raw)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{tgt:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.3f}")

print("\nDirectional-Accuracy (sign)\n" + "-"*60)
for tgt in BASE_TARGETS[:5]:          # 1-5 day horizons
    y_true = df_raw[tgt].iloc[SEQ_LEN:].values
    y_pred = _load_predict(tgt, df_raw)
    print(f"{tgt:<22}  {dir_acc(y_true, y_pred):.3f}")
