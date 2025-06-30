#  PYTHONPATH=./src  python3  src/analyze/evaluate_base_models.py
"""
Evaluate every base-model *against its own target column*.
מדפיס MAE / RMSE / R² לכל טארגט +  Dir-Acc (סימן) לטווחי 1-5 ימים.
"""
from datetime import date, timedelta
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    mean_absolute_percentage_error,
)

from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model
from routers.routers_entities import UpdateIndicatorsData


# ── קבועים ────────────────────────────────────────────────────────────
TICKER = "QQQ"
BASE_TARGETS = META_PARAMS["base_targets"]  # 11 targets
SEQ_LEN = META_PARAMS.get("seq_len", 60)
MODEL_DIR = Path("src/files/models")


def classification_metrics(y_true, y_pred, thr=0.0):
    """
    מחזיר dict עם precision/recall/F1 + brier + confusion-matrix
    עבור תחזית סימן (up / down).
    """
    y_bin = (y_true > 0).astype(int)  # 1 = Up, 0 = Down/Flat
    yhat_bin = (y_pred > thr).astype(int)

    return {
        "precision": precision_score(y_bin, yhat_bin, zero_division=0),
        "recall": recall_score(y_bin, yhat_bin, zero_division=0),
        "f1": f1_score(y_bin, yhat_bin, zero_division=0),
        "brier": brier_score_loss(y_bin, yhat_bin),
        "cm": confusion_matrix(y_bin, yhat_bin).tolist(),  # לשמירה כ-JSON
    }


# ──  DF עם כל הפיצ'רים והטארגטים ─────────────────────────────────────
req = UpdateIndicatorsData(
    stock_ticker=TICKER,
    start_date=(date.today() - timedelta(days=365 * 20)).isoformat(),
    end_date=date.today().isoformat(),
    indicators=[],
    scale=True,
)
df_raw = get_indicators_data(req).dropna().reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────────
def _guess_arch_params(chkpt: dict) -> dict:
    """Fallback – שולף hidden_size / num_layers מה-state-dict כאשר
    קובץ-הייפרים *.json* לא נמצא."""
    hidden = chkpt["net.tcn.network.0.conv1.bias"].shape[0]
    # כל בלוק TCN מוסיף שני קונבולוציות; נחפש כמה Layers מופיעים
    n_layers = (
        max(int(k.split(".")[3]) for k in chkpt if k.startswith("net.tcn.network.")) + 1
    )
    return {"hidden_size": hidden, "num_layers": n_layers}


# ───────────────────────────────────────────────────────────────────────
def _load_predict(target: str, df: pd.DataFrame) -> np.ndarray:
    """
    בונה את הטנסור בצורת (batch, F, 60) בדיוק עם מספר-הפיצ'רים
    שה-TCN אומן עליו, מריץ את המודל ומחזיר חיזוי וקטורי.
    """
    stem = MODEL_DIR / f"{TICKER}_{target}"
    ckpt = torch.load(stem.with_suffix(".pt"), map_location="cpu")

    # 1) כמה ערוצים המודל מצפה לקבל?
    in_ch = next(v.shape[1] for v in ckpt.values() if v.ndim == 3)

    # 2) רשימת-פיצ'רים באורך זהה
    feats_all = joblib.load(stem.parent / f"{stem.name}_features.pkl")
    feats = feats_all[:in_ch]

    scaler  = joblib.load(stem.parent / f"{stem.name}_scaler.pkl")
    out_dim = ckpt["net.head.weight"].shape[0]

    # 3) Hyper-params
    hp_file = stem.with_suffix(".json")
    if hp_file.exists():
        hp = json.loads(hp_file.read_text()); hp.pop("lr", None)
        hp["hidden_size"] = hp.pop("hidden",  hp.get("hidden_size", 128))
        hp["num_layers"]  = hp.pop("n_layers", hp.get("num_layers", 4))
    else:
        hp = _guess_arch_params(ckpt)

    model = get_model(len(feats), "TransformerTCN", out_dim, **hp)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 4) Build tensor  (n, F, 60)
    X = np.lib.stride_tricks.sliding_window_view(
            df[feats].values, (SEQ_LEN, len(feats))
        )[:-1]                                          # (n, 60, F)
    X = scaler.transform(
            X.reshape(-1, len(feats))
        ).reshape(-1, SEQ_LEN, len(feats))              # (n, 60, F)
    X = X.swapaxes(1, 2).astype(np.float32)             # (n, F, 60)

    # Safety-check
    if X.shape[1] != in_ch:
        raise RuntimeError(f"Tensor channels={X.shape[1]} ≠ model {in_ch}")

    # 5) Predict
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()

    return pred.squeeze() if out_dim == 1 else pred[:, 0]

# ──  metric helpers ───────────────────────────────────────────────────
def _rmse(y, yhat):
    try:
        return mean_squared_error(y, yhat, squared=False)
    except TypeError:  # sklearn<0.22
        return mean_squared_error(y, yhat) ** 0.5


def dir_acc(y, yhat, thr=0.0):
    return accuracy_score(np.sign(y), np.sign(yhat - thr))


# ──  Run evaluation ───────────────────────────────────────────────────
print("\nTarget evaluation\n" + "-" * 60)
for tgt in BASE_TARGETS:
    y_true = df_raw[tgt].iloc[SEQ_LEN:].values
    y_pred = _load_predict(tgt, df_raw)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    cls = classification_metrics(y_true, y_pred, thr=0.0)

    print(
        f"{tgt:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
        f"R²={r2:.3f}  MAPE={mape:.2%}  "
        f"P={cls['precision']:.3f}  R={cls['recall']:.3f}  "
        f"F1={cls['f1']:.3f}  Brier={cls['brier']:.3f}"
    )

print("\nDirectional-Accuracy (sign)\n" + "-" * 60)
for tgt in BASE_TARGETS[:5]:  # 1-5 day horizons
    y_true = df_raw[tgt].iloc[SEQ_LEN:].values
    y_pred = _load_predict(tgt, df_raw)
    print(f"{tgt:<22}  {dir_acc(y_true, y_pred):.3f}")
