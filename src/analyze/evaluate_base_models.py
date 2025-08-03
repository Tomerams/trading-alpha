#!/usr/bin/env python3
"""
Evaluate every base-model *against its own target column*.
מדפיס MAE / RMSE / R² + Precision / Recall / F1 לכל טארגט,
ובסוף Direction-Accuracy (Up / Down) לטווחי 1-5 ימים.

• נשען על load_pipeline()  →  טוען net, scaler ורשימת-פיצ'רים
• אין יותר ניחושים של hidden_size / num_layers
• Sliding-window ידני – לא משתמש ב-DataLoader
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, torch

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, precision_score, recall_score,
    f1_score, brier_score_loss, accuracy_score, confusion_matrix
)

from config.meta_data_config import META_PARAMS
from data.data_processing   import get_indicators_data
from analyze.load_trained   import load_pipeline                # ⬅️  loader אחיד
from routers.routers_entities import UpdateIndicatorsData


# ───────────────────────── קבועים ─────────────────────────
TICKER       = "QQQ"
BASE_TARGETS = META_PARAMS["base_targets"]          # 11 טארגטים
SEQ_LEN      = META_PARAMS.get("seq_len", 60)


# ───────────────────────── כלי-עזר ─────────────────────────
def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root-Mean-Squared-Error – חישוב ידני, ללא תלות ב-squared=True."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean-Absolute-Percentage-Error‏ (0-1). מדלג על ערכי-אמת אפס."""
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def _cls_metrics(y: np.ndarray, yhat: np.ndarray, thr=0.0) -> dict:
    """Precision / Recall / F1 + Brier עבור סימן Up/Down."""
    y_bin    = (y > 0).astype(int)
    yhat_bin = (yhat > thr).astype(int)
    return {
        "precision": precision_score(y_bin, yhat_bin, zero_division=0),
        "recall":    recall_score   (y_bin, yhat_bin, zero_division=0),
        "f1":        f1_score       (y_bin, yhat_bin, zero_division=0),
        "brier":     brier_score_loss(y_bin, yhat_bin),
        "cm":        confusion_matrix(y_bin, yhat_bin).tolist(),
    }


def _dir_acc(y: np.ndarray, yhat: np.ndarray, thr=0.0) -> float:
    """Directional-Accuracy (Up/Down)."""
    return accuracy_score(np.sign(y), np.sign(yhat - thr))


# ────────────────── טעינת נתוני-מחיר גולמיים ──────────────────
def _load_raw_prices() -> pd.DataFrame:
    req = UpdateIndicatorsData(
        stock_ticker=TICKER,
        start_date  =(date.today() - timedelta(days=20 * 365)).isoformat(),
        end_date    = date.today().isoformat(),
        indicators  = [],       # calculate_features תוסיף הכל
        scale       = False     # אין סקיילינג כפול
    )
    return (
        get_indicators_data(req)
        .dropna()
        .reset_index(drop=True)
    )


# ────────────────── חיזוי לטארגט בודד ──────────────────
def _predict_target(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    • load_pipeline → net, scaler, feature-list
    • בונה sliding-window: (n_samples, seq_len, n_features)
    • מחזיר וקטור-חיזוי (n_samples,)
    """
    net, scaler_x, feats = load_pipeline(TICKER, target)
    n_feats = len(feats)

    if any(f not in df.columns for f in feats):
        missing = [f for f in feats if f not in df.columns]
        raise ValueError(f"{target}: missing features {missing}")

    X_raw     = df[feats].to_numpy()
    n_samples = len(X_raw) - SEQ_LEN
    if n_samples <= 0:
        raise ValueError(f"Need >{SEQ_LEN} rows, got {len(X_raw)}")

    # (batch, seq_len, feats)
    X = np.stack([X_raw[i:i + SEQ_LEN] for i in range(n_samples)], axis=0)

    # scale
    X_scaled = scaler_x.transform(X.reshape(-1, n_feats))\
                       .reshape(n_samples, SEQ_LEN, n_feats)\
                       .astype(np.float32)

    with torch.no_grad():
        y_hat = net(torch.tensor(X_scaled)).cpu().numpy().squeeze()

    return y_hat


# ────────────────── MAIN ──────────────────
def main() -> None:
    print("Loading data …")
    df = _load_raw_prices()
    print(f"Data shape: {df.shape}")

    print(f"\nEvaluating {len(BASE_TARGETS)} targets …\n" + "-" * 80)
    ok, failed = 0, []

    for tgt in BASE_TARGETS:
        try:
            y_true = df[tgt].iloc[SEQ_LEN:].to_numpy()
            y_pred = _predict_target(df, tgt)

            m = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:m], y_pred[:m]

            mae  = mean_absolute_error(y_true, y_pred)
            rmse = _rmse(y_true, y_pred)
            r2   = r2_score(y_true, y_pred)
            mape = _mape(y_true, y_pred)
            cls  = _cls_metrics(y_true, y_pred)

            print(
                f"{tgt:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
                f"R²={r2:7.3f}  MAPE={mape:7.2%}  "
                f"P={cls['precision']:.3f}  R={cls['recall']:.3f}  "
                f"F1={cls['f1']:.3f}  Brier={cls['brier']:.3f}"
            )
            ok += 1
        except Exception as e:
            print(f"✗ {tgt}: {e}")
            failed.append(tgt)

    # --- Directional-Accuracy (Up/Down) לטווחים קצרים ---
    print("\nDirectional-Accuracy (sign) – first 5 targets\n" + "-" * 60)
    for tgt in BASE_TARGETS[:5]:
        if tgt in failed:
            print(f"{tgt:<22}  –")
            continue
        try:
            y_true = df[tgt].iloc[SEQ_LEN:].to_numpy()
            y_pred = _predict_target(df, tgt)
            print(f"{tgt:<22}  {_dir_acc(y_true, y_pred):.3f}")
        except Exception as e:
            print(f"{tgt:<22}  ERROR: {e}")

    print(f"\nSummary: {ok}/{len(BASE_TARGETS)} targets evaluated successfully")
    if failed:
        print("Failed:", failed)


# ────────────────── run ──────────────────
if __name__ == "__main__":
    main()