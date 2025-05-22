import logging
import time
import math
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS
from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

# --- Preload meta pipeline at import time to avoid request-time hangs ---
_meta_pipeline: Dict[str, Any]
try:
    t_import_start = time.time()
    log.info("⏳ Pre-loading meta pipeline from %s…", META_PARAMS["meta_model_path"])
    _meta_pipeline = joblib.load(META_PARAMS["meta_model_path"])
    log.info(
        "✅ Pre-loaded meta pipeline in %.2fs (%d base learners)",
        time.time() - t_import_start,
        len(_meta_pipeline.get("base", {})),
    )
except Exception as e:
    log.warning("⚠️ Pre-load meta pipeline failed: %s", e)
    _meta_pipeline = None


def backtest_model(request_data) -> Dict[str, Any]:
    ticker = request_data.stock_ticker
    log.info("▶️  Starting backtest for %s", ticker)
    t0 = time.time()

    # 1) Load deep base (TransformerTCN)
    log.info("⏳ Loading deep base-model…")
    t1 = time.time()
    deep_model, scaler, feature_cols = load_model(
        ticker, MODEL_TRAINER_PARAMS["model_type"]
    )
    seq_len = MODEL_TRAINER_PARAMS["seq_len"]
    log.info(
        "✅ Deep base loaded in %.2fs (features=%d)", time.time() - t1, len(feature_cols)
    )

    # 2) Use preloaded meta pipeline
    if _meta_pipeline:
        meta_base_models: Dict[str, Any] = _meta_pipeline["base"]
        meta_clf = _meta_pipeline["meta"]
    else:
        log.warning("⚠️ Running without meta pipeline; only deep base signals used")
        meta_base_models = {}
        meta_clf = None

    # 3) Fetch & prepare data
    log.info("⏳ Fetching & preparing data…")
    t3 = time.time()
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    if seq_len > 1:
        seqs = [
            df[feature_cols].iloc[i : i + seq_len].values
            for i in range(len(df) - seq_len)
        ]
        X = np.stack(seqs)
        idx = slice(seq_len, None)
    else:
        X = df[feature_cols].values
        idx = slice(None)

    dates = df["Date"].iloc[idx].reset_index(drop=True)
    prices = df["Close"].iloc[idx].reset_index(drop=True)

    if seq_len > 1:
        flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    log.info("✅ Data ready in %.2fs (rows=%d)", time.time() - t3, len(dates))

    # 4) Deep-model inference
    log.info("⏳ Running deep-model inference…")
    t4 = time.time()
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    with torch.no_grad():
        deep_preds = deep_model(tensor).cpu().numpy()
    log.info("✅ Deep inference done in %.2fs", time.time() - t4)

    # 5) Simulate trades (meta + stops)
    log.info("⏳ Simulating trades…")
    t5 = time.time()
    cash, shares, last_price, peak = (BACKTEST_PARAMS["initial_balance"], 0, 0.0, 0.0)
    trades: List[Dict[str, Any]] = []

    for i, date in enumerate(dates[:-1]):
        price = float(prices.iloc[i])
        # Determine action
        if meta_clf:
            feat = deep_preds[i : i + 1, :]
            meta_feats = np.column_stack(
                [clf.predict_proba(feat)[:, 1] for clf in meta_base_models.values()]
            )
            action = int(meta_clf.predict(meta_feats)[0])
        else:
            action = 1  # hold if no meta

        if shares:
            peak = max(peak, price)

        # BUY
        if action == 2 and shares == 0:
            fee = max(
                BACKTEST_PARAMS["minimum_fee"],
                BACKTEST_PARAMS["buy_sell_fee_per_share"],
            )
            shares = math.floor((cash - fee) / price)
            cash -= shares * price + fee
            last_price, peak = price, price
            trades.append({"Date": date, "Type": "BUY", "Price": price, "Cash": cash})

        # SELL or stop-loss
        elif shares > 0 and (
            action == 0
            or price <= last_price * (1 - BACKTEST_PARAMS["stop_loss_pct"])
            or price <= peak * (1 - BACKTEST_PARAMS["trailing_stop_pct"])
        ):
            fee = max(
                BACKTEST_PARAMS["minimum_fee"],
                shares * BACKTEST_PARAMS["buy_sell_fee_per_share"],
            )
            tax = BACKTEST_PARAMS["tax_rate"] * (price - last_price) * shares
            cash += shares * price - fee - tax
            trades.append({"Date": date, "Type": "SELL", "Price": price, "Cash": cash})
            shares = 0

    if shares > 0:
        final_price = float(prices.iloc[-1])
        cash += shares * final_price
        trades.append(
            {
                "Date": dates.iloc[-1],
                "Type": "SELL_FINAL",
                "Price": final_price,
                "Cash": cash,
            }
        )

    for tr in trades:
        dt = tr["Date"]
        if hasattr(dt, "isoformat"):
            tr["Date"] = dt.isoformat()

    log.info("✅ Simulation done in %.2fs", time.time() - t5)
    log.info("🏁 Total backtest time: %.2fs", time.time() - t0)

    stats = {
        "ticker_change": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
        "net_profit": (cash / BACKTEST_PARAMS["initial_balance"] - 1) * 100,
    }
    return {**stats, "trades_signals": trades}
