import logging
import os
import time
import math
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model
from backtest.backtest_utilities import decide_action_meta

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def backtest_model(request_data) -> Dict[str, Any]:
    """
    Orchestrates a backtest run: loads models, prepares data, simulates trades.
    Returns summary statistics and trade log.
    """
    ticker = request_data.stock_ticker
    log.info(f"▶️  Start backtest for {ticker}")
    start_time = time.time()

    # --- load models ---
    model, scaler, feature_cols, seq_len = _load_base(request_data)
    meta_model = _load_meta()

    # --- fetch and preprocess data ---
    df = _get_data(request_data)
    dates, prices, X_scaled, true_vals = _prepare_inputs(
        df, feature_cols, scaler, seq_len, request_data.target_cols
    )

    # --- run inference ---
    base_preds = _infer_base(model, X_scaled)

    # --- simulate trades ---
    trades, stats = _simulate_trades(
        dates, prices, base_preds, meta_model, request_data.target_cols
    )

    log.info(f"✅ Backtest complete in {time.time() - start_time:.1f}s")
    return {**stats, "trades_signals": trades}


# ------------ helper functions ------------


def _load_base(request_data):
    """Load base model, scaler, features, seq_len."""
    try:
        model, scaler, feature_cols, seq_len = load_model(
            request_data.stock_ticker, MODEL_TRAINER_PARAMS["model_type"]
        )
        log.info("   • Base model loaded")
        return model, scaler, feature_cols, seq_len
    except Exception as e:
        log.error("   • Failed to load base model: %s", e, exc_info=True)
        raise


def _load_meta() -> Optional[Any]:
    """Attempt to load meta-model; return None if unavailable."""
    try:
        meta = load_model()
        status = "FOUND" if meta else "NOT FOUND"
        log.info(f"   • Meta-model {status}")
        return meta
    except Exception as e:
        log.warning("   • Meta-model load error: %s", e, exc_info=True)
        return None


def _get_data(request_data) -> pd.DataFrame:
    """Fetch indicator-enriched DataFrame and drop NA."""
    df = get_indicators_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)
    log.info("   • Data fetched: %s rows", len(df))
    return df


def _prepare_inputs(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler,
    seq_len: int,
    target_cols: List[str],
):
    """Builds date/price arrays, feature matrix X, and true target values."""
    if seq_len > 1:
        sequences = [
            df[feature_cols].iloc[i : i + seq_len].values
            for i in range(len(df) - seq_len)
        ]
        X = np.stack(sequences)
        out_idx = slice(seq_len, None)
    else:
        X = df[feature_cols].values
        out_idx = slice(None)

    dates = df["Date"].iloc[out_idx].reset_index(drop=True)
    prices = df["Close"].iloc[out_idx].reset_index(drop=True)
    true_vals = df[target_cols].iloc[out_idx].values

    # scale
    if seq_len > 1:
        flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    return dates, prices, X_scaled, true_vals


def _infer_base(model, X_scaled):
    """Run base-model inference."""
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    log.info("   • Inference complete: preds shape %s", preds.shape)
    return preds


def _simulate_trades(
    dates, prices, preds, meta_model, target_cols
) -> (List[Dict], Dict[str, float]):
    """Executes the trading logic over time."""
    cash = BACKTEST_PARAMS["initial_balance"]
    shares = 0
    last_price = 0.0
    highest = 0.0
    trades = []

    stop_pct = BACKTEST_PARAMS["stop_loss_pct"]
    trail_pct = BACKTEST_PARAMS["trailing_stop"]
    profit_pct = BACKTEST_PARAMS["profit_target"]
    fee_per = BACKTEST_PARAMS["buy_sell_fee_per_share"]
    min_fee = BACKTEST_PARAMS["minimum_fee"]
    tax_rate = BACKTEST_PARAMS["tax_rate"]

    for i, date in enumerate(dates[:-1]):
        price = float(prices.iloc[i])
        action = 1  # hold default
        if meta_model:
            action = decide_action_meta(meta_model, preds, target_cols, i)

        # update highest for trailing stop
        if shares:
            highest = max(highest, price)

        # decide buy/sell
        # BUY
        if action == 2 and not shares:
            fee = max(min_fee, fee_per)
            shares = math.floor((cash - fee) / price)
            cash -= shares * price + fee
            last_price = price
            highest = price
            trades.append({"Date": date, "Type": "BUY", "Price": price, "Cash": cash})

        # SELL or stops
        elif shares and (
            action == 0
            or price <= last_price * (1 - stop_pct)
            or price <= highest * (1 - trail_pct)
            or price >= last_price * (1 + profit_pct)
        ):
            fee = max(min_fee, shares * fee_per)
            tax = tax_rate * (price - last_price) * shares
            cash += shares * price - fee - tax
            trades.append({"Date": date, "Type": "SELL", "Price": price, "Cash": cash})
            shares = 0

    # finalize open position
    if shares:
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

    stats = {
        "ticker_change": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
        "net_profit": (cash / BACKTEST_PARAMS["initial_balance"] - 1) * 100,
        # other stats can be added here
    }
    return trades, stats
