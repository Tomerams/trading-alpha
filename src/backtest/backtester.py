import logging
import time
import math
from typing import Any, Dict, List, Tuple
import os
import pickle

import numpy as np
import pandas as pd
import torch

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS
from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model

# Configure root logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Preload deep base loader remains per-request ---

def _load_deep_base(ticker: str) -> Tuple[Any, Any, List[str], int]:
    """Load TransformerTCN, scaler, feature columns, and sequence length."""
    logger.info("⏳ Loading deep base-model for %s…", ticker)
    start = time.time()
    model, scaler, feature_cols = load_model(
        ticker, MODEL_TRAINER_PARAMS['model_type']
    )
    seq_len = MODEL_TRAINER_PARAMS['seq_len']
    logger.info("✅ Deep base loaded in %.2f s (features=%d)", time.time()-start, len(feature_cols))
    return model, scaler, feature_cols, seq_len

# --- Preload meta pipeline at import time to avoid per-request hangs ---
_meta_base: Dict[str, Any] = {}
_meta_clf: Any = None
_meta_loaded = False
try:
    meta_path = META_PARAMS['meta_model_path']
    logger.info("⏳ Preloading meta-model pipeline from %s…", meta_path)
    start = time.time()
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, 'rb') as f:
        pipeline = pickle.load(f)
    _meta_base = pipeline.get('base', {})
    _meta_clf = pipeline.get('meta')
    _meta_loaded = _meta_clf is not None and bool(_meta_base)
    logger.info(
        "✅ Meta pipeline preloaded in %.2f s (%d base models)",
        time.time() - start,
        len(_meta_base)
    )
except Exception as e:
    logger.error("❌ Failed to preload meta pipeline: %s", e, exc_info=True)
    _meta_base = {}
    _meta_clf = None
    _meta_loaded = False


def _prepare_data(
    request_data: Any,
    feature_cols: List[str],
    scaler: Any,
    seq_len: int
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Fetch indicator data, build sequences, and scale features."""
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if seq_len > 1:
        X = np.stack([
            df[feature_cols].iloc[i:i+seq_len].values
            for i in range(len(df)-seq_len)
        ])
        idx = slice(seq_len, None)
    else:
        X = df[feature_cols].values
        idx = slice(None)
    dates = df['Date'].iloc[idx].reset_index(drop=True)
    prices = df['Close'].iloc[idx].reset_index(drop=True)
    if seq_len > 1:
        flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)
    logger.info("✅ Data prepared: %d rows", len(dates))
    return dates, prices, X_scaled


def _infer_deep(model: Any, X_scaled: np.ndarray) -> np.ndarray:
    """Run deep model inference and return predictions."""
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    with torch.no_grad():
        preds = model(tensor).cpu().numpy()
    logger.info("✅ Deep inference complete (shape=%s)", preds.shape)
    return preds


def _simulate_trades(
    dates: pd.Series,
    prices: pd.Series,
    deep_preds: np.ndarray
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Simulate trades using preloaded meta-model and stop rules."""
    cash = BACKTEST_PARAMS['initial_balance']
    shares = 0
    last_price = 0.0
    peak = 0.0
    trades: List[Dict[str, Any]] = []
    for i, date in enumerate(dates[:-1]):
        price = float(prices.iloc[i])
        if _meta_loaded:
            feat = deep_preds[i:i+1]
            probs = [clf.predict_proba(feat)[0,1] for clf in _meta_base.values()]
            action = int(_meta_clf.predict([probs])[0])
        else:
            action = int(np.argmax(deep_preds[i]))
        if shares:
            peak = max(peak, price)
        if action == 2 and shares == 0:
            fee = max(
                BACKTEST_PARAMS['minimum_fee'],
                BACKTEST_PARAMS['buy_sell_fee_per_share']
            )
            shares = math.floor((cash - fee) / price)
            cash -= shares * price + fee
            last_price, peak = price, price
            trades.append({'Date': date.isoformat(), 'Type': 'BUY', 'Price': price, 'Cash': cash})
        elif shares > 0 and (
            action == 0 or
            price <= last_price * (1 - BACKTEST_PARAMS['stop_loss_pct']) or
            price <= peak * (1 - BACKTEST_PARAMS['trailing_stop_pct'])
        ):
            fee = max(
                BACKTEST_PARAMS['minimum_fee'],
                shares * BACKTEST_PARAMS['buy_sell_fee_per_share']
            )
            tax = BACKTEST_PARAMS['tax_rate'] * (price - last_price) * shares
            cash += shares * price - fee - tax
            trades.append({'Date': date.isoformat(), 'Type': 'SELL', 'Price': price, 'Cash': cash})
            shares = 0
    if shares > 0:
        final_price = float(prices.iloc[-1])
        cash += shares * final_price
        trades.append({'Date': dates.iloc[-1].isoformat(), 'Type': 'SELL_FINAL', 'Price': final_price, 'Cash': cash})
    stats = {
        'ticker_change': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
        'net_profit': (cash / BACKTEST_PARAMS['initial_balance'] - 1) * 100,
    }
    return trades, stats


def backtest_model(request_data) -> Dict[str, Any]:
    """Run full backtest: load, prepare, infer, simulate."""
    ticker = request_data.stock_ticker
    logger.info("▶️  Backtest for %s", ticker)
    start = time.time()
    deep_model, scaler, feature_cols, seq_len = _load_deep_base(ticker)
    dates, prices, X_scaled = _prepare_data(request_data, feature_cols, scaler, seq_len)
    deep_preds = _infer_deep(deep_model, X_scaled)
    trades, stats = _simulate_trades(dates, prices, deep_preds)
    logger.info("🏁 Completed in %.2f s", time.time() - start)
    return {**stats, 'trades_signals': trades}
