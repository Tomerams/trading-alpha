"""
Back‑testing engine (Pro)
────────────────────────
• Loads **multiple** base DL models (as listed in `META_PARAMS["base_targets"]`).
• Feeds their class‑1 probabilities into a stacked LightGBM meta‑model.
• Executes BUY / SELL / HOLD decisions returned by that meta‑model.
• Simulates brokerage fees, minimal fee, stop‑loss, trailing stop, and tax on realised profit.
• Compares the equity curve against a buy‑and‑hold benchmark.
• Returns a **fully JSON‑serialisable** dict – safe for FastAPI responses.

This file replaces previous `backtest/backtester.py` implementations.

Usage (stand‑alone):
```
python -m backtest.backtester  # runs a demo on QQQ based on BACKTEST_PARAMS
```

The `run()` method is designed to be called by FastAPI via the helper
`backtest_model()` at the bottom of the file.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS
from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model
from models.model_meta_trainer import load_meta_model  # meta‑model bundle loader
from routers.routers_entities import UpdateIndicatorsData

# ────────────────────────────────────────────────────────────────────────
JSON_SAFE = (int, float, str, bool, type(None))

# ────────────────────────────────────────────────────────────────────────
class Backtester:
    """Simulates a trading strategy over historical data."""

    def __init__(self, ticker: str, debug: bool = False):
        self.ticker = ticker.upper()
        self.debug = debug
        self.log = self._init_logger()

        # load all base models (one per target listed in meta config)
        self.base_models: dict[str, Tuple[nn.Module, Any, List[str]]]  # model, scaler, feats
        self.seq_len: int
        self.base_models, self.seq_len = self._load_base_models()

        # load meta‑model (returns feature list + classifier)
        self.meta_feat_cols: List[str] | None
        self.meta_clf: Any | None
        self.meta_feat_cols, self.meta_clf = self._load_meta_model()

    # ------------------------------------------------------------------
    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Backtester[{self.ticker}]")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s • %(levelname)-8s • %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        return logger

    # ------------------------------------------------------------------
    def _load_base_models(self) -> Tuple[dict, int]:
        """Load one DL base model per target defined in META_PARAMS."""
        targets = META_PARAMS.get("base_targets", [])
        if not targets:
            raise ValueError("META_PARAMS['base_targets'] is empty – nothing to back‑test.")

        seq_len = MODEL_TRAINER_PARAMS["seq_len"]
        models: dict[str, Tuple[nn.Module, Any, List[str]]] = {}
        for tgt in targets:
            model, scaler, feats = load_model(self.ticker, tgt)
            models[tgt] = (model, scaler, feats)
            self.log.info("Loaded base model %-18s  (features=%d)", tgt, len(feats))
        return models, seq_len

    # ------------------------------------------------------------------
    def _load_meta_model(self) -> Tuple[List[str] | None, Any | None]:
        path = META_PARAMS.get("meta_model_path", "files/models/meta_action_model.pkl")
        try:
            bundle = load_meta_model(path)
            feats = bundle["feature_columns"]
            clf = bundle["meta_model"]
            self.log.info("Loaded meta‑model ➜ %s (features=%d)", path, len(feats))
            return feats, clf
        except FileNotFoundError:
            self.log.warning("Meta‑model file not found (%s) – falling back to arg‑max", path)
            return None, None

    # ------------------------------------------------------------------
    def _prepare_meta_features(self, req: UpdateIndicatorsData) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        """Construct the meta‑feature matrix from base model probabilites."""
        df = get_indicators_data(req).dropna().reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        if len(df) <= self.seq_len:
            raise ValueError("Not enough rows to back‑test (need > seq_len).")

        n_rows = len(df) - self.seq_len
        n_models = len(self.base_models)
        meta_mat = np.empty((n_rows, n_models), dtype=np.float32)

        for col_idx, (tgt, (model, scaler, feats)) in enumerate(self.base_models.items()):
            # Build rolling window tensor
            windows = np.stack(
                [df[feats].iloc[i : i + self.seq_len].values for i in range(n_rows)]
            )
            windows = scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape)
            with torch.no_grad():
                logits = model(torch.tensor(windows, dtype=torch.float32))
                probs = logits.softmax(dim=1).cpu().numpy()
            # class‑1 probability (buy probability)
            meta_mat[:, col_idx] = probs[:, 1]

        dates = df["Date"].iloc[self.seq_len :].dt.strftime("%Y-%m-%d").reset_index(drop=True)
        prices = df["Close"].iloc[self.seq_len :].astype(float).reset_index(drop=True)
        return dates, prices, meta_mat

    # ------------------------------------------------------------------
    def _decide_actions(self, meta_X: np.ndarray) -> np.ndarray:
        """Return 0/1/2 decision per bar (SELL / HOLD / BUY)."""
        if self.meta_clf is None:
            # fallback: BUY if prob_buy > prob_sell, otherwise HOLD/SELL heuristic
            buy_prob = meta_X[:, 0]  # assuming first column is buy prob – heuristic
            sell_prob = meta_X[:, 0]  # same – we have only one model per target
            return np.where(buy_prob > sell_prob, 2, 0)
        return self.meta_clf.predict(meta_X)

    # ------------------------------------------------------------------
    def _fee(self, qty: int) -> float:
        per_share = qty * BACKTEST_PARAMS["buy_sell_fee_per_share"]
        return max(per_share, BACKTEST_PARAMS["minimum_fee"])

    # ------------------------------------------------------------------
    def _simulate(self, dates: pd.Series, prices: pd.Series, actions: np.ndarray) -> Dict[str, Any]:
        cash: float = BACKTEST_PARAMS["initial_balance"]
        shares = 0
        last_buy_px = None
        high_water = None  # for trailing stop
        total_fees = 0.0
        trades: List[Dict[str, Any]] = []

        for dt, px, act in zip(dates, prices, actions):
            px = float(px)

            # Trailing / stop‑loss override
            if shares:
                high_water = max(high_water, px)
                if BACKTEST_PARAMS["trailing_stop_pct"] and px <= high_water * (1 - BACKTEST_PARAMS["trailing_stop_pct"]):
                    act = 0
                if BACKTEST_PARAMS["stop_loss_pct"] and px <= last_buy_px * (1 - BACKTEST_PARAMS["stop_loss_pct"]):
                    act = 0

            # BUY --------------------------------------------------
            if act == 2 and shares == 0:
                qty = math.floor(cash / px)
                if qty == 0:
                    continue
                fee = self._fee(qty)
                cash -= qty * px + fee
                shares = qty
                last_buy_px = px
                high_water = px
                total_fees += fee
                trades.append({"Date": dt, "Type": "BUY", "shares": qty, "price": px, "cash": round(cash, 2), "fee": round(fee, 2)})

            # SELL -------------------------------------------------
            elif act == 0 and shares > 0:
                proceeds = shares * px
                fee = self._fee(shares)
                profit = proceeds - fee - shares * last_buy_px
                tax = max(profit, 0) * BACKTEST_PARAMS["tax_rate"]
                cash += proceeds - fee - tax
                total_fees += fee
                trades.append({
                    "Date": dt,
                    "Type": "SELL",
                    "shares": shares,
                    "price": px,
                    "cash": round(cash, 2),
                    "fee": round(fee, 2),
                    "tax": round(tax, 2),
                })
                shares = 0
                last_buy_px = None
                high_water = None

        # Liquidate at end
        if shares:
            px = float(prices.iloc[-1])
            proceeds = shares * px
            fee = self._fee(shares)
            profit = proceeds - fee - shares * last_buy_px
            tax = max(profit, 0) * BACKTEST_PARAMS["tax_rate"]
            cash += proceeds - fee - tax
            total_fees += fee
            trades.append({
                "Date": dates.iloc[-1],
                "Type": "SELL_EOD",
                "shares": shares,
                "price": px,
                "cash": round(cash, 2),
                "fee": round(fee, 2),
                "tax": round(tax, 2),
            })

        gross_pct = (cash + total_fees) / BACKTEST_PARAMS["initial_balance"] - 1
        net_pct = cash / BACKTEST_PARAMS["initial_balance"] - 1
        tax_paid = sum(t.get("tax", 0) for t in trades)

        return {
            "strategy_gross_pct": round(gross_pct * 100, 2),
            "strategy_net_pct": round(net_pct * 100, 2),
            "total_fees": round(total_fees, 2),
            "tax_paid_pct": round(tax_paid / BACKTEST_PARAMS["initial_balance"] * 100, 2),
            "final_balance": round(cash, 2),
            "trades": trades,
        }

    # ------------------------------------------------------------------
    def _benchmark(self, prices: pd.Series) -> float:
        return round((prices.iloc[-1] / prices.iloc[0] - 1) * 100, 2)

    # ------------------------------------------------------------------
    def run(self, req: UpdateIndicatorsData) -> Dict[str, Any]:
        self.log.info("▶️ Back‑testing %s", self.ticker)
        dates, prices, meta_X = self._prepare_meta_features(req)
        actions = self._decide_actions(meta_X)
        result = self._simulate(dates, prices, actions)
        result["benchmark_pct"] = self._benchmark(prices)
        return result

# ────────────────────────────────────────────────────────────────────────
# Helper for FastAPI router
# ────────────────────────────────────────────────────────────────────────

def _safe(v):
    if isinstance(v, JSON_SAFE):
        return v
    if isinstance(v, (np.integer, np.int64)):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return float(round(v, 6))
    return str(v)


def backtest_model(request_data: UpdateIndicatorsData) -> Dict[str, Any]:
    raw = Backtester(request_data.stock_ticker, debug=request_data.__dict__.get("debug", False)).run(request_data)
    raw["trades"] = [{k: _safe(v) for k, v in t.items()} for t in raw["trades"]]
    return {k: _safe(v) if k != "trades" else v for k, v in raw.items()}

# ----------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date, timedelta

    class _Demo(UpdateIndicatorsData):
        stock_ticker: str = "QQQ"
        from_date: str = (date.today() - timedelta(days=365 * 5)).isoformat()
        to_date: str = date.today().isoformat()

    demo_req = _Demo()
    output = Backtester("QQQ", debug=True).run(demo_req)
    print(json.dumps({k: v for k, v in output.items() if k != "trades"}, indent=2))
