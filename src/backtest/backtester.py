# ──────────────────────────────────────────────────────────────────────────────
# backtest/backtester.py   ← קובץ מלא (רלוונטי רק אם אתה שומר פה את המחלקה)
# ──────────────────────────────────────────────────────────────────────────────
import logging
import time
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model
from models.meta_loader import load_meta_model          # ← use the new loader

# -----------------------------------------------------------------------------


class Backtester:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.logger = self._configure_logger()
        self.model, self.scaler, self.features, self.seq_len = self._load_deep_base()
        self.meta_base, self.meta_clf = self._load_meta()

    # ────────────────────────────────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────────────────────────────────
    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Backtester[{self.ticker}]")
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            h.setFormatter(fmt)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    def _load_deep_base(self) -> Tuple[torch.nn.Module, Any, List[str], int]:
        self.logger.info("Loading deep base model...")
        t0 = time.time()
        model, scaler, features = load_model(
            self.ticker,
            MODEL_TRAINER_PARAMS["model_type"],
        )
        seq_len = MODEL_TRAINER_PARAMS["seq_len"]
        self.logger.info(
            "Deep base loaded (features=%d, seq_len=%d) in %.2fs",
            len(features),
            seq_len,
            time.time() - t0,
        )
        return model, scaler, features, seq_len

    # -------- meta ---------------------------------------------------------
    def _load_meta(self) -> Tuple[Dict[str, Any], Any]:
        """Load {'base': …, 'meta': …} pipeline from .pkl (or return empty)."""
        self.logger.info("Loading meta-model...")
        t0 = time.time()
        try:
            pipeline = load_meta_model(      # ← new utility
                META_PARAMS["meta_model_path"]
            )
            base = pipeline.get("base", {})
            clf = pipeline.get("meta")
            self.logger.info(
                "Meta loaded (%d base models) in %.2fs",
                len(base),
                time.time() - t0,
            )
            return base, clf
        except FileNotFoundError as e:
            self.logger.warning("Meta-model skipped: %s", e)
            return {}, None

    # -------- data prep ----------------------------------------------------
    def _prepare_data(
        self, request_data: Any
    ) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        self.logger.info("Fetching and preparing data...")
        df = (
            get_indicators_data(request_data)
            .dropna()
            .reset_index(drop=True)
        )
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df.empty or len(df) < self.seq_len:
            raise ValueError("Not enough data to backtest.")

        n = len(df) - self.seq_len
        seq = np.stack(
            [df[self.features].iloc[i : i + self.seq_len].values for i in range(n)]
        )
        dates = df["Date"].iloc[self.seq_len :].reset_index(drop=True)
        prices = df["Close"].iloc[self.seq_len :].reset_index(drop=True)

        flat = seq.reshape(-1, seq.shape[2])
        scaled = self.scaler.transform(flat).reshape(seq.shape)
        return dates, prices, scaled

    # -------- inference ----------------------------------------------------
    def _infer(self, Xs: np.ndarray) -> np.ndarray:
        self.logger.info("Running inference...")
        if Xs.size == 0:
            return np.empty((0, len(TRAIN_TARGETS_PARAMS["target_cols"])))
        with torch.no_grad():
            preds = self.model(torch.tensor(Xs, dtype=torch.float32)).cpu().numpy()
        self.logger.info("Inference done; preds.shape=%s", preds.shape)
        return preds

    # -------- simulation ---------------------------------------------------
    def _simulate(
        self,
        dates: pd.Series,
        prices: pd.Series,
        preds: np.ndarray,
    ) -> Tuple[List[Dict], Dict]:
        self.logger.info("Simulating trades...")
        cash = BACKTEST_PARAMS["initial_balance"]
        shares = 0
        trades: List[Dict] = []

        for date, price, row_pred in zip(dates, prices, preds):
            action = int(np.argmax(row_pred)) if preds.size else 1  # 0=SELL,1=HOLD,2=BUY
            if action == 2 and shares == 0:  # BUY
                shares = math.floor(cash / price)
                cash -= shares * price
                trades.append({"Date": date, "Type": "BUY", "Shares": shares, "Cash": cash})
            elif action == 0 and shares > 0:  # SELL
                cash += shares * price
                trades.append({"Date": date, "Type": "SELL", "Shares": shares, "Cash": cash})
                shares = 0

        if shares:
            cash += shares * prices.iloc[-1]
            trades.append({"Date": dates.iloc[-1], "Type": "SELL_EOD", "Shares": shares, "Cash": cash})

        net_pct = (cash / BACKTEST_PARAMS["initial_balance"] - 1) * 100
        stats = {"net_profit_pct": net_pct, "final_balance": cash}
        self.logger.info("Simulation complete; net_profit_pct=%.2f", net_pct)
        return trades, stats

    # -------- public API ---------------------------------------------------
    def run(self, request_data: Any) -> Dict[str, Any]:
        self.logger.info(f"▶️ Starting backtest for {self.ticker}")
        dates, prices, Xs = self._prepare_data(request_data)
        preds = self._infer(Xs)
        trades, stats = self._simulate(dates, prices, preds)
        return {**stats, "trades": trades}


# helper for FastAPI router
def backtest_model(request_data: Any) -> Dict[str, Any]:
    return Backtester(request_data.stock_ticker).run(request_data)
