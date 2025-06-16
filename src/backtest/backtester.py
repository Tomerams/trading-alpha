"""
Back-testing engine (Pro) – Rev-E 2025-06-11
────────────────────────────────────────────
* טוען את כל מודלי-הבסיס המופיעים ב-META_PARAMS["base_targets"]
* מפיק לכל יום את הסתברות BUY (class-1) מכל מודל-בסיס
* מזין את המטריצה למטא-מודל (LightGBM-stack) ומקבל BUY / HOLD / SELL
* מדמה עמלות, Stop-Loss, Trailing-Stop ומס רווח-הון
* משווה לעקום Buy-&-Hold ומחזיר JSON-סדרתי
"""
from __future__ import annotations

import math, logging, time, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np, pandas as pd, torch
from torch import nn

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS
from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model
from models.meta_loader import load_meta_model  # ← loader יחיד
from routers.routers_entities import UpdateIndicatorsData

# ---------------------------------------------------------------------
JSON_SAFE = (int, float, str, bool, type(None))


class Backtester:
    def __init__(self, ticker: str, debug: bool = False):
        self.ticker = ticker.upper()
        self.debug = debug
        self.log = self._logger()

        self.base_models, self.seq_len = self._load_base_models()
        self.meta_clf = self._load_meta_model()

    # ──────────────────────────────────────────────────────────────
    def _logger(self) -> logging.Logger:
        lg = logging.getLogger(f"Backtester[{self.ticker}]")
        if not lg.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s • %(levelname)-7s • %(message)s")
            )
            lg.addHandler(h)
        lg.setLevel(logging.DEBUG if self.debug else logging.INFO)
        return lg

    # ──────────────────────────────────────────────────────────────
    def _load_base_models(self) -> Tuple[dict, int]:
        targets = META_PARAMS["base_targets"]
        models: dict[str, Tuple[nn.Module, Any, List[str]]] = {}
        L = MODEL_TRAINER_PARAMS["seq_len"]

        for tgt in targets:
            mdl, scl, feats = load_model(
                self.ticker,  # ← model_type = arch.
                MODEL_TRAINER_PARAMS["model_type"],
                target_override=tgt,
            )  # ← middle-part
            models[tgt] = (mdl, scl, feats)
            self.log.info("Loaded base model %-18s (feats=%d)", tgt, len(feats))

        return models, L

    # ──────────────────────────────────────────────────────────────
    def _load_meta_model(self):
        path = Path(META_PARAMS["meta_model_path"])
        try:
            bundle = load_meta_model(path)
            self.log.info("Loaded meta-model → %s", path)
            return bundle["meta_model"]  # bundle = {base, meta, metrics}
        except FileNotFoundError:
            self.log.warning("Meta-model NOT found – fallback argmax.")
            return None

    # ──────────────────────────────────────────────────────────────
    def _meta_features(
        self, req: UpdateIndicatorsData
    ) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        df = get_indicators_data(req).dropna().reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"])

        if len(df) <= self.seq_len:
            raise ValueError("Data shorter than seq_len")

        n = len(df) - self.seq_len
        M = len(self.base_models)
        X = np.empty((n, M), dtype=np.float32)

        for j, (tgt, (mdl, scl, feats)) in enumerate(self.base_models.items()):
            win = np.stack(
                [df[feats].iloc[i : i + self.seq_len].values for i in range(n)]
            )
            win = scl.transform(win.reshape(-1, win.shape[-1])).reshape(win.shape)
            with torch.no_grad():
                prob_buy = (
                    mdl(torch.tensor(win, dtype=torch.float32))
                    .softmax(1)[:, 1]
                    .cpu()
                    .numpy()
                )
            X[:, j] = prob_buy

        dates = (
            df["Date"]
            .iloc[self.seq_len :]
            .dt.strftime("%Y-%m-%d")
            .reset_index(drop=True)
        )
        prices = df["Close"].iloc[self.seq_len :].astype(float).reset_index(drop=True)
        return dates, prices, X

    # ──────────────────────────────────────────────────────────────
    def _decide(self, X: np.ndarray) -> np.ndarray:
        if self.meta_clf is not None:
            return self.meta_clf.predict(X)
        # fallback: BUY if mean(prob)>0.5 else SELL
        p = X.mean(1)
        return np.where(p > 0.5, 2, 0)

    # ──────────────────────────────────────────────────────────────
    def _fee(self, qty: int) -> float:
        return max(
            qty * BACKTEST_PARAMS["buy_sell_fee_per_share"],
            BACKTEST_PARAMS["minimum_fee"],
        )

    def _simulate(self, dates, prices, acts) -> Dict[str, Any]:
        cash = BACKTEST_PARAMS["initial_balance"]
        shares = 0
        fees = 0.0
        H, lbp = None, None  # high-water, last-buy-price
        trades = []

        for dt, px, a in zip(dates, prices, acts):
            px = float(px)
            if shares:
                H = max(H, px)
                if BACKTEST_PARAMS["trailing_stop_pct"] and px <= H * (
                    1 - BACKTEST_PARAMS["trailing_stop_pct"]
                ):
                    a = 0
                if BACKTEST_PARAMS["stop_loss_pct"] and px <= lbp * (
                    1 - BACKTEST_PARAMS["stop_loss_pct"]
                ):
                    a = 0

            # BUY
            if a == 2 and shares == 0:
                qty = math.floor(cash / px)
                if qty:
                    f = self._fee(qty)
                    fees += f
                    cash -= qty * px + f
                    shares = qty
                    lbp = px
                    H = px
                    trades.append(
                        {
                            "Date": dt,
                            "Type": "BUY",
                            "shares": qty,
                            "price": px,
                            "cash": round(cash, 2),
                            "fee": round(f, 2),
                        }
                    )

            # SELL
            elif a == 0 and shares > 0:
                f = self._fee(shares)
                fees += f
                profit = (px - lbp) * shares - f
                tax = max(profit, 0) * BACKTEST_PARAMS["tax_rate"]
                cash += shares * px - f - tax
                trades.append(
                    {
                        "Date": dt,
                        "Type": "SELL",
                        "shares": shares,
                        "price": px,
                        "cash": round(cash, 2),
                        "fee": round(f, 2),
                        "tax": round(tax, 2),
                    }
                )
                shares = 0
                lbp = None
                H = None

        # final liquidation
        if shares:
            f = self._fee(shares)
            fees += f
            profit = (prices.iloc[-1] - lbp) * shares - f
            tax = max(profit, 0) * BACKTEST_PARAMS["tax_rate"]
            cash += shares * prices.iloc[-1] - f - tax
            trades.append(
                {
                    "Date": dates.iloc[-1],
                    "Type": "SELL_EOD",
                    "shares": shares,
                    "price": float(prices.iloc[-1]),
                    "cash": round(cash, 2),
                    "fee": round(f, 2),
                    "tax": round(tax, 2),
                }
            )

        gross = (cash + fees) / BACKTEST_PARAMS["initial_balance"] - 1
        net = cash / BACKTEST_PARAMS["initial_balance"] - 1
        tax_paid = sum(t.get("tax", 0) for t in trades)
        return dict(
            strategy_gross_pct=round(gross * 100, 2),
            strategy_net_pct=round(net * 100, 2),
            total_fees=round(fees, 2),
            tax_paid_pct=round(tax_paid / BACKTEST_PARAMS["initial_balance"] * 100, 2),
            final_balance=round(cash, 2),
            trades=trades,
        )

    def _benchmark(self, prices):  # buy-hold
        return round((prices.iloc[-1] / prices.iloc[0] - 1) * 100, 2)

    # -----------------------------------------------------------------
    def run(self, req: UpdateIndicatorsData) -> Dict[str, Any]:
        self.log.info("▶️ Back-testing %s", self.ticker)
        dt, px, X = self._meta_features(req)
        acts = self._decide(X)
        out = self._simulate(dt, px, acts)
        out["benchmark_pct"] = self._benchmark(px)
        return out


# ──────────────────────────────────────────────────────────────────
def _safe(v):
    if isinstance(v, JSON_SAFE):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v)


def backtest_model(data: UpdateIndicatorsData) -> Dict[str, Any]:
    raw = Backtester(data.stock_ticker, debug=data.__dict__.get("debug", False)).run(
        data
    )
    raw["trades"] = [{k: _safe(v) for k, v in t.items()} for t in raw["trades"]]
    return {k: _safe(v) if k != "trades" else v for k, v in raw.items()}


# standalone demo
if __name__ == "__main__":
    from datetime import date, timedelta

    class _Req(UpdateIndicatorsData):
        stock_ticker: str = "QQQ"
        from_date: str = (date.today() - timedelta(days=365 * 5)).isoformat()
        to_date: str = date.today().isoformat()

    print(json.dumps(backtest_model(_Req()), indent=2))
