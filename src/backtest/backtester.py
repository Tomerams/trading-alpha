import logging, time, math

logging.getLogger("matplotlib").setLevel(logging.WARNING)

from typing import Any, Dict, List, Tuple
from collections import Counter

import numpy as np, pandas as pd, torch

from config.backtest_config import BACKTEST_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model
from models.meta_loader import load_meta_model
from routers.routers_entities import UpdateIndicatorsData

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Backtester:
    def __init__(self, ticker: str, debug: bool = False):
        self.ticker = ticker
        self.debug = debug
        self.logger = self._create_logger()
        self.model, self.scaler, self.features, self.seq_len = self._load_deep_base()
        self.meta_base, self.meta_clf = self._load_meta()
        if self.debug:
            self._debug_label_distribution()

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Backtester[{self.ticker}]")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    def _load_deep_base(self) -> Tuple[torch.nn.Module, Any, List[str], int]:
        self.logger.info("Loading deep base model...")
        t0 = time.time()
        model, scaler, features = load_model(self.ticker, MODEL_TRAINER_PARAMS["model_type"])
        seq_len = MODEL_TRAINER_PARAMS["seq_len"]
        self.logger.info("Deep base loaded (%d features, seq_len=%d) in %.2fs", len(features), seq_len, time.time() - t0)
        return model, scaler, features, seq_len

    def _load_meta(self) -> Tuple[Dict[str, Any], Any]:
        self.logger.info("Loading meta-model...")
        t0 = time.time()
        try:
            pipeline = load_meta_model(META_PARAMS["meta_model_path"])
            base, clf = pipeline.get("base", {}), pipeline.get("meta")
            self.logger.info("Meta loaded (%d base models) in %.2fs", len(base), time.time() - t0)
            return base, clf
        except FileNotFoundError as e:
            self.logger.warning("Meta-model skipped: %s", e)
            return {}, None

    def _debug_label_distribution(self) -> None:
        req = UpdateIndicatorsData(stock_ticker=self.ticker, start_date="2000-01-01", end_date="2024-12-31")
        df = get_indicators_data(req).dropna()
        thr = 0.02
        labels = df["Target_3_Days"].apply(lambda x: 2 if x > thr else (0 if x < -thr else 1))
        cnt = Counter(labels)
        print(f"\n### LABELS DISTRIBUTION ### {dict(cnt)}\n")
        self.logger.info("Label distribution 0/1/2 = %s", cnt)
        if plt:
            plt.bar(cnt.keys(), cnt.values())
            plt.title("Label Distribution (0=SELL,1=HOLD,2=BUY)")
            plt.show()

    def _prepare_data(self, request_data: Any) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        df = get_indicators_data(request_data).dropna().reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if len(df) < self.seq_len:
            raise ValueError("Not enough data to backtest.")
        n = len(df) - self.seq_len
        seq = np.stack([df[self.features].iloc[i : i + self.seq_len].values for i in range(n)])
        dates = df["Date"].iloc[self.seq_len :].reset_index(drop=True)
        prices = df["Close"].iloc[self.seq_len :].reset_index(drop=True)
        flat = seq.reshape(-1, seq.shape[2])
        scaled = self.scaler.transform(flat).reshape(seq.shape)
        return dates, prices, scaled
    

    def _infer(self, xs: np.ndarray) -> np.ndarray:
        if xs.size == 0:
            return np.empty((0, len(TRAIN_TARGETS_PARAMS["target_cols"])))
        with torch.no_grad():
            preds = self.model(torch.tensor(xs, dtype=torch.float32)).cpu().numpy()
        if self.debug:
            mean_vec = preds.mean(axis=0)
            argmax_hist = Counter(np.argmax(preds, axis=1))
            # ---- שורה אחת ברורה ----
            print(f"\n### DEBUG SUMMARY ###\n"
                f"Mean pred vec  : {np.round(mean_vec,4).tolist()}\n"
                f"Argmax histogram: {dict(argmax_hist)}\n"
                f"####################\n")
        return preds


    def _simulate(self, dates: pd.Series, prices: pd.Series, preds: np.ndarray) -> Tuple[List[Dict], Dict]:
        cash = BACKTEST_PARAMS["initial_balance"]
        shares = 0
        trades: List[Dict] = []
        for date, price, row in zip(dates, prices, preds):
            action = int(np.argmax(row)) if preds.size else 1
            if action == 2 and shares == 0:
                shares = math.floor(cash / price)
                cash -= shares * price
                trades.append({"Date": date, "Type": "BUY", "Shares": shares, "Cash": cash})
            elif action == 0 and shares > 0:
                cash += shares * price
                trades.append({"Date": date, "Type": "SELL", "Shares": shares, "Cash": cash})
                shares = 0
        if shares:
            cash += shares * prices.iloc[-1]
            trades.append({"Date": dates.iloc[-1], "Type": "SELL_EOD", "Shares": shares, "Cash": cash})
        stats = {"net_profit_pct": (cash / BACKTEST_PARAMS["initial_balance"] - 1) * 100, "final_balance": cash}
        return trades, stats

    def run(self, request_data: Any) -> Dict[str, Any]:
        self.logger.info("▶️ Backtest start for %s", self.ticker)
        dates, prices, xs = self._prepare_data(request_data)
        preds = self._infer(xs)
        trades, stats = self._simulate(dates, prices, preds)
        return {**stats, "trades": trades}


def backtest_model(request_data: Any) -> Dict[str, Any]:
    return Backtester(request_data.stock_ticker, debug=request_data.__dict__.get("debug", False)).run(request_data)
