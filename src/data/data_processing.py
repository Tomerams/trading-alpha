import numpy as np
import pandas as pd
import yfinance as yf

from config import MODEL_PARAMS
from data.features import calculate_features
from routers.routers_entities import UpdateIndicatorsData
from data.data_utilities import get_data, get_exclude_from_scaling
from scipy.signal import argrelextrema


def calculate_bars_to_next_turning(prices: np.ndarray, order: int):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    troughs = argrelextrema(prices, np.less, order=order)[0]
    n = len(prices)
    bars_to_max = np.full(n, np.nan)
    bars_to_min = np.full(n, np.nan)

    for i in range(n):
        future_peaks = peaks[peaks > i]
        future_troughs = troughs[troughs > i]
        if future_peaks.size:
            bars_to_max[i] = future_peaks[0] - i
        if future_troughs.size:
            bars_to_min[i] = future_troughs[0] - i

    return bars_to_max, bars_to_min


def get_indicators_data(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    # 1) Load and prepare raw price data
    df = get_data(
        stock_ticker=request_data.stock_ticker,
        start_date=request_data.start_date,
        end_date=request_data.end_date,
    )
    df = df.rename_axis("Date").reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 2) Feature calculation
    df = calculate_features(df)

    # 3) Target calculation (shift, extrema, bars, trend)
    df = calculate_targets(df)

    # 4) Feature scaling
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = get_exclude_from_scaling()
    to_scale = [c for c in numeric if c not in exclude]
    df[to_scale] = df[to_scale].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    # 5) Cleanup and formatting
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Shift‐based targets
    for tgt in MODEL_PARAMS.get("shift_targets", []):
        name, shift = tgt["name"], tgt["shift"]
        df[f"Target_{name}"] = (df["Close"].shift(shift) - df["Close"]) / df["Close"]

    # 2) Window‐based pct‐change to extreme
    window = MODEL_PARAMS.get("extrema_window", 10)
    highs = df["Close"].rolling(window).max().shift(-window)
    lows = df["Close"].rolling(window).min().shift(-window)
    df["NextLocalMaxPct"] = (highs - df["Close"]) / df["Close"]
    df["NextLocalMinPct"] = (lows - df["Close"]) / df["Close"]

    # 3) Bars until next *true* local max/min
    prices = df["Close"].values
    bars_to_max, bars_to_min = calculate_bars_to_next_turning(prices, order=window)
    df["BarsToNextLocalMax"] = bars_to_max
    df["BarsToNextLocalMin"] = bars_to_min

    return df
