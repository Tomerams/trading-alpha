
import numpy as np
import pandas as pd
from config.model_trainer_config import TRAIN_TARGETS_PARAMS
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



def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    for tgt in TRAIN_TARGETS_PARAMS.get("shift_targets", []):
        name, shift = tgt["name"], tgt["shift"]
        df[f"Target_{name}"] = (df["Close"].shift(shift) - df["Close"]) / df["Close"]

    window = TRAIN_TARGETS_PARAMS.get("extrema_window", 10)
    highs = df["Close"].rolling(window).max().shift(-window)
    lows = df["Close"].rolling(window).min().shift(-window)
    df["NextLocalMaxPct"] = (highs - df["Close"]) / df["Close"]
    df["NextLocalMinPct"] = (lows - df["Close"]) / df["Close"]

    prices = df["Close"].values
    bars_to_max, bars_to_min = calculate_bars_to_next_turning(prices, order=window)
    df["BarsToNextLocalMax"] = bars_to_max
    df["BarsToNextLocalMin"] = bars_to_min

    return df