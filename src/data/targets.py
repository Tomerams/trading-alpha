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


def _pct_change(a: pd.Series, shift: int) -> pd.Series:
    """אחוז שינוי קדימה/אחורה – חוסם division-by-zero."""
    return (a.shift(shift) - a) / a.replace(0, np.nan)


def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    מוסיף 11 עמודות Target ל-DataFrame:
    • Target_±N_Days                       – אחוז שינוי עתידי
    • NextLocalMax/MinPct                  – כמה אחוז עד השיא/שפל המקומי
    • BarsToNextLocalMax/Min  (log1p)      – כמה ברים קדימה עד השיא/שפל
    """
    close = df["Close"]

    # 1)  Shifts (Tomorrow, 2_Days, …)
    for s in TRAIN_TARGETS_PARAMS["shift_targets"]:
        df[f"Target_{s['name']}"] = _pct_change(close, s["shift"])

    # 2)  אחוזים לשיא/שפל מקומי בחלון 'extrema_window'
    win = TRAIN_TARGETS_PARAMS.get("extrema_window", 10)
    highs = close.rolling(win).max().shift(-win)
    lows  = close.rolling(win).min().shift(-win)
    df["NextLocalMaxPct"] = (highs - close) / close
    df["NextLocalMinPct"] = (lows  - close) / close

    # 3)  כמות הבר-ים עד השיא/שפל המקומי  →  log1p-normalize
    bars_max, bars_min = calculate_bars_to_next_turning(close.values, order=win)
    df["BarsToNextLocalMax"] = np.log1p(bars_max)  # log1p: log(1+x)
    df["BarsToNextLocalMin"] = np.log1p(bars_min)

    # טיפ קטן: אם יש NaN בסוף הסדרה – ניפטר
    return df.dropna().reset_index(drop=True)
