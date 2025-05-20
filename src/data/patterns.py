import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from config import Pattern


def detect_double_top(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    if len(peaks) < 2:
        return 0
    return int(abs(prices[peaks[-1]] - prices[peaks[-2]]) / prices[peaks[-1]] < 0.02)


def detect_double_bottom(prices, order=5):
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(troughs) < 2:
        return 0
    return int(
        abs(prices[troughs[-1]] - prices[troughs[-2]]) / prices[troughs[-1]] < 0.02
    )


def detect_triple_top(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    return int(len(peaks) >= 3)


def detect_triple_bottom(prices, order=5):
    valleys = argrelextrema(prices, np.less, order=order)[0]
    return int(len(valleys) >= 3)


def detect_head_and_shoulders(prices: np.ndarray, order: int = 5) -> int:
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    if len(peaks) < 3:
        return 0
    left, head, right = peaks[0], peaks[1], peaks[2]
    ph, pl, pr = prices[head], prices[left], prices[right]
    if not (ph > pl and ph > pr):
        return 0
    if abs(pl - pr) / ph > 0.05:
        return 0
    return 1


def detect_inverse_head_and_shoulders(prices: np.ndarray, order: int = 5) -> int:
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(troughs) < 3:
        return 0
    left, head, right = troughs[0], troughs[1], troughs[2]
    th, tl, tr = prices[head], prices[left], prices[right]
    if not (th < tl and th < tr):
        return 0
    if abs(tl - tr) / tl > 0.05:
        return 0
    return 1


def detect_cup_and_handle(prices: np.ndarray, order: int = 10) -> int:
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(troughs) < 2:
        return 0
    first, last = troughs[0], troughs[-1]
    if not (
        prices[first] > prices[last]
        and prices[last] < prices[0]
        and prices[last] < prices[-1]
    ):
        return 0
    recent_max = max(prices[last:])
    handle_depth = (recent_max - prices[-1]) / recent_max
    return int(0 < handle_depth < 0.03)


def detect_ascending_triangle(prices: np.ndarray, order: int = 5) -> int:
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(peaks) < 2 or len(troughs) < 2:
        return 0
    if abs(prices[peaks[-1]] - prices[peaks[-2]]) / prices[peaks[-2]] > 0.01:
        return 0
    return int(prices[troughs[-1]] > prices[troughs[-2]])


def detect_descending_triangle(prices: np.ndarray, order: int = 5) -> int:
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(peaks) < 2 or len(troughs) < 2:
        return 0
    if abs(prices[troughs[-1]] - prices[troughs[-2]]) / prices[troughs[-2]] > 0.01:
        return 0
    return int(prices[peaks[-1]] < prices[peaks[-2]])


def detect_symmetrical_triangle(prices: np.ndarray, order: int = 5) -> int:
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(peaks) < 2 or len(troughs) < 2:
        return 0
    return int(
        (prices[peaks[-1]] < prices[peaks[-2]])
        and (prices[troughs[-1]] > prices[troughs[-2]])
    )


def detect_rectangle(prices: np.ndarray, order: int = 5) -> int:
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    troughs = argrelextrema(prices, np.less, order=order)[0]
    if len(peaks) < 2 or len(troughs) < 2:
        return 0
    high_vals = [prices[peaks[-2]], prices[peaks[-1]]]
    low_vals = [prices[troughs[-2]], prices[troughs[-1]]]
    flat_high = abs(high_vals[1] - high_vals[0]) / np.mean(high_vals) < 0.02
    flat_low = abs(low_vals[1] - low_vals[0]) / np.mean(low_vals) < 0.02
    return int(flat_high and flat_low)


def detect_flag(prices: np.ndarray, window: int = 5, threshold: float = 0.02) -> int:
    split = len(prices) - window
    if split < window:
        return 0
    pole = prices[:split]
    flag = prices[split:]
    pole_ret = (pole[-1] - pole[0]) / pole[0]
    flag_range = (max(flag) - min(flag)) / np.mean(flag)
    if pole_ret > threshold and flag_range < threshold:
        return 1
    if pole_ret < -threshold and flag_range < threshold:
        return -1
    return 0


def detect_bullish_engulfing(prices: np.ndarray) -> int:
    return 0


def detect_bearish_engulfing(prices: np.ndarray) -> int:
    return 0


def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # basic multi-peaks/troughs
    df[Pattern.DOUBLE_TOP.value] = close.rolling(50).apply(detect_double_top, raw=True)
    df[Pattern.DOUBLE_BOTTOM.value] = close.rolling(50).apply(
        detect_double_bottom, raw=True
    )
    df[Pattern.TRIPLE_TOP.value] = close.rolling(50).apply(detect_triple_top, raw=True)
    df[Pattern.TRIPLE_BOTTOM.value] = close.rolling(50).apply(
        detect_triple_bottom, raw=True
    )

    # head & shoulders patterns
    df[Pattern.HEAD_SHOULDERS.value] = close.rolling(100).apply(
        detect_head_and_shoulders, raw=True
    )
    df[Pattern.INVERSE_HEAD_AND_SHOULDERS.value] = close.rolling(100).apply(
        detect_inverse_head_and_shoulders, raw=True
    )

    # cup and handle
    df[Pattern.CUP_HANDLE.value] = close.rolling(200).apply(
        detect_cup_and_handle, raw=True
    )

    # triangles
    df[Pattern.ASCENDING_TRIANGLE.value] = close.rolling(50).apply(
        detect_ascending_triangle, raw=True
    )
    df[Pattern.DESCENDING_TRIANGLE.value] = close.rolling(50).apply(
        detect_descending_triangle, raw=True
    )
    df[Pattern.SYMMETRICAL_TRIANGLE.value] = close.rolling(50).apply(
        detect_symmetrical_triangle, raw=True
    )

    # rectangle/consolidation
    df[Pattern.RECTANGLE_PATTERN.value] = close.rolling(50).apply(
        detect_rectangle, raw=True
    )

    # flags (bullish/bearish)
    df[Pattern.BULLISH_FLAG.value] = close.rolling(50).apply(
        lambda x: 1 if detect_flag(x) == 1 else 0, raw=True
    )
    df[Pattern.BEARISH_FLAG.value] = close.rolling(50).apply(
        lambda x: 1 if detect_flag(x) == -1 else 0, raw=True
    )

    # engulfing (placeholders)
    df[Pattern.BULLISH_ENGULFING.value] = 0
    df[Pattern.BEARISH_ENGULFING.value] = 0

    return df
