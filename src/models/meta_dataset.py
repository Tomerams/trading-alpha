import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from config.meta_data_config import META_PARAMS


def create_meta_ai_dataset(
    preds: np.ndarray,
    true_vals: np.ndarray,
    target_cols: list[str],
    buy_threshold: float = None,
    sell_threshold: float = None,
    peak_window: int = 5,
    peak_prominence: float = 0.01,
) -> pd.DataFrame:
    buy_thr = buy_threshold or META_PARAMS["buy_threshold"]
    sell_thr = sell_threshold or META_PARAMS["sell_threshold"]

    df = pd.DataFrame(preds, columns=[f"Pred_{c}" for c in target_cols])

    actual = true_vals[:, target_cols.index("Target_Tomorrow")]
    df["Action"] = np.where(
        actual > buy_thr, 2, np.where(actual < sell_thr, 0, 1)
    ).astype(int)

    series = pd.Series(actual).reset_index(drop=True)
    w = peak_window
    df["is_local_max"] = (
        series == series.rolling(2 * w + 1, center=True).max()
    ).astype(int)
    df["is_local_min"] = (
        series == series.rolling(2 * w + 1, center=True).min()
    ).astype(int)

    peaks, _ = find_peaks(series, distance=w, prominence=peak_prominence)
    troughs, _ = find_peaks(-series, distance=w, prominence=peak_prominence)

    df["is_peak"] = 0
    df.loc[peaks, "is_peak"] = 1
    df["is_trough"] = 0
    df.loc[troughs, "is_trough"] = 1

    return df
