import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ── BUY / HOLD / SELL ע"פ קוונטילים ─────────────────────────────────────────
def make_action_label_quantile(df, horizon: str = "Target_3_Days", q: float = 0.20):
    ret = df[horizon]
    high = np.quantile(ret, 1 - q)
    low = np.quantile(ret, q)
    return np.where(ret >= high, 2, np.where(ret <= low, 0, 1))

# -*- coding: utf-8 -*-
"""
Action label by significant swing-points (peak / trough prominence).
"""



# ────────────────────────────────────────────────────────────────────
# helper: add ΔtoNextPeak / ΔtoNextTrough columns
# ────────────────────────────────────────────────────────────────────
def _next_peaks_troughs(close: np.ndarray,
                        prominence: float) -> tuple[np.ndarray, np.ndarray]:
    """returns 2 arrays same length as price: price_of_next_peak / next_trough."""
    peaks,   _ = find_peaks(close,   prominence=prominence)
    troughs, _ = find_peaks(-close,  prominence=prominence)

    nxt_pk   = np.full_like(close, np.nan, dtype=float)
    nxt_tr   = np.full_like(close, np.nan, dtype=float)

    # walk backwards so “next” is simple
    p_iter, t_iter = iter(peaks[::-1]), iter(troughs[::-1])
    p, t = next(p_iter, None), next(t_iter, None)
    for i in range(len(close) - 1, -1, -1):
        if p is not None and i <= p: p = next(p_iter, None)
        if t is not None and i <= t: t = next(t_iter, None)
        if p is not None: nxt_pk[i] = close[p]
        if t is not None: nxt_tr[i] = close[t]
    return nxt_pk, nxt_tr


# ────────────────────────────────────────────────────────────────────
def make_action_label_swing(df: pd.DataFrame,
                            atr_col: str = "ATR",
                            prom_mult: float = 1.5,
                            up_thr: float = 0.02,
                            dn_thr: float = -0.02) -> np.ndarray:
    """
    Returns ndarray of {0=SELL,1=HOLD,2=BUY} sized like df
    BUY  – צפי עליה ≥ up_thr אל הפסגה הקרובה.
    SELL – צפי ירידה ≤ dn_thr אל השפל הקרוב.
    """
    price = df["Close"].to_numpy(dtype=float)

    # ── קובעים כמה “משמעותי” פיק: prominence = 1.5 × ATR ממוצע ──
    if atr_col in df.columns:
        atr_mean = np.nanmedian(df[atr_col].to_numpy())
    else:
        # כ-fallback: 1 % מהמחיר
        atr_mean = 0.01 * np.nanmedian(price)
    prom = prom_mult * atr_mean

    next_pk, next_tr = _next_peaks_troughs(price, prom)
    delta_up = (next_pk - price) / price        # יכול להיות nan
    delta_dn = (next_tr - price) / price

    # החזרה לפורמט מלא – nan→0 כדי שחישובי scaling לא יפלו
    df["ΔtoNextPeak"]   = np.nan_to_num(delta_up)
    df["ΔtoNextTrough"] = np.nan_to_num(delta_dn)

    # ── derive action ──
    action = np.where(
        delta_up >= up_thr, 2,
        np.where(delta_dn <= dn_thr, 0, 1)
    ).astype(int)

    return action


def _next_pk_tr(price: np.ndarray, prom: float):
    pk, _ = find_peaks(price,  prominence=prom)
    tr, _ = find_peaks(-price, prominence=prom)
    nxt_pk = np.full_like(price, np.nan);  nxt_tr = np.full_like(price, np.nan)
    p_iter, t_iter = iter(pk[::-1]), iter(tr[::-1]);  p, t = next(p_iter,None), next(t_iter,None)
    for i in range(len(price)-1, -1, -1):
        if p is not None and i <= p: p = next(p_iter, None)
        if t is not None and i <= t: t = next(t_iter, None)
        if p is not None: nxt_pk[i] = price[p]
        if t is not None: nxt_tr[i] = price[t]
    return nxt_pk, nxt_tr

def make_action_label_clean(df: pd.DataFrame,
                            up_thr: float = 0.015,
                            dn_thr: float = -0.015,
                            atr_mult: float = 1.0,
                            atr_col: str = "ATR") -> np.ndarray:
    price = df["Close"].to_numpy(float)
    atr   = df[atr_col].fillna(method="ffill").to_numpy(float)
    prom  = atr_mult * np.nanmedian(atr)          # פיק “משמעותי”
    nxt_pk, nxt_tr = _next_pk_tr(price, prom)

    delta_up = (nxt_pk - price) / price
    delta_dn = (nxt_tr - price) / price

    lbl = np.where(
        delta_up >= up_thr, 2,
        np.where(delta_dn <= dn_thr, 0, 1)
    ).astype(int)
    return lbl