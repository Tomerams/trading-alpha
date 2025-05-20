import os
import pandas as pd
from config import (
    MODEL_PARAMS,
    Gaps,
    Pattern,
    BinaryIndicator,
    ExternalDerivedFeatures,
    DateFeatures,
)
import yfinance as yf


def get_exclude_from_scaling() -> set:
    """
    Build and return a set of column names to exclude from feature scaling,
    based on MODEL_PARAMS and enums from data.features.
    """
    base = ["Date", "Close"]
    # shift-based targets
    targets = MODEL_PARAMS.get("target_cols", [])
    # pattern features
    pattern_cols = [p.value for p in Pattern]
    # external derived features
    external_cols = [
        ExternalDerivedFeatures.SPY_VIX_RATIO.value,
        ExternalDerivedFeatures.YIELD_SPREAD_10Y_2Y.value,
    ]
    # binary indicators
    binary_cols = [i.value for i in BinaryIndicator]
    # date-related features
    datefeature_cols = [d.value for d in DateFeatures]

    gap_flag_cols = [
        Gaps.GAP_UP.value,
        Gaps.GAP_DOWN.value,
        Gaps.GAP_STILL_OPEN.value,
    ]
    return set(
        base
        + targets
        + pattern_cols
        + external_cols
        + binary_cols
        + datefeature_cols
        + gap_flag_cols
    )


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "files", "data")


def get_data(stock_ticker, start_date, end_date) -> pd.DataFrame:
    fname = f"{stock_ticker}_{start_date}_{end_date}.pkl"
    fpath = os.path.join(CACHE_DIR, fname)

    # 1) If cache exists, load it
    if os.path.exists(fpath):
        return pd.read_pickle(fpath)

    # 2) Otherwise download
    df = yf.download(
        stock_ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True
    )
    if df.empty:
        raise ValueError(f"No data for {stock_ticker}")

    # 3) Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.to_pickle(df, fpath)

    return df
