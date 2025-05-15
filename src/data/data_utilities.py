import pandas as pd
from config import MODEL_PARAMS
from data.features import (
    Pattern,
    ExternalDerivedFeatures,
    BinaryIndicator,
    DateFeatures,
)
from routers.routers_entities import UpdateIndicatorsData
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

    return set(
        base + targets + pattern_cols + external_cols + binary_cols + datefeature_cols
    )


CACHE_DIR = "flies/data/cache"


def get_data(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    ticker = request_data.stock_ticker
    start = request_data.start_date
    end = request_data.end_date

    # fname = f"{ticker}_{start}_{end}.csv"
    # fpath = os.path.join(CACHE_DIR, fname)

    # # 1) If cache exists, load it
    # if os.path.exists(fpath):
    #     df = pd.read_csv(fpath, parse_dates=[0], index_col=0)
    #     df.index.name = "Date"
    #     return df

    # 2) Otherwise download
    df = yf.download(ticker, start=start, end=end, interval="1d")
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    # 3) Save to cache (ensure the file has a "Date" column header)
    # os.makedirs(CACHE_DIR, exist_ok=True)
    # df.index.name = "Date"
    # df.to_csv(fpath, index=True)

    return df
