import numpy as np
import pandas as pd
import yfinance as yf

from config import MODEL_PARAMS
from data.features import calculate_features
from routers.routers_entities import UpdateIndicatorsData
from data.data_utilities import get_exclude_from_scaling


import numpy as np
import pandas as pd
import yfinance as yf
from config import MODEL_PARAMS
from data.features import calculate_features
from data.data_utilities import get_exclude_from_scaling


def get_data(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    # 1) Download and basic clean
    df = yf.download(
        request_data.stock_ticker,
        start=request_data.start_date,
        end=request_data.end_date,
        interval="1d",
    )
    if df.empty:
        raise ValueError(f"No data for {request_data.stock_ticker}")

    df = df.rename_axis("Date").reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = calculate_features(df)

    # 2) Shift-based targets
    for tgt in MODEL_PARAMS.get("shift_targets", []):
        name, shift = tgt["name"], tgt["shift"]
        df[f"Target_{name}"] = (df["Close"].shift(shift) - df["Close"]) / df["Close"]

    # 3) Extrema-based targets
    window = MODEL_PARAMS.get("extrema_window", 10)
    highs = df["Close"].rolling(window).max().shift(-window)
    lows = df["Close"].rolling(window).min().shift(-window)
    df["NextLocalMaxPct"] = (highs - df["Close"]) / df["Close"]
    df["NextLocalMinPct"] = (lows - df["Close"]) / df["Close"]

    # 4) Trend direction
    short_ema = (
        df["Close"].ewm(span=MODEL_PARAMS["trend_ema_short"], adjust=False).mean()
    )
    long_ema = df["Close"].ewm(span=MODEL_PARAMS["trend_ema_long"], adjust=False).mean()
    df["TrendDirection"] = 0
    df.loc[short_ema > long_ema, "TrendDirection"] = 1
    df.loc[short_ema < long_ema, "TrendDirection"] = -1

    # 5) Feature scaling using utility function
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = get_exclude_from_scaling()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    df[feature_cols] = df[feature_cols].apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # 6) Clean up and format
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df
