import pandas as pd
import numpy as np
import yfinance as yf
import ta
from scipy.signal import argrelextrema

from config import Indicator, Pattern, GannFeaturs, ExternalFeatures


def get_data(stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError(f"No data retrieved for {stock_ticker}.")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns:
        raise KeyError("'Close' column missing in dataset.")

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    df = calculate_indicators(df)

    df["Target_Tomorrow"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["Target_3_Days"] = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
    df["Target_Next_Week"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df["Close_Normal"] = df["Close"]

    exclude_cols = [
        "Date",
        "Target_Tomorrow",
        "Target_3_Days",
        "Target_Next_Week",
        "Close",
    ]

    feature_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    df[feature_cols] = df[feature_cols].apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df["Month"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.weekday

    df.dropna(inplace=True)

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(f"data/{stock_ticker}_indicators_data.csv", index=False)

    return df


def calculate_indicators(
    df: pd.DataFrame, add_external_features: bool = True
) -> pd.DataFrame:
    df[Indicator.RSI.value] = ta.momentum.rsi(df["Close"], window=14)
    df[Indicator.RSI_2.value] = ta.momentum.rsi(df["Close"], window=2)
    df[Indicator.RSI_ABOVE_70.value] = (df[Indicator.RSI.value] > 70).astype(int)
    df[Indicator.RSI_BELOW_30.value] = (df[Indicator.RSI.value] < 30).astype(int)
    df[Indicator.SMA_50.value] = df["Close"].rolling(window=50).mean()
    df[Indicator.EMA_50.value] = df["Close"].ewm(span=50, adjust=False).mean()
    df[Indicator.MACD.value] = ta.trend.macd(
        df["Close"], window_slow=26, window_fast=12
    )
    df[Indicator.MACD_SIGNAL.value] = ta.trend.macd_signal(
        df["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    df[Indicator.MACD_HIST.value] = (
        df[Indicator.MACD.value] - df[Indicator.MACD_SIGNAL.value]
    )

    if "High" in df.columns and "Low" in df.columns:
        df[Indicator.ATR.value] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], window=14
        )
        df[Indicator.CCI.value] = ta.trend.cci(
            df["High"], df["Low"], df["Close"], window=20
        )

    df = calculate_patterns(df)
    df = calculate_gann_features(df)
    df = calculate_fibonacci_levels(df)

    if add_external_features:
        df = add_external_market_features(df)

    return df


def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df[Pattern.DOUBLE_TOP.value] = (
        df["Close"].rolling(50).apply(detect_double_top, raw=True)
    )
    df[Pattern.TRIPLE_TOP.value] = (
        df["Close"].rolling(50).apply(detect_triple_top, raw=True)
    )
    df[Pattern.TRIPLE_BOTTOM.value] = (
        df["Close"].rolling(50).apply(detect_triple_bottom, raw=True)
    )
    return df


def calculate_gann_features(df: pd.DataFrame) -> pd.DataFrame:
    df[GannFeaturs.ANGLE_1D.value] = df["Close"] - df["Close"].shift(1)
    df[GannFeaturs.ANGLE_2D.value] = (df["Close"] - df["Close"].shift(2)) / 2
    (
        df[GannFeaturs.UP_CYCLE_LENGTH.value],
        df[GannFeaturs.DOWN_CYCLE_LENGTH.value],
    ) = get_cycle_times(df)
    return df


def calculate_fibonacci_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    highest = df["Close"].rolling(window=window).max()
    lowest = df["Close"].rolling(window=window).min()
    range_ = highest - lowest

    df[GannFeaturs.FIB_23_ABOVE.value] = (
        df["Close"] > (highest - 0.236 * range_)
    ).astype(int)
    df[GannFeaturs.FIB_50_ABOVE.value] = (
        df["Close"] > (highest - 0.5 * range_)
    ).astype(int)
    return df


def add_external_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])

    def download(ticker, colname):
        ext = yf.download(
            ticker, start=df["Date"].min(), end=df["Date"].max(), interval="1d"
        )
        ext = ext.copy()
        if isinstance(ext.columns, pd.MultiIndex):
            ext.columns = ext.columns.get_level_values(0)
        ext.reset_index(inplace=True)
        ext = ext[["Date", "Close"]].rename(columns={"Close": colname})
        return ext

    vix = download("^VIX", ExternalFeatures.VIX_CLOSE.value)
    spy = download("SPY", ExternalFeatures.SPY_CLOSE.value)
    us10y = download("^TNX", ExternalFeatures.US10Y_YIELD.value)
    us2y = download("^IRX", ExternalFeatures.US2Y_YIELD.value)

    for ext_df in [vix, spy, us10y, us2y]:
        df = pd.merge(df, ext_df, on="Date", how="left")

    df[ExternalFeatures.SPY_VIX_RATIO.value] = (
        df[ExternalFeatures.SPY_CLOSE.value] / df[ExternalFeatures.VIX_CLOSE.value]
    )
    df[ExternalFeatures.YIELD_SPREAD_10Y_2Y.value] = (
        df[ExternalFeatures.US10Y_YIELD.value] - df[ExternalFeatures.US2Y_YIELD.value]
    )
    return df


# Patterns helpers
def detect_double_top(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    if len(peaks) < 2:
        return 0
    return int(abs(prices[peaks[-1]] - prices[peaks[-2]]) / prices[peaks[-1]] < 0.02)


def detect_triple_top(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    return int(len(peaks) >= 3)


def detect_triple_bottom(prices, order=5):
    valleys = argrelextrema(prices, np.less, order=order)[0]
    return int(len(valleys) >= 3)


def get_cycle_times(df: pd.DataFrame):
    up, down = 0, 0
    ups, downs = [], []

    for change in df["Close"].diff():
        if change > 0:
            up += 1
            down = 0
        elif change < 0:
            down += 1
            up = 0
        ups.append(up)
        downs.append(down)

    return ups, downs


if __name__ == "__main__":
    CONFIG = {
        "stock_ticker": "TQQQ",
        "start_date": "2022-06-06",
        "end_date": "2025-01-01",
    }
    get_data(**CONFIG)
