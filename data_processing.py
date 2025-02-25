import ta.momentum
import yfinance as yf
import ta.volatility
import ta.trend
import ta
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

from config import Indicator, Pattern


def get_data(stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError(f"No data retrieved for {stock_ticker}. Check the stock_ticker symbol and date range.")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns:
        raise KeyError(
            f"'Close' column is missing in the dataset for {stock_ticker}. Available columns: {df.columns.tolist()}"
        )

    df.dropna(subset=["Close"], inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = calculate_indicators(df)
    forecast_period = 7
    df["target"] = (df["Close"].shift(-forecast_period) - df["Close"]) / df["Close"]

    df.dropna(inplace=True)

    df.to_csv(f"data/{stock_ticker}_indicators_data.csv")

    return df


def detect_double_top(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    if len(peaks) < 2:
        return 0
    peak1, peak2 = peaks[-2], peaks[-1]
    if abs(prices[peak1] - prices[peak2]) / prices[peak1] < 0.02:
        return 1
    return 0


def detect_head_and_shoulders(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    valleys = argrelextrema(prices, np.less, order=order)[0]
    if len(peaks) < 3 or len(valleys) < 2:
        return 0
    left_shoulder, head, right_shoulder = peaks[-3], peaks[-2], peaks[-1]
    neckline = valleys[-1]
    if (
        prices[head] > prices[left_shoulder]
        and prices[head] > prices[right_shoulder]
        and prices[left_shoulder] > prices[neckline]
        and prices[right_shoulder] > prices[neckline]
    ):
        return 1
    return 0


def detect_cup_and_handle(prices, order=10):
    min_idx = argrelextrema(prices, np.less, order=order)[0]
    if len(min_idx) < 1:
        return 0
    min_val = prices[min_idx[-1]]
    max_val = prices[-1]
    if max_val > prices[0] and (max_val - min_val) / min_val > 0.1 and min_idx[-1] < len(prices) * 0.7:
        return 1
    return 0


def detect_triple_top(prices, order=5):
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    if len(peaks) < 3:
        return 0
    if (
        abs(prices[peaks[-3]] - prices[peaks[-2]]) / prices[peaks[-3]] < 0.02
        and abs(prices[peaks[-2]] - prices[peaks[-1]]) / prices[peaks[-2]] < 0.02
    ):
        return 1
    return 0


def detect_triple_bottom(prices, order=5):
    valleys = argrelextrema(prices, np.less, order=order)[0]
    if len(valleys) < 3:
        return 0
    if (
        abs(prices[valleys[-3]] - prices[valleys[-2]]) / prices[valleys[-3]] < 0.02
        and abs(prices[valleys[-2]] - prices[valleys[-1]]) / prices[valleys[-2]] < 0.02
    ):
        return 1
    return 0


def detect_bullish_engulfing(df, index):
    if isinstance(index, pd.Timestamp):
        index = df.index.get_loc(index)

    if index < 1:
        return 0

    prev_candle = df.iloc[index - 1]
    curr_candle = df.iloc[index]

    if (
        prev_candle["Close"] < prev_candle["Open"]  # Previous candle is red
        and curr_candle["Close"] > curr_candle["Open"]  # Current candle is green
        and curr_candle["Close"] > prev_candle["Open"]
        and curr_candle["Open"] < prev_candle["Close"]
    ):
        return 1
    return 0


def detect_bearish_engulfing(df, index):
    if isinstance(index, pd.Timestamp):
        index = df.index.get_loc(index)

    if index < 1:
        return 0
    prev = df.iloc[index - 1]
    curr = df.iloc[index]
    if (
        prev["Close"] > prev["Open"]
        and curr["Close"] < curr["Open"]
        and curr["Open"] > prev["Close"]
        and curr["Close"] < prev["Open"]
    ):
        return 1
    return 0


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df[Indicator.RSI.value] = ta.momentum.rsi(df["Close"], window=14)
    df[Indicator.RSI_2.value] = ta.momentum.rsi(df["Close"], window=2)
    df[Indicator.RSI_ABOVE_70.value] = (df[Indicator.RSI.value] > 70).astype(int)
    df[Indicator.RSI_BELOW_30.value] = (df[Indicator.RSI.value] < 30).astype(int)
    df[Indicator.RSI_CROSS_DOWN_70.value] = np.where((df[Indicator.RSI.value].shift(1) > 70) & (df[Indicator.RSI.value] <= 70), 1, 0)
    df[Indicator.RSI_CROSS_UP_30.value] = np.where((df[Indicator.RSI.value].shift(1) < 30) & (df[Indicator.RSI.value] >= 30), 1, 0)
    df[Indicator.SMA_50.value] = df["Close"].rolling(window=50).mean()
    df[Indicator.EMA_50.value] = df["Close"].ewm(span=50, adjust=False).mean()
    df[Indicator.RSI_2_ABOVE_90.value] = (df[Indicator.RSI_2.value] > 90).astype(int)
    df[Indicator.RS_2_BELOW_10.value] = (df[Indicator.RSI_2.value] < 10).astype(int)
    df[Indicator.RSI_2_CROSS_90.value] = np.where(
        (df[Indicator.RSI_2.value].shift(1) < 90) & (df[Indicator.RSI_2.value] >= 90), 1, 0
    )
    df[Indicator.RSI_2_CROSS_10.value] = np.where(
        (df[Indicator.RSI_2.value].shift(1) > 10) & (df[Indicator.RSI_2.value] <= 10), 1, 0
    )
    df[Indicator.RSI_SLOPE.value] = df[Indicator.RSI.value].diff()
    df[Indicator.RSI_DIVERGENCE.value] = (
        (df[Indicator.RSI.value].shift(1) > df[Indicator.RSI.value].shift(2))
        & (df[Indicator.RSI.value] < df[Indicator.RSI.value].shift(1))
    ).astype(int)
    df[Indicator.BOLLINGER_MIDDLE.value] = df["Close"].rolling(window=20).mean()
    df[Indicator.BOLLINGER_UPPER.value] = df[Indicator.BOLLINGER_MIDDLE.value] + (
        df["Close"].rolling(window=20).std() * 2
    )
    df[Indicator.BOLLINGER_LOWER.value] = df[Indicator.BOLLINGER_MIDDLE.value] - (
        df["Close"].rolling(window=20).std() * 2
    )
    df[Indicator.BOLINGER_STRONG.value] = (df["Close"] > df[Indicator.BOLLINGER_UPPER.value]).astype(int)
    df[Indicator.BOLLINGER_2PCT_LOWER.value] = (df["Close"] < df[Indicator.BOLLINGER_LOWER.value] * 1.02).astype(int)
    df[Indicator.RSI_SLOPE.value] = df[Indicator.RSI.value].diff()
    df[Indicator.RSI_DIVERGENCE.value] = (
        (df[Indicator.RSI.value].shift(1) > df[Indicator.RSI.value].shift(2))
        & (df[Indicator.RSI.value] < df[Indicator.RSI.value].shift(1))
    ).astype(int)
    df[Indicator.MOMENTUM.value] = df["Close"].diff()
    df[Indicator.VOLATILITY.value] = df["Close"].rolling(window=20).std()
    df[Indicator.CCI.value] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=20)
    df[Indicator.ROC.value] = ta.momentum.roc(df["Close"], window=12)
    df[Indicator.WILLIAMS_R.value] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"], lbp=14)
    df[Indicator.TRIX.value] = ta.trend.trix(df["Close"], window=15)
    df[Indicator.TSI.value] = ta.momentum.tsi(df["Close"], window_slow=25, window_fast=13)
    df[Indicator.MACD.value] = ta.trend.macd(df["Close"], window_slow=26, window_fast=12)
    df[Indicator.MACD_SIGNAL.value] = ta.trend.macd_signal(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df[Indicator.MACD_HIST.value] = df[Indicator.MACD.value] - df[Indicator.MACD_SIGNAL.value]
    df[Indicator.ADX.value] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df[Indicator.VWAP.value] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df[Indicator.ATR.value] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df[Indicator.STOCHASTIC_K.value] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    df[Indicator.STOCHASTIC_D.value] = df[Indicator.STOCHASTIC_K.value].rolling(window=3).mean()
    df[Indicator.STOCHASTIC_RSI.value] = ta.momentum.stochrsi(df["Close"], window=14, smooth1=3, smooth2=3)
    df[Pattern.DOUBLE_TOP.value] = df["Close"].rolling(50).apply(detect_double_top, raw=True)
    df[Pattern.HEAD_SHOULDERS.value] = df["Close"].rolling(50).apply(detect_head_and_shoulders, raw=True)
    df[Pattern.CUP_HANDLE.value] = df["Close"].rolling(50).apply(detect_cup_and_handle, raw=True)
    df[Pattern.TRIPLE_TOP.value] = df["Close"].rolling(50).apply(detect_triple_top, raw=True)
    df[Pattern.TRIPLE_BOTTOM.value] = df["Close"].rolling(50).apply(detect_triple_bottom, raw=True)
    df[Pattern.BULLISH_ENGULFING.value] = df.apply(lambda row: detect_bullish_engulfing(df, row.name), axis=1)
    df[Pattern.BEARISH_ENGULFING.value] = df.apply(lambda row: detect_bearish_engulfing(df, row.name), axis=1)
    return df


if __name__ == "__main__":
    CONFIG = {
        "stock_ticker": "TQQQ",
        "start_date": "2012-06-06",
        "end_date": "2025-01-01",
    }

    get_data(**CONFIG)
