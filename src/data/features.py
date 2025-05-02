import pandas as pd
import numpy as np
import ta
import yfinance as yf
from scipy.signal import argrelextrema
from config import (
    BinaryIndicator,
    DateFeatures,
    ExternalDerivedFeatures,
    Indicator,
    GannFeaturs,
    ExternalFeatures,
    Pattern,
)


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_indicators(df)
    df = calculate_derived_indicators(df)
    df = calculate_additional_logic(df)
    df = calculate_gann_features(df)
    df = calculate_patterns(df)
    df = add_external_market_features(df)
    df = add_time_features(df)
    df = calculate_binary_indicators(df)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["Close"], df["High"], df["Low"]

    df[Indicator.RSI.value] = ta.momentum.rsi(close, window=14)
    df[Indicator.RSI_2.value] = ta.momentum.rsi(close, window=2)
    df[Indicator.SMA_50.value] = close.rolling(window=50).mean()
    df[Indicator.EMA_50.value] = close.ewm(span=50, adjust=False).mean()

    macd = ta.trend.macd(close)
    macd_signal = ta.trend.macd_signal(close)
    df[Indicator.MACD.value] = macd
    df[Indicator.MACD_SIGNAL.value] = macd_signal
    df[Indicator.MACD_HIST.value] = macd - macd_signal

    df[Indicator.ATR.value] = ta.volatility.average_true_range(high, low, close)
    df[Indicator.CCI.value] = ta.trend.cci(high, low, close)
    return df


def calculate_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df[BinaryIndicator.RSI_ABOVE_70.value] = (df[Indicator.RSI.value] > 70).astype(int)
    df[BinaryIndicator.RSI_BELOW_30.value] = (df[Indicator.RSI.value] < 30).astype(int)
    df[BinaryIndicator.RSI_CROSS_DOWN_70.value] = (
        (df[Indicator.RSI.value].shift(1) > 70) & (df[Indicator.RSI.value] <= 70)
    ).astype(int)
    df[BinaryIndicator.RSI_CROSS_UP_30.value] = (
        (df[Indicator.RSI.value].shift(1) < 30) & (df[Indicator.RSI.value] >= 30)
    ).astype(int)
    df[BinaryIndicator.BOLLINGER_2PCT_LOWER.value] = (
        close < df[Indicator.BOLLINGER_MIDDLE.value] * 0.98
    ).astype(int)
    df[BinaryIndicator.BOLLINGER_STRONG.value] = (
        (close < df[Indicator.BOLLINGER_LOWER.value])
        | (close > df[Indicator.BOLLINGER_UPPER.value])
    ).astype(int)
    df[BinaryIndicator.RSI_BOLLINGER_STRONG_ABOVE.value] = (
        df[Indicator.RSI.value] > df[Indicator.BOLLINGER_UPPER.value]
    ).astype(int)
    df[BinaryIndicator.RSI_BOLLINGER_STRONG_BELOW.value] = (
        df[Indicator.RSI.value] < df[Indicator.BOLLINGER_LOWER.value]
    ).astype(int)

    return df


def calculate_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df[Indicator.RSI_SLOPE.value] = df[Indicator.RSI.value].diff()

    bb = ta.volatility.BollingerBands(close)
    df[Indicator.BOLLINGER_UPPER.value] = bb.bollinger_hband()
    df[Indicator.BOLLINGER_LOWER.value] = bb.bollinger_lband()
    df[Indicator.BOLLINGER_MIDDLE.value] = bb.bollinger_mavg()
    df[Indicator.BOLLINGER_WIDTH.value] = bb.bollinger_wband()

    df[Indicator.VWAP.value] = (
        volume * (high + low + close) / 3
    ).cumsum() / volume.cumsum()
    df[Indicator.MACD_DELTA.value] = (
        df[Indicator.MACD.value] - df[Indicator.MACD_SIGNAL.value]
    )
    df[Indicator.RSI_DELTA.value] = df[Indicator.RSI.value].diff()
    df[Indicator.MOMENTUM.value] = ta.momentum.roc(close, window=5)
    df[Indicator.MOMENTUM_CHANGE.value] = df[Indicator.MOMENTUM.value].diff()
    df[Indicator.VOLATILITY.value] = ta.volatility.average_true_range(high, low, close)
    df[Indicator.VOLATILITY_CHANGE.value] = df[Indicator.VOLATILITY.value].diff()
    df[Indicator.ROC.value] = ta.momentum.roc(close)
    df[Indicator.WILLIAMS_R.value] = ta.momentum.williams_r(high, low, close)
    df[Indicator.TRIX.value] = ta.trend.trix(close)
    df[Indicator.TSI.value] = ta.momentum.tsi(close)
    df[Indicator.ADX.value] = ta.trend.adx(high, low, close)
    df[Indicator.SHORT_MOMENTUM.value] = close.pct_change(2)
    df[Indicator.TREND_5D.value] = close.pct_change(5)
    df[Indicator.TREND_10D.value] = close.pct_change(10)
    df[Indicator.VOLUME_TRAND.value] = volume.pct_change(3)
    return df


def calculate_additional_logic(df: pd.DataFrame) -> pd.DataFrame:
    df[Indicator.RSI_BOLLINGER_MIDDLE.value] = (
        df[Indicator.RSI.value] - df[Indicator.BOLLINGER_MIDDLE.value]
    )
    df[Indicator.RSI_BOLLINGER_UPPER.value] = (
        df[Indicator.RSI.value] - df[Indicator.BOLLINGER_UPPER.value]
    )
    df[Indicator.RSI_BOLLINGER_LOWER.value] = (
        df[Indicator.RSI.value] - df[Indicator.BOLLINGER_LOWER.value]
    )
    return df


def calculate_gann_features(df: pd.DataFrame) -> pd.DataFrame:
    df[GannFeaturs.ANGLE_1D.value] = df["Close"].diff(1)
    df[GannFeaturs.ANGLE_2D.value] = df["Close"].diff(2) / 2
    (
        df[GannFeaturs.UP_CYCLE_LENGTH.value],
        df[GannFeaturs.DOWN_CYCLE_LENGTH.value],
    ) = get_cycle_times(df)
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

    # External raw features
    df[ExternalDerivedFeatures.SPY_VIX_RATIO.value] = (
        df[ExternalFeatures.SPY_CLOSE.value] / df[ExternalFeatures.VIX_CLOSE.value]
    )
    df[ExternalDerivedFeatures.YIELD_SPREAD_10Y_2Y.value] = (
        df[ExternalFeatures.US10Y_YIELD.value] - df[ExternalFeatures.US2Y_YIELD.value]
    )

    # External derived change features
    df[ExternalDerivedFeatures.SPY_Change.value] = df[
        ExternalFeatures.SPY_CLOSE.value
    ].pct_change()
    df[ExternalDerivedFeatures.VIX_Change.value] = df[
        ExternalFeatures.VIX_CLOSE.value
    ].pct_change()
    df[ExternalDerivedFeatures.US10Y_Change.value] = df[
        ExternalFeatures.US10Y_YIELD.value
    ].diff()
    df[ExternalDerivedFeatures.US2Y_Change.value] = df[
        ExternalFeatures.US2Y_YIELD.value
    ].diff()

    return df


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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df[DateFeatures.DAY_OF_WEEK.value] = df["Date"].dt.dayofweek
    df[DateFeatures.MONTH.value] = df["Date"].dt.month
    return df
