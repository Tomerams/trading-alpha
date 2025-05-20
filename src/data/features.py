import pandas as pd
import numpy as np
import ta
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.trend import EMAIndicator
from config import (
    MODEL_PARAMS,
    BinaryIndicator,
    DateFeatures,
    ExternalDerivedFeatures,
    Indicator,
    GannFeaturs,
    ExternalFeatures,
)
from data.data_utilities import get_data
from data.patterns import calculate_patterns


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_indicators(df)
    df = calculate_derived_indicators(df)
    df = calculate_additional_logic(df)
    df = calculate_gann_features(df)
    df = calculate_patterns(df)
    df = add_external_market_features(df)
    df = add_time_features(df)
    df = calculate_gap_features(df)
    df = add_volume_features(df)
    df = add_volatility_breakout_features(df)
    df = add_trend_crossover_features(df)
    # df = add_price_action_patterns(df)
    # df = add_multitimeframe_features(df)
    df = add_longer_window_indicators(df)
    # df = add_cross_asset_features(df)
    # df = add_event_sentiment_features(df)
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


def get_cycle_times(df: pd.DataFrame):
    ups, downs = [], []
    up = down = 0
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


def calculate_gann_features(df: pd.DataFrame) -> pd.DataFrame:
    df[GannFeaturs.ANGLE_1D.value] = df["Close"].diff(1)
    df[GannFeaturs.ANGLE_2D.value] = df["Close"].diff(2) / 2
    (
        df[GannFeaturs.UP_CYCLE_LENGTH.value],
        df[GannFeaturs.DOWN_CYCLE_LENGTH.value],
    ) = get_cycle_times(df)
    return df


def add_external_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.name == "Date":
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    start = df["Date"].min().strftime("%Y-%m-%d")
    end = df["Date"].max().strftime("%Y-%m-%d")

    def download(ticker, colname):
        ext = get_data(ticker, start, end)
        if isinstance(ext.columns, pd.MultiIndex):
            ext.columns = ext.columns.get_level_values(0)
        return ext.reset_index()[["Date", "Close"]].rename(columns={"Close": colname})

    for col, ticker in {
        ExternalFeatures.VIX_CLOSE.value: "^VIX",
        ExternalFeatures.SPY_CLOSE.value: "SPY",
        ExternalFeatures.US10Y_YIELD.value: "^TNX",
        ExternalFeatures.US2Y_YIELD.value: "^IRX",
    }.items():
        df = df.merge(download(ticker, col), on="Date", how="left")
    df[ExternalDerivedFeatures.SPY_VIX_RATIO.value] = (
        df[ExternalFeatures.SPY_CLOSE.value] / df[ExternalFeatures.VIX_CLOSE.value]
    )
    df[ExternalDerivedFeatures.YIELD_SPREAD_10Y_2Y.value] = (
        df[ExternalFeatures.US10Y_YIELD.value] - df[ExternalFeatures.US2Y_YIELD.value]
    )
    df[ExternalDerivedFeatures.SPY_Change.value] = df[
        ExternalFeatures.SPY_CLOSE.value
    ].pct_change(fill_method=None)
    df[ExternalDerivedFeatures.VIX_Change.value] = df[
        ExternalFeatures.VIX_CLOSE.value
    ].pct_change(fill_method=None)
    df[ExternalDerivedFeatures.US10Y_Change.value] = df[
        ExternalFeatures.US10Y_YIELD.value
    ].diff()
    df[ExternalDerivedFeatures.US2Y_Change.value] = df[
        ExternalFeatures.US2Y_YIELD.value
    ].diff()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # basic date parts
    df[DateFeatures.DAY_OF_WEEK.value] = df["Date"].dt.dayofweek
    df[DateFeatures.MONTH.value] = df["Date"].dt.month

    # additional date parts
    df[DateFeatures.DAY_OF_MONTH.value] = df["Date"].dt.day
    df[DateFeatures.QUARTER.value] = df["Date"].dt.quarter
    df[DateFeatures.DAY_OF_YEAR.value] = df["Date"].dt.dayofyear

    return df


def calculate_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    prev = df["Close"].shift(1)
    df["Gap"] = df["Open"] - prev
    df["Gap_Pct"] = df["Open"] / prev - 1
    df["Gap_Up"] = (df["Gap"] > 0).astype(int)
    df["Gap_Down"] = (df["Gap"] < 0).astype(int)
    df["Gap_Still_Open"] = (
        ((df["Gap"] > 0) & (df["Low"] > prev)) | ((df["Gap"] < 0) & (df["High"] < prev))
    ).astype(int)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    adl = AccDistIndexIndicator(
        df["High"], df["Low"], df["Close"], df["Volume"]
    ).acc_dist_index()
    df["Chaikin_Osc"] = (
        EMAIndicator(adl, window=3, fillna=False).ema_indicator()
        - EMAIndicator(adl, window=10, fillna=False).ema_indicator()
    )
    return df


def add_volatility_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Returns"] = df["Close"].pct_change(fill_method=None)
    df["RollStd_10"] = df["Returns"].rolling(10).std()
    df["RollStd_21"] = df["Returns"].rolling(21).std()
    df["Donchian_Width"] = (
        df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    ) / df["Close"]
    df["Volatility_Breakout"] = (
        df["Close"] > df["High"].shift(1) + df[Indicator.ATR.value]
    ).astype(int)
    return df


def add_trend_crossover_features(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA10_50_Ratio"] = df["EMA_10"] / df[Indicator.EMA_50.value]
    df["EMA50_200_Ratio"] = df[Indicator.EMA_50.value] / df["EMA_200"]
    df["DI_Pos"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"])
    df["DI_Neg"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"])

    return df


def hurst_exponent(ts: pd.Series) -> float:
    lags = range(2, 20)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0


def add_price_action_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # df['Hammer'] = ta.pattern.cdl_hammer(df['Open'], df['High'], df['Low'], df['Close'])
    # df['Shooting_Star'] = ta.pattern.cdl_shooting_star(df['Open'], df['High'], df['Low'], df['Close'])
    # df['Morning_Star'] = ta.pattern.cdl_morning_star(df['Open'], df['High'], df['Low'], df['Close'])
    # df['Evening_Star'] = ta.pattern.cdl_evening_star(df['Open'], df['High'], df['Low'], df['Close'])
    df["Hurst"] = df["Close"].rolling(100).apply(hurst_exponent, raw=False)
    return df


def add_multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    weekly = df_copy.set_index("Date")["Close"].resample("W").last()
    w_ret = weekly.pct_change(fill_method=None).rename("Return_Weekly")
    w_vol = weekly.pct_change(fill_method=None).rolling(4).std().rename("Vol_Weekly")
    df = df.merge(w_ret.reset_index(), on="Date", how="left")
    df = df.merge(w_vol.reset_index(), on="Date", how="left")
    monthly = df_copy.set_index("Date")["Close"].resample("M").last()
    m_ret = monthly.pct_change(fill_method=None).rename("Return_Monthly")
    m_vol = monthly.pct_change(fill_method=None).rolling(3).std().rename("Vol_Monthly")
    df = df.merge(m_ret.reset_index(), on="Date", how="left")
    df = df.merge(m_vol.reset_index(), on="Date", how="left")
    return df


def add_longer_window_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI_30"] = ta.momentum.rsi(df["Close"], window=30)
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["SMA200_Slope"] = df["SMA_200"].diff()
    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    df["SPY_Return"] = df[ExternalFeatures.SPY_CLOSE.value].pct_change(fill_method=None)
    df["Corr_SPY_10"] = (
        df["Close"]
        .pct_change(fill_method=None)
        .rolling(window=10)
        .corr(df["SPY_Return"])
    )
    df["Advance_Decline_Ratio"] = np.nan  # requires broader market data
    return df


def add_event_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Earnings_Surprise"] = np.nan
    df["News_Sentiment"] = np.nan
    df["Put_Call_Ratio"] = np.nan
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
