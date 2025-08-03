import pandas as pd
import numpy as np
import ta
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.trend import EMAIndicator

# ────────────────────────────────────────────────────────────
# FULL FEATURE PIPELINE – single source of truth
# ────────────────────────────────────────────────────────────

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """run the full feature-generation pipeline – order matters!"""
    df = df.copy()

    df = _calculate_indicators(df)
    df = _calculate_derived_indicators(df)
    df = _calculate_logic_combos(df)
    df = _calculate_gann_features(df)
    df = _calculate_patterns(df)
    df = _add_market_context(df)          # ← תוספת הקריפטו בפונקציה זו
    df = _add_calendar_features(df)
    df = _calculate_gap_features(df)
    df = _add_volume_features(df)
    df = _add_volatility_breakout(df)
    df = _add_trend_crossover(df)
    df = _add_long_window_features(df)
    df = _calculate_binary_signals(df)
    # df = _add_lags(df)                  # CAN-DROP – כבוי כברירת-מחדל

    cols = ["Date"] + [c for c in df.columns if c != "Date"]
    return df[cols]

# ────────────────────────────────────────────────────────────
# 1) CORE TECHNICALS
# ────────────────────────────────────────────────────────────
def _calculate_indicators(df):
    close, high, low = df["Close"], df["High"], df["Low"]
    df["RSI"]      = ta.momentum.rsi(close, 14)
    df["RSI_2"]    = ta.momentum.rsi(close, 2)
    df["SMA_50"]   = close.rolling(50).mean()
    df["EMA_50"]   = close.ewm(span=50, adjust=False).mean()

    df["MACD"]        = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    df["ATR"] = ta.volatility.average_true_range(high, low, close)
    df["CCI"] = ta.trend.cci(high, low, close)
    df["Volatility"] = df["ATR"]
    return df

# ────────────────────────────────────────────────────────────
# 2) SECOND-ORDER & DELTAS
# ────────────────────────────────────────────────────────────
def _calculate_derived_indicators(df):
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    df["RSI_slope"]    = df["RSI"].diff()
    df["MACD_Delta"]   = df["MACD"] - df["MACD_Signal"]
    df["RSI_Delta"]    = df["RSI"].diff()
    df["Momentum"]     = ta.momentum.roc(close, 5)
    df["Momentum_Change"] = df["Momentum"].diff()

    bb = ta.volatility.BollingerBands(close)
    df["Bollinger_Upper"]  = bb.bollinger_hband()
    df["Bollinger_Lower"]  = bb.bollinger_lband()
    df["Bollinger_Middle"] = bb.bollinger_mavg()
    df["Bollinger_Width"]  = bb.bollinger_wband()

    df["VWAP"] = (vol * (high + low + close) / 3).cumsum() / vol.cumsum()
    df["Volatility_Change"] = df["Volatility"].diff()

    df["ROC"]         = ta.momentum.roc(close)
    df["Williams_%R"] = ta.momentum.williams_r(high, low, close)
    df["TRIX"]        = ta.trend.trix(close)
    df["TSI"]         = ta.momentum.tsi(close)
    df["ADX"]         = ta.trend.adx(high, low, close)
    df["Short_Momentum"] = close.pct_change(2)
    df["Trend_5D"]       = close.pct_change(5)
    df["Trend_10D"]      = close.pct_change(10)
    df["Volume_Trend"]   = vol.pct_change(3)
    df["High_Low_Range"] = (high - low) / close

    df["RSI_Bollinger_Middle"] = df["RSI"] - df["Bollinger_Middle"]
    df["RSI_Bollinger_Upper"]  = df["RSI"] - df["Bollinger_Upper"]
    df["RSI_Bollinger_Lower"]  = df["RSI"] - df["Bollinger_Lower"]
    return df

# ────────────────────────────────────────────────────────────
def _calculate_logic_combos(df): return df  # CAN-DROP placeholder
# ────────────────────────────────────────────────────────────

def _calculate_gann_features(df):
    df["Gann_Angle_1D"] = df["Close"].diff()
    df["Gann_Angle_2D"] = df["Close"].diff(2) / 2
    ups, downs = [], []
    up = down = 0
    for ch in df["Close"].diff():
        if ch > 0: up += 1; down = 0
        elif ch < 0: down += 1; up = 0
        ups.append(up); downs.append(down)
    df["Gann_Up_Cycle_Length"]   = ups
    df["Gann_Down_Cycle_Length"] = downs
    return df

# -----------------------------------------------------------------
def _calculate_patterns(df):
    try:
        from data.patterns import calculate_patterns as _calc
        return _calc(df)
    except Exception:
        pats = ["double_top","double_bottom","triple_top","triple_bottom",
                "head_and_shoulders","inverse_head_and_shoulders",
                "cup_and_handle","ascending_triangle","descending_triangle",
                "symmetrical_triangle","rectangle_pattern","bullish_flag",
                "bearish_flag","bullish_engulfing","bearish_engulfing"]
        for c in pats:
            df[c] = 0
        return df
# -----------------------------------------------------------------
# ⬇️  MARKET CONTEXT  (כולל קריפטו)  # NEW
# -----------------------------------------------------------------

_BASE_TICKERS = {
    "VIX_Close":     "^VIX",
    "SPY_Close":     "SPY",
    "US10Y_Yield":   "^TNX",
    "US2Y_Yield":    "^IRX",
    "GOLD_Close":    "GC=F",   # Gold futures
    "OIL_Close":     "CL=F",   # WTI Crude
}

_CRYPTO_TICKERS = {            # NEW – גמיש להרחבה
    "BTC_Close": "BTC-USD",
    "ETH_Close": "ETH-USD",
    # פשוט הוסף כאן: "SOL_Close": "SOL-USD", ...
}

def _download_external(ticker, col, start, end):
    from data.data_utilities import get_data
    ext = get_data(ticker, start, end)
    if isinstance(ext.columns, pd.MultiIndex):
        ext.columns = ext.columns.get_level_values(0)
    return ext.reset_index()[["Date", "Close"]].rename(columns={"Close": col})

def _add_market_context(df):
    df = df.copy()
    if df.index.name == "Date":
        df = df.reset_index()

    start, end = df["Date"].min().strftime("%Y-%m-%d"), df["Date"].max().strftime("%Y-%m-%d")
    all_ticks = {**_BASE_TICKERS, **_CRYPTO_TICKERS}           # NEW

    for col, tk in all_ticks.items():
        try:
            ext = _download_external(tk, col, start, end)
            df  = df.merge(ext, on="Date", how="left")
        except Exception:
            df[col] = np.nan  # אם ההורדה נכשלה (למשל לפני שהנכס קיים)

    # ---- ratios / changes (מחולל כללי)  # NEW
    for col in all_ticks:
        if col.endswith("_Close"):
            asset = col.split("_")[0]                 # SPY / BTC / GOLD …
            df[f"{asset}_Change"] = df[col].pct_change(fill_method=None)

    if "SPY_Close" in df:
        for col in all_ticks:
            if col != "SPY_Close" and col.endswith("_Close"):
                asset = col.split("_")[0]
                df[f"{asset}_SPY_RATIO"] = df[col] / df["SPY_Close"]

    # yield spread
    if {"US10Y_Yield","US2Y_Yield"}.issubset(df.columns):
        df["YIELD_SPREAD_10Y_2Y"] = df["US10Y_Yield"] - df["US2Y_Yield"]
        df["US10Y_Yield_Change"]  = df["US10Y_Yield"].diff()
        df["US2Y_Yield_Change"]   = df["US2Y_Yield"].diff()

    # relative strength (10-day) מול SPY
    if "SPY_Close" in df:
        df["Rel_Strength_SPY"]     = df["Close"] / df["SPY_Close"]
        df["Rel_Strength_SPY_10d"] = df["Rel_Strength_SPY"].pct_change(10)

    return df

# -----------------------------------------------------------------
def _add_calendar_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day_Of_Week"]  = df["Date"].dt.dayofweek
    df["Month"]        = df["Date"].dt.month
    df["Day_of_Month"] = df["Date"].dt.day
    df["Quarter"]      = df["Date"].dt.quarter
    df["Day_of_Year"]  = df["Date"].dt.dayofyear

    df["DoW_sin"]   = np.sin(2*np.pi*df["Day_Of_Week"]/7)
    df["DoW_cos"]   = np.cos(2*np.pi*df["Day_Of_Week"]/7)
    df["Month_sin"] = np.sin(2*np.pi*df["Month"]/12)
    df["Month_cos"] = np.cos(2*np.pi*df["Month"]/12)
    return df
# -----------------------------------------------------------------
def _calculate_gap_features(df):
    prev = df["Close"].shift()
    df["Gap"]      = df["Open"] - prev
    df["Gap_Pct"]  = df["Open"] / prev - 1
    df["Gap_Up"]   = (df["Gap"] > 0).astype(int)
    df["Gap_Down"] = (df["Gap"] < 0).astype(int)
    df["Gap_Still_Open"] = (((df["Gap"]>0)&(df["Low"]>prev)) | ((df["Gap"]<0)&(df["High"]<prev))).astype(int)
    return df
# -----------------------------------------------------------------
def _add_volume_features(df):
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    adl = AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
    df["Chaikin_Osc"] = EMAIndicator(adl,3).ema_indicator() - EMAIndicator(adl,10).ema_indicator()
    df["Turnover"]    = df["Volume"] * df["Close"]
    df["Turnover_z"]  = (df["Turnover"]-df["Turnover"].rolling(20).mean()) / df["Turnover"].rolling(20).std()
    return df
# -----------------------------------------------------------------
def _add_volatility_breakout(df):
    df["Returns"] = df["Close"].pct_change(fill_method=None)
    df["RollStd_10"] = df["Returns"].rolling(10).std()
    df["RollStd_21"] = df["Returns"].rolling(21).std()
    df["Donchian_Width"] = (df["High"].rolling(20).max()-df["Low"].rolling(20).min())/df["Close"]
    df["Volatility_Breakout"] = (df["Close"] > df["High"].shift()+df["ATR"]).astype(int)
    df["RealisedVol_5"]  = df["Returns"].rolling(5).std()
    df["RealisedVol_10"] = df["Returns"].rolling(10).std()
    return df
# -----------------------------------------------------------------
def _add_trend_crossover(df):
    df["EMA_10"]  = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["EMA10_50_Ratio"]  = df["EMA_10"] / df["EMA_50"]
    df["EMA50_200_Ratio"] = df["EMA_50"] / df["EMA_200"]
    df["DI_Pos"] = ta.trend.adx_pos(df["High"], df["Low"], df["Close"])
    df["DI_Neg"] = ta.trend.adx_neg(df["High"], df["Low"], df["Close"])
    return df
# -----------------------------------------------------------------
def _add_long_window_features(df):
    df["RSI_30"] = ta.momentum.rsi(df["Close"], 30)
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["SMA200_Slope"] = df["SMA_200"].diff()
    return df
# -----------------------------------------------------------------
def _calculate_binary_signals(df):
    c = df["Close"]
    df["RSI_above_70"] = (df["RSI"]>70).astype(int)
    df["RSI_below_30"] = (df["RSI"]<30).astype(int)
    df["RSI_cross_down_70"] = ((df["RSI"].shift()>70)&(df["RSI"]<=70)).astype(int)
    df["RSI_cross_up_30"]   = ((df["RSI"].shift()<30)&(df["RSI"]>=30)).astype(int)
    df["Bollinger_2pct_Lower"] = (c < df["Bollinger_Middle"]*0.98).astype(int)
    df["Bollinger_Strong"]     = ((c < df["Bollinger_Lower"])|(c > df["Bollinger_Upper"])).astype(int)
    df["RSI_Bollinger_Strong_Above"] = (df["RSI"]>df["Bollinger_Upper"]).astype(int)
    df["RSI_Bollinger_Strong_Below"] = (df["RSI"]<df["Bollinger_Lower"]).astype(int)
    return df
# -----------------------------------------------------------------
# (לגים כבדים – מושבת כברירת-מחדל)
def _add_lags(df):
    for col in ["RSI","MACD","Returns"]:
        for lag in (1,2,3):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df
# -----------------------------------------------------------------
# END OF FILE