from enum import Enum

USE_GANN_FEATURES = False

GANN_FEATURES = {
    "angles": {"use": True, "windows": [0.5, 1, 2]},
    "cycles": {"use": True, "up_days": 5, "down_days": 3},
}


class Indicator(Enum):
    RSI = "RSI"
    RSI_2 = "RSI_2"
    RSI_ABOVE_70 = "RSI_above_70"
    RSI_BELOW_30 = "RSI_below_30"
    RSI_SLOPE = "RSI_slope"
    RSI_DIVERGENCE = "RSI_divergence"
    RSI_CROSS_DOWN_70 = "RSI_cross_down_70"
    RSI_CROSS_UP_30 = "RSI_cross_up_30"
    SMA_50 = "SMA_50"
    EMA_50 = "EMA_50"
    BOLLINGER_UPPER = "Bollinger_Upper"
    BOLLINGER_LOWER = "Bollinger_Lower"
    BOLLINGER_MIDDLE = "Bollinger_Middle"
    MOMENTUM = "Momentum"
    VOLATILITY = "Volatility"
    BOLINGER_STRONG = "Bollinger_Strong"
    MACD = "MACD"
    MACD_SIGNAL = "MACD_Signal"
    MACD_HIST = "MACD_Hist"
    ADX = "ADX"
    VWAP = "VWAP"
    ATR = "ATR"
    BOLLINGER_2PCT_LOWER = "Bollinger_2pct_Lower"
    RSI_2_ABOVE_90 = "RSI_2_above_90"
    RS_2_BELOW_10 = "RSI_2_below_10"
    RSI_2_CROSS_90 = "RSI_2_cross_90"
    RSI_2_CROSS_10 = "RSI_2_cross_10"
    STOCHASTIC_K = "Stochastic_K"
    STOCHASTIC_D = "Stochastic_D"
    STOCHASTIC_RSI = "Stoch_RSI"
    CCI = "CCI"
    ROC = "ROC"
    WILLIAMS_R = "Williams_%R"
    TRIX = "TRIX"
    TSI = "TSI"
    RSI_CROSS_50 = "RSI_cross_50"
    VOLATILITY_BREAKOUT = "Volatility_Breakout"
    SHORT_MOMENTUM = "Short_Momentum"
    TREND_5D = "Trend_5D"
    TREND_10D = "Trend_10D"
    VOLUME_TRAND = "Volume_Trend"
    RSI_DELTA = "RSI_Delta"
    MACD_DELTA = "MACD_Delta"
    BOLLINGER_WIDTH = "Bollinger_Width"
    MOMENTUM_CHANGE = "Momentum_Change"
    VOLATILITY_CHANGE = "Volatility_Change"
    VOLUME = "Volume"


class Pattern(Enum):
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_and_shoulders"
    CUP_HANDLE = "cup_and_handle"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottm"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"


FEATURE_COLUMNS = [
    *[indicator.value for indicator in Indicator],
    *[pattern.value for pattern in Pattern],
]

MODEL_PARAMS = {
    "hidden_size": 128,
    "output_size": 1,
    "learning_rate": 0.001,
    "epochs": 60,
}

BACKTEST_PARAMS = {
    "initial_balance": 10000,
    "tax_rate": 0.25,
    "forecast_steps": 5,
    "starting_point": 20,
    "buy_sell_fee_per_share": 0.01,
    "minimum_fee": 1,
}
