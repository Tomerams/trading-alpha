from enum import Enum


class GannFeaturs(Enum):
    ANGLE_0_5D = "Gann_Angle_0.5D"
    ANGLE_1D = "Gann_Angle_1D"
    ANGLE_2D = "Gann_Angle_2D"
    UP_CYCLE_LENGTH = "Gann_Up_Cycle_Length"
    DOWN_CYCLE_LENGTH = "Gann_Down_Cycle_Length"
    FIB_23_ABOVE = "Fibonacci_23_Above"
    FIB_38_ABOVE = "Fibonacci_38_Above"
    FIB_50_ABOVE = "Fibonacci_50_Above"
    FIB_62_ABOVE = "Fibonacci_62_Above"


class ExternalFeatures(Enum):
    VIX_CLOSE = "VIX_Close"
    SPY_CLOSE = "SPY_Close"
    US10Y_YIELD = "US10Y_Yield"
    US2Y_YIELD = "US2Y_Yield"


class Indicator(Enum):
    RSI = "RSI"
    RSI_2 = "RSI_2"
    RSI_SLOPE = "RSI_slope"
    RSI_DIVERGENCE = "RSI_divergence"
    SMA_50 = "SMA_50"
    EMA_50 = "EMA_50"
    BOLLINGER_UPPER = "Bollinger_Upper"
    BOLLINGER_LOWER = "Bollinger_Lower"
    BOLLINGER_MIDDLE = "Bollinger_Middle"
    MOMENTUM = "Momentum"
    VOLATILITY = "Volatility"
    MACD = "MACD"
    MACD_SIGNAL = "MACD_Signal"
    MACD_HIST = "MACD_Hist"
    ADX = "ADX"
    VWAP = "VWAP"
    ATR = "ATR"
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
    RSI_BOLLINGER_MIDDLE = "RSI_Bollinger_Middle"
    RSI_BOLLINGER_UPPER = "RSI_Bollinger_Upper"
    RSI_BOLLINGER_LOWER = "RSI_Bollinger_Lower"


class BinaryIndicator(Enum):
    RSI_ABOVE_70 = "RSI_above_70"
    RSI_BELOW_30 = "RSI_below_30"
    RSI_CROSS_DOWN_70 = "RSI_cross_down_70"
    RSI_CROSS_UP_30 = "RSI_cross_up_30"
    RSI_2_ABOVE_90 = "RSI_2_above_90"
    RS_2_BELOW_10 = "RSI_2_below_10"
    RSI_2_CROSS_90 = "RSI_2_cross_90"
    RSI_2_CROSS_10 = "RSI_2_cross_10"
    BOLLINGER_STRONG = "Bollinger_Strong"
    BOLLINGER_2PCT_LOWER = "Bollinger_2pct_Lower"
    RSI_BOLLINGER_STRONG_ABOVE = "RSI_Bollinger_Strong_Above"
    RSI_BOLLINGER_STRONG_BELOW = "RSI_Bollinger_Strong_Below"
    TREND_UP = ("TrendUp",)
    TREND_DOWN = "TrendDown"


class Pattern(Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    HEAD_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    CUP_HANDLE = "cup_and_handle"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    RECTANGLE_PATTERN = "rectangle_pattern"
    BULLISH_FLAG = "bullish_flag"
    BEARISH_FLAG = "bearish_flag"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"


class Gaps(Enum):
    GAP = "Gap"
    GAP_PCT = "Gap_Pct"
    GAP_UP = "Gap_Up"
    GAP_DOWN = "Gap_Down"
    GAP_STILL_OPEN = "Gap_Still_Open"


class DateFeatures(Enum):
    DAY_OF_WEEK = "Day_Of_Week"
    MONTH = "Month"
    DAY_OF_MONTH = "Day_of_Month"
    QUARTER = "Quarter"
    DAY_OF_YEAR = "Day_of_Year"


class ExternalDerivedFeatures(Enum):
    SPY_Change = "SPY_Change"
    VIX_Change = "VIX_Change"
    US10Y_Change = "US10Y_Change"
    US2Y_Change = "US2Y_Change"
    SPY_VIX_RATIO = "SPY_VIX_RATIO"
    YIELD_SPREAD_10Y_2Y = "YIELD_SPREAD_10Y_2Y"


FEATURE_COLUMNS = [
    *[i.value for i in Indicator],
    *[b.value for b in BinaryIndicator],
    *[p.value for p in Pattern],
    *[g.value for g in GannFeaturs],
    *[d.value for d in DateFeatures],
    *[ed.value for ed in ExternalDerivedFeatures],
    *[gap.value for gap in Gaps],
]
