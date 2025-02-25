from enum import Enum


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
    SMA_200 = "SMA_200"
    EMA_50 = "EMA_50"
    EMA_200 = "EMA_200"
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
    PE_RATIO = "P/E_Ratio"
    EPS = "EPS"
    DEBT_TO_EQUITY = "Debt_to_Equity"
    BOLLINGER_2PCT_LOWER = "Bollinger_2pct_Lower"
    RSI_2_ABOVE_90 = "RSI_2_above_90"
    RS_2_BELOW_10 = "RSI_2_below_10"
    RSI_2_CROSS_90 = "RSI_2_cross_90"
    RSI_2_CROSS_10 = "RSI_2_cross_10"
    STOCHASTIC_K = "Stochastic_K"
    STOCHASTIC_D = "Stochastic_D"
    STOCHASTIC_RSI = "Stoch_RSI"


FEATURE_COLUMNS = [
    Indicator.RSI.value,
    Indicator.RSI_2.value,
    Indicator.RSI_ABOVE_70.value,
    Indicator.RSI_BELOW_30.value,
    Indicator.RSI_SLOPE.value,
    Indicator.RSI_DIVERGENCE.value,
    Indicator.RS_2_BELOW_10.value,
    Indicator.RSI_2_ABOVE_90.value,
    Indicator.RSI_2_CROSS_10.value,
    Indicator.RSI_2_CROSS_90.value,
    Indicator.STOCHASTIC_K.value,
    Indicator.STOCHASTIC_D.value,
    Indicator.STOCHASTIC_RSI.value,
    # Indicator.SMA_50.value,
    # Indicator.SMA_200.value,
    # Indicator.EMA_50.value,
    # Indicator.EMA_200.value,
    # Indicator.BOLLINGER_UPPER.value,
    # Indicator.BOLLINGER_LOWER.value,
    # Indicator.BOLLINGER_MIDDLE.value,
    Indicator.MOMENTUM.value,
    Indicator.VOLATILITY.value,
    Indicator.MACD.value,
    Indicator.MACD_SIGNAL.value,
    Indicator.MACD_HIST.value,
    Indicator.ADX.value,
    # Indicator.VWAP.value,
    # Indicator.ATR.value,
]

MODEL_PARAMS = {
    "hidden_size": 64,
    "output_size": 1,
    "learning_rate": 0.001,
    "epochs": 100,
}

BACKTEST_PARAMS = {
    "initial_balance": 10000,
    "tax_rate": 0.25,
    "forecast_steps": 5,
    "starting_point": 20,
    "buy_sell_fee_per_share": 0.01,
    "minimum_fee": 1,
}
