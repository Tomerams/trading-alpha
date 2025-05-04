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


class Pattern(Enum):
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_and_shoulders"
    CUP_HANDLE = "cup_and_handle"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottm"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"


class DateFeatures(Enum):
    DAY_OF_WEEK = "Day_Of_Week"
    MONTH = "Month"


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
]


MODEL_PARAMS = {
    "hidden_size": 64,
    "output_size": 3,
    # Model architecture
    "seq_len": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 10,
    "model_type": "TransformerTCN",
    # Backtest parameters
    "initial_balance": 10000,
    "buy_sell_fee_per_share": 0.01,
    "minimum_fee": 1,
    "tax_rate": 0.25,
    "buying_threshold": 0.05,
    "selling_threshold": 0.005,
    "profit_target": 0.05,
    "trailing_stop": 0.03,
    "stop_loss_pct": 0.03,
    # Grids for /optimize-signals endpoint
    "grid_buying_threshold": [0.0, 0.01, 0.02, 0.05],
    "grid_selling_threshold": [0.0, 0.005, 0.01, 0.02],
    "grid_profit_target": [0.02, 0.05, 0.1],
    "grid_trailing_stop": [0.02, 0.03, 0.05],
    # Other hyperparameters
    "weight_decay": 1e-4,
    "val_ratio": 0.2,
    "lr_patience": 5,
    "lr_factor": 0.5,
    "min_lr": 1e-6,
    "early_stopping_patience": 10,
    "target_type": "log",
    # dynamic targets
    "target_cols": [
        "Target_Tomorrow",
        "Target_3_Days",
        "Target_Next_Week",
        "NextLocalMaxPct",
        "NextLocalMinPct",
        "TrendDirection",
    ],
    "shift_targets": [
        {"name": "Tomorrow", "shift": -1},
        {"name": "3_Days", "shift": -3},
        {"name": "Next_Week", "shift": -5},
    ],
    "extrema_window": 10,
    "trend_ema_short": 10,
    "trend_ema_long": 30,
    # GBM signal model flags & thresholds
    "use_gbm_signals": False,
    "gbm_buy_threshold": 0.6,
    "gbm_sell_threshold": 0.4,
    "gbm_params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
}


OPTUNA_PARAMS = {
    "n_trials": 50,
    "timeout_seconds": 3600,
    "max_epochs": 100,
    "batch_size": 64,
    "early_stopping_patience": 5,
}


BACKTEST_PARAMS = {
    "initial_balance": 10000,
    "tax_rate": 0.25,
    "forecast_steps": 5,
    "starting_point": 20,
    "buy_sell_fee_per_share": 0.01,
    "minimum_fee": 1,
}
