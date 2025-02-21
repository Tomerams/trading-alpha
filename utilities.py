import ta.momentum
import yfinance as yf
import ta
import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS, Indicator
from model_architecture import (
    AttentionModel,
    CNNLSTMModel,
    TCNModel,
    TransformerModel,
    GRUModel,
)


def get_model(input_size: int, model_type: str):
    """Initialize and return the selected trading model with fixed input shape."""
    hidden_size = 64
    output_size = 1

    model_map = {
        "LSTM": AttentionModel(input_size, hidden_size, output_size),
        "Transformer": TransformerModel(input_size, hidden_size, output_size),
        "CNNLSTM": CNNLSTMModel(input_size, hidden_size, output_size),
        "GRU": GRUModel(input_size, hidden_size, output_size),
        "TCN": TCNModel(input_size, hidden_size, output_size),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_map[model_type]

    # Force explicit model building for Keras models
    if isinstance(model, TCNModel) or isinstance(model, GRUModel):
        model.model.build((None, input_size, 1))

    return model


def get_data(stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Retrieve historical market data for a given stock_ticker and date range."""
    df = yf.download(stock_ticker, start=start_date, end=end_date, interval="1d")

    if df.empty:
        raise ValueError(
            f"No data retrieved for {stock_ticker}. Check the stock_ticker symbol and date range."
        )

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
    forecast_period = 4
    df["target"] = (df["Close"].shift(-forecast_period) - df["Close"]) / df["Close"]

    df.dropna(inplace=True)

    df.to_csv(f"data/{stock_ticker}_indicators_data.csv")

    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    rsi = ta.momentum.rsi(df["Close"], window=14)
    rsi_2 = ta.momentum.rsi(df["Close"], window=2)
    stochastic_k = ta.momentum.stoch(
        df["High"], df["Low"], df["Close"], window=14, smooth_window=3
    )

    if Indicator.RSI.value in FEATURE_COLUMNS:
        df["RSI"] = rsi

    if Indicator.RSI_2.value in FEATURE_COLUMNS:
        df["RSI_2"] = rsi_2

    if Indicator.RSI_ABOVE_70.value in FEATURE_COLUMNS:
        df["RSI_above_70"] = (rsi > 70).astype(int)

    if Indicator.RSI_BELOW_30.value in FEATURE_COLUMNS:
        df["RSI_2_below_30"] = (rsi < 30).astype(int)

    if Indicator.RSI_ABOVE_70.value in FEATURE_COLUMNS:
        df["RSI_above_70"] = (rsi > 70).astype(int)

    if Indicator.RSI_BELOW_30.value in FEATURE_COLUMNS:
        df["RSI_below_30"] = (rsi < 30).astype(int)

    if Indicator.RSI_2_ABOVE_90.value in FEATURE_COLUMNS:
        df["RSI_2_above_90"] = (rsi_2 > 90).astype(int)

    if Indicator.RS_2_BELOW_10.value in FEATURE_COLUMNS:
        df["RSI_2_below_10"] = (rsi_2 < 10).astype(int)

    if Indicator.RSI_2_CROSS_90.value in FEATURE_COLUMNS:
        df["RSI_2_cross_90"] = np.where((rsi_2.shift(1) < 90) & (rsi_2 >= 90), 1, 0)

    if Indicator.RSI_2_CROSS_10.value in FEATURE_COLUMNS:
        df["RSI_2_cross_10"] = np.where((rsi_2.shift(1) > 10) & (rsi_2 <= 10), 1, 0)

    if Indicator.RSI_SLOPE.value in FEATURE_COLUMNS:
        df["RSI_slope"] = rsi.diff()

    if Indicator.RSI_DIVERGENCE.value in FEATURE_COLUMNS:
        df["RSI_divergence"] = (
            (rsi.shift(1) > rsi.shift(2)) & (rsi < rsi.shift(1))
        ).astype(int)

    if Indicator.BOLLINGER_MIDDLE.value in FEATURE_COLUMNS:
        df["Bollinger_Middle"] = df["Close"].rolling(window=20).mean()

    if Indicator.BOLLINGER_UPPER.value in FEATURE_COLUMNS:
        df["Bollinger_Upper"] = df["Close"].rolling(window=20).mean() + (
            df["Close"].rolling(window=20).std() * 2
        )

    if Indicator.BOLLINGER_LOWER.value in FEATURE_COLUMNS:
        df["Bollinger_Lower"] = df["Close"].rolling(window=20).mean() - (
            df["Close"].rolling(window=20).std() * 2
        )

    if Indicator.SMA_50.value in FEATURE_COLUMNS:
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

    if Indicator.SMA_200.value in FEATURE_COLUMNS:
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

    if Indicator.EMA_50.value in FEATURE_COLUMNS:
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    if Indicator.EMA_200.value in FEATURE_COLUMNS:
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    if Indicator.MOMENTUM.value in FEATURE_COLUMNS:
        df["Momentum"] = df["Close"].diff()

    if Indicator.VOLATILITY.value in FEATURE_COLUMNS:
        df["Volatility"] = df["Close"].rolling(window=20).std()

    if Indicator.MACD.value in FEATURE_COLUMNS:
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = ta.MACD(
            df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
        )

    if Indicator.ADX.value in FEATURE_COLUMNS:
        df["ADX"] = ta.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)

    if Indicator.VWAP.value in FEATURE_COLUMNS:
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    if Indicator.ATR.value in FEATURE_COLUMNS:
        df["ATR"] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)

    if Indicator.STOCHASTIC_K.value in FEATURE_COLUMNS:
        df["Stochastic_K"] = stochastic_k

    if Indicator.STOCHASTIC_D.value in FEATURE_COLUMNS:
        df["Stochastic_D"] = stochastic_k.rolling(window=3).mean()

    if Indicator.STOCHASTIC_RSI.value in FEATURE_COLUMNS:
        df["Stoch_RSI"] = ta.momentum.stochrsi(
            df["Close"], window=14, smooth1=3, smooth2=3
        )

    return df


if __name__ == "__main__":
    CONFIG = {
        "stock_ticker": "FNGU",
        "start_date": "2020-06-06",
        "end_date": "2025-01-01",
    }

    get_data(**CONFIG)
