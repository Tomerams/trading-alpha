import numpy as np
import pandas as pd
import yfinance as yf

from data.features import calculate_features
from routers.routers_entities import UpdateIndicatorsData
from data.data_utilities import exclude_from_scaling
import os


def get_data(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    os.makedirs("files", exist_ok=True)

    df = yf.download(
        request_data.stock_ticker,
        start=request_data.start_date,
        end=request_data.end_date,
        interval="1d",
    )

    if df.empty:
        raise ValueError(f"No data retrieved for {request_data.stock_ticker}.")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns:
        raise KeyError("'Close' column missing in dataset.")

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")

    df = calculate_features(df)

    df["Target_Tomorrow"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["Target_3_Days"] = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
    df["Target_Next_Week"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df["Close_Normal"] = df["Close"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_from_scaling]

    df[feature_cols] = df[feature_cols].apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df.dropna(inplace=True)

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(f"files/{request_data.stock_ticker}_indicators_data.csv", index=False)

    return df
