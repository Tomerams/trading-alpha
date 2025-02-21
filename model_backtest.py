import torch
import joblib
import os
import pandas as pd
import logging
from utilities import get_data, get_model
from config import BACKTEST_PARAMS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(ticker, model_type):
    model_filename = f"models/trained_model-{ticker}-{model_type}.pkl"
    scaler_filename = f"models/scaler-{ticker}-{model_type}.pkl"
    features_filename = f"models/features-{ticker}-{model_type}.pkl"

    if (
        not os.path.exists(model_filename)
        or not os.path.exists(scaler_filename)
        or not os.path.exists(features_filename)
    ):
        raise FileNotFoundError(
            f"❌ Model files for {ticker}-{model_type} not found! Train the model first."
        )

    checkpoint = torch.load(model_filename, map_location=torch.device("cpu"))

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"❌ Invalid model file: {model_filename}. The key 'model_state_dict' is missing!"
        )

    features = joblib.load(features_filename)
    scaler = joblib.load(scaler_filename)

    model = get_model(input_size=len(features), model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, features


def backtest_model(
    stock_ticker, start_date, end_date, trained_model, data_scaler, selected_features
):
    df = get_data(stock_ticker, start_date, end_date)
    if df is None or df.empty:
        raise ValueError("No data available after applying the date range.")

    df = df[selected_features + ["Close"]].copy()
    df.index = pd.to_datetime(df.index)

    trade_log = []

    X = df[selected_features].values
    X_scaled = data_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).view(
        X_scaled.shape[0], 1, X_scaled.shape[1]
    )

    with torch.no_grad():
        predictions = trained_model(X_tensor).numpy().flatten()

    df["Predicted_Return"] = predictions
    df["Predicted_Close"] = df["Close"] * (1 + df["Predicted_Return"])

    # Trading simulation parameters
    cash = BACKTEST_PARAMS["initial_balance"]
    shares = 0
    last_buy_price = None
    peak_value = cash

    for i in range(len(df) - 1):
        trade_date = df.index[i]
        price = df.iloc[i]["Close"]
        predicted_return = predictions[i]

        transaction_fee = (
            max(
                BACKTEST_PARAMS["buy_sell_fee_per_share"] * shares,
                BACKTEST_PARAMS["minimum_fee"],
            )
            if shares > 0
            else 0
        )

        if predicted_return > 0 and cash > price + transaction_fee:  # BUY Condition
            shares = cash / (price + transaction_fee)
            cash = 0
            last_buy_price = price
            trade_log.append(
                [
                    trade_date,
                    "BUY",
                    price,
                    cash + (shares * df.iloc[i + 1]["Close"]),
                    None,
                    None,
                ]
            )

        elif predicted_return < 0 and shares > 0:  # SELL Condition
            sell_price = price
            cash = (shares * sell_price) - transaction_fee
            shares = 0
            price_change = (
                ((sell_price - last_buy_price) / last_buy_price) * 100
                if last_buy_price
                else None
            )
            portfolio_gain = (
                (cash - BACKTEST_PARAMS["initial_balance"])
                / BACKTEST_PARAMS["initial_balance"]
            ) * 100
            trade_log.append(
                [trade_date, "SELL", sell_price, cash, price_change, portfolio_gain]
            )

        peak_value = max(peak_value, cash + (shares * df.iloc[i]["Close"]))

    total_value = cash + (shares * df.iloc[-1]["Close"])
    net_profit = (
        (total_value - BACKTEST_PARAMS["initial_balance"])
        / BACKTEST_PARAMS["initial_balance"]
    ) * 100

    trade_df = pd.DataFrame(
        trade_log,
        columns=[
            "Date",
            "Trade Type",
            f"{stock_ticker} Price",
            "Portfolio Value",
            "Price Change %",
            "Portfolio Gain %",
        ],
    )
    trade_df["Date"] = pd.to_datetime(trade_df["Date"])
    trade_df.to_csv("data/backtest_results.csv", index=False)

    return net_profit, trade_df


if __name__ == "__main__":
    ticker = "FNGU"
    start_date = "2024-01-01"
    end_date = "2025-02-02"

    print("🚀 Running Feature Experiments...")

    model_type = "Transformer"
    trained_model, data_scaler, best_features = load_model(ticker, model_type)

    print("🚀 Running Backtest Using Trained Model...")
    net_profit, trade_df = backtest_model(
        ticker, start_date, end_date, trained_model, data_scaler, best_features
    )

    print(f"✅ Net Profit from Trading Strategy: {net_profit:.2f}%")
    print(f"📊 Trade history saved to 'backtest_results.csv'!")
    print(trade_df.head())
