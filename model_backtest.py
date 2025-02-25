import torch
import pandas as pd
import logging
from data_processing import get_data
from config import BACKTEST_PARAMS
from utilities import load_model
from sklearn.metrics import precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def backtest_model(stock_ticker, start_date, end_date, trained_model, data_scaler, selected_features):
    df = get_data(stock_ticker, start_date, end_date)
    if df is None or df.empty:
        raise ValueError("No data available after applying the date range.")

    df = df[selected_features + ["Close"]].copy()
    df.index = pd.to_datetime(df.index)

    trade_log = []

    X = df[selected_features].values
    X_scaled = data_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).view(X_scaled.shape[0], 1, X_scaled.shape[1])

    with torch.no_grad():
        predictions = trained_model(X_tensor).numpy().flatten()

    df["Predicted_Return"] = predictions
    df["Predicted_Close"] = df["Close"] * (1 + df["Predicted_Return"])
    df["Predicted_Direction"] = (df["Predicted_Return"] > 0).astype(int)
    df["Actual_Direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(subset=["Actual_Direction", "Predicted_Direction"], inplace=True)

    cash = BACKTEST_PARAMS["initial_balance"]
    shares = 0
    last_buy_price = None
    peak_value = cash
    max_loss_per_trade = 0

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
        buying_threshold = 0.005
        selling_threshold = -0.01

        if predicted_return > buying_threshold and shares == 0:
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

        elif predicted_return < selling_threshold and shares > 0:
            sell_price = price
            cash = (shares * sell_price) - transaction_fee
            shares = 0
            price_change = ((sell_price - last_buy_price) / last_buy_price) * 100 if last_buy_price else None
            portfolio_gain = ((cash - BACKTEST_PARAMS["initial_balance"]) / BACKTEST_PARAMS["initial_balance"]) * 100

            if price_change and price_change < 0:
                max_loss_per_trade = min(max_loss_per_trade, price_change)

            trade_log.append([trade_date, "SELL", sell_price, cash, price_change, portfolio_gain])

        peak_value = max(peak_value, cash + (shares * df.iloc[i]["Close"]))

    total_value = cash + (shares * df.iloc[-1]["Close"])
    net_profit = ((total_value - BACKTEST_PARAMS["initial_balance"]) / BACKTEST_PARAMS["initial_balance"]) * 100
    ticker_change = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100

    precision = precision_score(df["Actual_Direction"], df["Predicted_Direction"], zero_division=0)
    recall = recall_score(df["Actual_Direction"], df["Predicted_Direction"], zero_division=0)

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

    print(f"📉 Ticker Change: {ticker_change:.2f}%")
    print(f"📈 Portfolio Change: {net_profit:.2f}%")
    print(f"⚠️ Maximum Loss per Trade: {max_loss_per_trade:.2f}%")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔍 Recall: {recall:.4f}")

    return net_profit, trade_df, ticker_change, max_loss_per_trade, precision, recall

if __name__ == "__main__":
    ticker = "TQQQ"
    start_date = "2020-01-01"
    end_date = "2025-01-01"
    model_type = "TransformerRNN"
    trained_model, data_scaler, best_features = load_model(ticker, model_type)

    print("🚀 Running Backtest Using Trained Model...")
    net_profit, trade_df, ticker_change, max_loss_per_trade, precision, recall = backtest_model(
        ticker, start_date, end_date, trained_model, data_scaler, best_features
    )

    print(trade_df.head())
