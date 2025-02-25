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

    # Ensure valid trading data
    df["Target_Tomorrow"] = df["Close"].shift(-1)
    df["Target_3_Days"] = df["Close"].shift(-3)
    df["Target_Next_Week"] = df["Close"].shift(-5)

    df.dropna(inplace=True)  # Remove rows where future prices aren't available

    trade_log = []

    X = df[selected_features].values
    X_scaled = data_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).view(X_scaled.shape[0], 1, X_scaled.shape[1])

    with torch.no_grad():
        predictions = trained_model(X_tensor).numpy()

    df["Predicted_Tomorrow"] = predictions[:, 0]
    df["Predicted_3_Days"] = predictions[:, 1]
    df["Predicted_Next_Week"] = predictions[:, 2]

    df["Predicted_Direction_Tomorrow"] = (df["Predicted_Tomorrow"] > df["Close"]).astype(int)
    df["Predicted_Direction_3_Days"] = (df["Predicted_3_Days"] > df["Close"]).astype(int)
    df["Predicted_Direction_Next_Week"] = (df["Predicted_Next_Week"] > df["Close"]).astype(int)

    df["Actual_Direction_Tomorrow"] = (df["Target_Tomorrow"] > df["Close"]).astype(int)
    df["Actual_Direction_3_Days"] = (df["Target_3_Days"] > df["Close"]).astype(int)
    df["Actual_Direction_Next_Week"] = (df["Target_Next_Week"] > df["Close"]).astype(int)

    cash = BACKTEST_PARAMS["initial_balance"]
    shares = 0
    last_buy_price = None
    peak_value = cash
    max_loss_per_trade = 0

    for i in range(len(df) - 1):
        trade_date = df.index[i]
        price = df.iloc[i]["Close"]

        predicted_return_tomorrow = (df["Predicted_Tomorrow"].iloc[i] - price) / price
        predicted_return_3_days = (df["Predicted_3_Days"].iloc[i] - price) / price
        predicted_return_next_week = (df["Predicted_Next_Week"].iloc[i] - price) / price

        avg_predicted_return = (predicted_return_tomorrow + predicted_return_3_days + predicted_return_next_week) / 3

        transaction_fee = (
            max(
                BACKTEST_PARAMS["buy_sell_fee_per_share"] * shares,
                BACKTEST_PARAMS["minimum_fee"],
            )
            if shares > 0
            else 0
        )

        buying_threshold = 0.002
        selling_threshold = -0.005

        if (
            avg_predicted_return > buying_threshold
            and predicted_return_next_week > buying_threshold
            and shares == 0
        ):
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

        elif avg_predicted_return < selling_threshold and shares > 0:
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

    # Compute Precision and Recall for Multiple Horizons
    precision_tomorrow = precision_score(
        df["Actual_Direction_Tomorrow"], df["Predicted_Direction_Tomorrow"], zero_division=0
    )
    recall_tomorrow = recall_score(df["Actual_Direction_Tomorrow"], df["Predicted_Direction_Tomorrow"], zero_division=0)

    precision_3_days = precision_score(df["Actual_Direction_3_Days"], df["Predicted_Direction_3_Days"], zero_division=0)
    recall_3_days = recall_score(df["Actual_Direction_3_Days"], df["Predicted_Direction_3_Days"], zero_division=0)

    precision_next_week = precision_score(
        df["Actual_Direction_Next_Week"], df["Predicted_Direction_Next_Week"], zero_division=0
    )
    recall_next_week = recall_score(
        df["Actual_Direction_Next_Week"], df["Predicted_Direction_Next_Week"], zero_division=0
    )

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
    print(f"🎯 Precision (Tomorrow): {precision_tomorrow:.4f}")
    print(f"🔍 Recall (Tomorrow): {recall_tomorrow:.4f}")
    print(f"🎯 Precision (3 Days): {precision_3_days:.4f}")
    print(f"🔍 Recall (3 Days): {recall_3_days:.4f}")
    print(f"🎯 Precision (Next Week): {precision_next_week:.4f}")
    print(f"🔍 Recall (Next Week): {recall_next_week:.4f}")

    return net_profit, trade_df, ticker_change, max_loss_per_trade, precision_tomorrow, recall_tomorrow


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
