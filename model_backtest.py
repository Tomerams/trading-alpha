import math
import torch
import pandas as pd
import logging
from data_processing import get_data
from config import BACKTEST_PARAMS
from utilities import load_model
from sklearn.metrics import precision_score, recall_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def backtest_model(
    stock_ticker, start_date, end_date, trained_model, data_scaler, selected_features
):
    logging.info(
        f"📊 Fetching data for {stock_ticker} from {start_date} to {end_date}..."
    )

    df = get_data(stock_ticker, start_date, end_date)

    if df is None or df.empty:
        raise ValueError(
            "⚠️ No data available after applying the date range. Check `get_data()` output."
        )

    logging.info(f"✅ Data loaded with shape: {df.shape}")

    print(df)

    # df.index = pd.to_datetime(df.index)

    # Ensure 'Close' is a Series, not a DataFrame
    # df["Close"] = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]

    # Create targets
    df["Target_Tomorrow"] = df["Close"].shift(-1).astype(float)
    df["Target_3_Days"] = df["Close"].shift(-3).astype(float)
    df["Target_Next_Week"] = df["Close"].shift(-5).astype(float)

    df.dropna(inplace=True)

    trade_log = []

    X = df[selected_features].values
    X_scaled = data_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).view(
        X_scaled.shape[0], 1, X_scaled.shape[1]
    )

    with torch.no_grad():
        predictions = trained_model(X_tensor).numpy()

    df["Predicted_Tomorrow"] = df["Close"] * (predictions[:, 0] + 1)
    df["Predicted_3_Days"] = df["Close"] * (predictions[:, 1] + 1)
    df["Predicted_Next_Week"] = df["Close"] * (predictions[:, 2] + 1)

    df["Predicted_Direction_Tomorrow"] = (
        df["Predicted_Tomorrow"] > df["Close"]
    ).astype(int)
    df["Predicted_Direction_3_Days"] = (df["Predicted_3_Days"] > df["Close"]).astype(
        int
    )
    df["Predicted_Direction_Next_Week"] = (
        df["Predicted_Next_Week"] > df["Close"]
    ).astype(int)

    df["Actual_Direction_Tomorrow"] = (df["Target_Tomorrow"] > df["Close"]).astype(int)
    df["Actual_Direction_3_Days"] = (df["Target_3_Days"] > df["Close"]).astype(int)
    df["Actual_Direction_Next_Week"] = (df["Target_Next_Week"] > df["Close"]).astype(
        int
    )

    cash = BACKTEST_PARAMS["initial_balance"]
    shares = 0
    last_buy_price = None
    peak_value = cash
    max_loss_per_trade = 0

    buying_threshold = 0.005
    selling_threshold = -0.01

    for i in range(len(df) - 1):
        trade_date = df.iloc[i]["Date"]
        stock_price = df.iloc[i]["Close"]
        stop_loss_triggered = shares > 0 and (stock_price < last_buy_price * 0.97)

        avg_predicted_return = (
            predictions[i, 0] + predictions[i, 1] + predictions[i, 2]
        ) / 3
        transaction_fee = (
            max(
                BACKTEST_PARAMS["buy_sell_fee_per_share"] * shares,
                BACKTEST_PARAMS["minimum_fee"],
            )
            if shares > 0
            else 0
        )
        if (
            avg_predicted_return > buying_threshold
            and df.iloc[i]["Predicted_Next_Week"] > df.iloc[i]["Predicted_Tomorrow"]
            and shares == 0
        ):
            shares = math.floor((cash - transaction_fee) / stock_price)
            cash = cash - shares * stock_price
            last_buy_price = stock_price
            trade_log.append(
                [trade_date, "BUY", stock_price, shares * stock_price, None, None, None]
            )

        elif (
            avg_predicted_return < selling_threshold and shares > 0
        ) or stop_loss_triggered:
            tax_paid = (
                BACKTEST_PARAMS["tax_rate"] * (stock_price - last_buy_price) * shares
            )
            cash = cash + (shares * stock_price) - transaction_fee - tax_paid
            shares = 0
            price_change = (
                ((stock_price - last_buy_price) / last_buy_price) * 100
                if last_buy_price
                else None
            )
            portfolio_gain = (
                (cash - BACKTEST_PARAMS["initial_balance"])
                / BACKTEST_PARAMS["initial_balance"]
            ) * 100

            if price_change and price_change < 0:
                max_loss_per_trade = min(max_loss_per_trade, price_change)

            trade_log.append(
                [
                    trade_date,
                    "SELL",
                    stock_price,
                    cash,
                    price_change,
                    portfolio_gain,
                    tax_paid,
                ]
            )

        peak_value = max(peak_value, cash + (shares * df.iloc[i]["Close"]))

    total_value = cash + (shares * stock_price)
    net_profit = (total_value / BACKTEST_PARAMS["initial_balance"]) * 100

    ticker_change = (df["Close"].iloc[-1] / df["Close"].iloc[0]) * 100

    # Compute Precision and Recall for Multiple Horizons
    precision_tomorrow = precision_score(
        df["Actual_Direction_Tomorrow"],
        df["Predicted_Direction_Tomorrow"],
        zero_division=0,
    )
    recall_tomorrow = recall_score(
        df["Actual_Direction_Tomorrow"],
        df["Predicted_Direction_Tomorrow"],
        zero_division=0,
    )

    precision_3_days = precision_score(
        df["Actual_Direction_3_Days"], df["Predicted_Direction_3_Days"], zero_division=0
    )
    recall_3_days = recall_score(
        df["Actual_Direction_3_Days"], df["Predicted_Direction_3_Days"], zero_division=0
    )

    precision_next_week = precision_score(
        df["Actual_Direction_Next_Week"],
        df["Predicted_Direction_Next_Week"],
        zero_division=0,
    )
    recall_next_week = recall_score(
        df["Actual_Direction_Next_Week"],
        df["Predicted_Direction_Next_Week"],
        zero_division=0,
    )

    # Compute Mean Absolute Error (MAE)
    mae_tomorrow = mean_absolute_error(df["Target_Tomorrow"], df["Predicted_Tomorrow"])
    mae_3_days = mean_absolute_error(df["Target_3_Days"], df["Predicted_3_Days"])
    mae_next_week = mean_absolute_error(
        df["Target_Next_Week"], df["Predicted_Next_Week"]
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
            "Tax Paid",
        ],
    )
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
    print(f"📊 MAE (Tomorrow): {mae_tomorrow:.4f}")
    print(f"📊 MAE (3 Days): {mae_3_days:.4f}")
    print(f"📊 MAE (Next Week): {mae_next_week:.4f}")

    return net_profit, trade_df, ticker_change, max_loss_per_trade


if __name__ == "__main__":
    ticker = "FNGA"
    start_date = "2023-01-01"
    end_date = "2024-04-24"
    model_type = "TransformerRNN"
    trained_model, data_scaler, best_features = load_model(ticker, model_type)

    print("🚀 Running Backtest Using Trained Model...")
    backtest_model(
        ticker, start_date, end_date, trained_model, data_scaler, best_features
    )
