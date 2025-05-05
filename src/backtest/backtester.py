import pandas as pd
from backtest.backtest_utilities import decide_action_meta
from backtest.meta_model import load_meta_model
from config import MODEL_PARAMS
import torch
from data.data_processing import get_data
from models.model_utilities import load_model
import math
import numpy as np


def backtest_model(request_data, verbose=True):
    ticker = request_data.stock_ticker
    model_type = MODEL_PARAMS["model_type"]
    model, scaler, feature_cols = load_model(ticker, model_type)
    seq_len = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]

    df = get_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)

    if seq_len > 1:
        X = np.stack(
            [
                df[feature_cols].iloc[i : i + seq_len].values
                for i in range(len(df) - seq_len)
            ]
        )
        dates = df["Date"].iloc[seq_len:].reset_index(drop=True)
        prices = df["Close"].iloc[seq_len:].values
        true_vals = df[target_cols].iloc[seq_len:].values
    else:
        X = df[feature_cols].values
        dates = df["Date"].reset_index(drop=True)
        prices = df["Close"].values
        true_vals = df[target_cols].values

    if seq_len > 1:
        n_feat = X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        preds = model(X_tensor).cpu().numpy()

    # 📌 Load meta-model
    meta_model = load_meta_model()
    if meta_model is None:
        raise ValueError("Meta model not found. Please train it first.")

    cash = MODEL_PARAMS["initial_balance"]
    shares = 0
    last_price = 0.0
    highest_price = None
    max_loss = 0.0
    trades = []

    stop_pct = MODEL_PARAMS["stop_loss_pct"]
    profit_tgt = MODEL_PARAMS["profit_target"]
    trail_stop = MODEL_PARAMS["trailing_stop"]
    fee_share = MODEL_PARAMS["buy_sell_fee_per_share"]
    min_fee = MODEL_PARAMS["minimum_fee"]
    tax_rate = MODEL_PARAMS["tax_rate"]

    for i, date in enumerate(dates[:-1]):
        price = float(prices[i])

        action = decide_action_meta(meta_model, preds, target_cols, i)

        if shares:
            highest_price = max(highest_price, price)

        stop_hit = shares and price < last_price * (1 - stop_pct)
        trail_hit = shares and price < highest_price * (1 - trail_stop)
        profit_hit = shares and price >= last_price * (1 + profit_tgt)

        fee = max(shares * fee_share, min_fee) if shares else 0.0

        # BUY
        if not shares and action == 2:
            shares = math.floor((cash - fee) / price)
            cash -= shares * price + fee
            last_price = price
            highest_price = price
            trades.append(
                {
                    "Date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "Type": "BUY",
                    "Price": price,
                    "Portfolio": cash + shares * price,
                }
            )

        # SELL
        elif shares and (action == 0 or stop_hit or trail_hit or profit_hit):
            tax = tax_rate * (price - last_price) * shares
            cash += shares * price - fee - tax
            change_pct = (price - last_price) / last_price * 100
            max_loss = min(max_loss, change_pct)
            trades.append(
                {
                    "Date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                    "Type": "SELL",
                    "Price": price,
                    "Portfolio": cash,
                    "Change_%": change_pct,
                    "Tax": tax,
                }
            )
            shares = 0
            highest_price = None

    if shares:
        price = float(prices[-1])
        tax = tax_rate * (price - last_price) * shares
        cash += shares * price - tax
        change_pct = (price - last_price) / last_price * 100
        max_loss = min(max_loss, change_pct)
        trades.append(
            {
                "Date": dates.iloc[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                "Type": "SELL",
                "Price": price,
                "Portfolio": cash,
                "Change_%": change_pct,
                "Tax": tax,
            }
        )

    ticker_ret = (prices[-1] / prices[0] - 1) * 100
    net_ret = (cash / MODEL_PARAMS["initial_balance"] - 1) * 100

    if verbose:
        print(
            f"📉 Ticker: {ticker_ret:.2f}%  📈 Portfolio: {net_ret:.2f}%  ⚠️ Max Loss: {max_loss:.2f}%"
        )

    return {
        "ticker_change": ticker_ret,
        "net_profit": net_ret,
        "max_loss_per_trade": max_loss,
        "trades_signals": trades,
    }
