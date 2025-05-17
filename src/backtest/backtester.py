import logging
import time
import pandas as pd
import math
import numpy as np
import torch
from backtest.backtest_utilities import decide_action_meta
from models.model_signals_decision_train import load_meta_model
from config import MODEL_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model


def backtest_model(request_data, verbose=True):
    ticker = request_data.stock_ticker
    logging.info("▶️  backtest_model start for %s", ticker)
    t_all = time.time()

    # 1) Load base model & scaler
    model_type = MODEL_PARAMS["model_type"]
    model, scaler, feature_cols = load_model(ticker, model_type)
    seq_len = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]

    # 2) Fetch indicators
    t0 = time.time()
    df = get_indicators_data(request_data)
    logging.info("   • data fetched in %.2fs — shape %s", time.time() - t0, df.shape)

    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)

    # 3) Build X, dates, prices, true_vals
    if seq_len > 1:
        t0 = time.time()
        X = np.stack(
            [
                df[feature_cols].iloc[i : i + seq_len].values
                for i in range(len(df) - seq_len)
            ]
        )
        logging.info(
            "   • built X array in %.2fs — X.shape=%s", time.time() - t0, X.shape
        )
        dates = df["Date"].iloc[seq_len:].reset_index(drop=True)
        prices = df["Close"].iloc[seq_len:].values
        true_vals = df[target_cols].iloc[seq_len:].values
    else:
        t0 = time.time()
        X = df[feature_cols].values
        logging.info(
            "   • built X array in %.2fs — X.shape=%s", time.time() - t0, X.shape
        )
        dates = df["Date"].reset_index(drop=True)
        prices = df["Close"].values
        true_vals = df[target_cols].values

    # 4) Scale & infer
    t0 = time.time()
    if seq_len > 1:
        n_feat = X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    if seq_len == 1 and X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    logging.info(
        "   • inference done in %.2fs — preds.shape=%s", time.time() - t0, preds.shape
    )

    # 5) LOAD META-MODEL
    logging.info("   • loading meta-model…")
    t0 = time.time()
    meta_model = load_meta_model()
    logging.info(
        "   • load_meta_model returned in %.2fs — %s",
        time.time() - t0,
        "FOUND" if meta_model else "NOT FOUND",
    )
    if meta_model is None:
        raise ValueError("Meta model not found. Please train it first.")

    # 6) Prepare backtest state
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

    # 7) TRADE LOOP
    n_steps = len(dates) - 1
    logging.info("   • entering trade loop for %d steps", n_steps)
    t_loop = time.time()
    for i, date in enumerate(dates[:-1]):
        if i == 0 or i % max(1, n_steps // 5) == 0:
            logging.info("      – loop step %d/%d", i, n_steps)

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

    logging.info("   • trade loop done in %.2fs", time.time() - t_loop)

    # 8) Finalize any open position
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

    # 9) Compute returns
    ticker_ret = (prices[-1] / prices[0] - 1) * 100
    net_ret = (cash / MODEL_PARAMS["initial_balance"] - 1) * 100

    if verbose:
        print(
            f"📉 Ticker: {ticker_ret:.2f}%  📈 Portfolio: {net_ret:.2f}%  ⚠️ Max Loss: {max_loss:.2f}%"
        )

    logging.info("✅ backtest_model complete in %.2fs", time.time() - t_all)
    return {
        "ticker_change": ticker_ret,
        "net_profit": net_ret,
        "max_loss_per_trade": max_loss,
        "trades_signals": trades,
    }
