import logging
import math
import itertools
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, mean_absolute_error

from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_data
from models.model_utilities import load_model
from config import MODEL_PARAMS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def backtest_model(request_data: UpdateIndicatorsData, verbose: bool = True) -> dict:
    # 1) Load model + scaler + the exact feature list used in training
    ticker      = request_data.stock_ticker
    model_type  = MODEL_PARAMS.get("model_type", "LSTM")
    model, scaler, feature_cols = load_model(ticker, model_type)
    seq_len     = MODEL_PARAMS.get("seq_len", 10)
    target_cols = MODEL_PARAMS.get("target_cols", [])

    # 2) Fetch enriched data (this now already includes shift-targets, extrema, trend, etc.)
    df = get_data(request_data)
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"])
    df.dropna(inplace=True)

    # 3) Build your input matrix X and align dates/prices/true-Y
    if seq_len > 1:
        X      = np.stack([df[feature_cols].iloc[i : i + seq_len].values
                           for i in range(len(df) - seq_len)])
        dates  = df["Date"].iloc[seq_len:].reset_index(drop=True)
        prices = df["Close"].iloc[seq_len:].values
        y_true = df[target_cols].iloc[seq_len:].values
    else:
        X      = df[feature_cols].values
        dates  = df["Date"].reset_index(drop=True)
        prices = df["Close"].values
        y_true = df[target_cols].values

    # 4) Scale with the exact same scaler you saved
    if seq_len > 1:
        n_feat   = X.shape[2]
        X_flat   = X.reshape(-1, n_feat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    # 5) Predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        preds = model(X_tensor).cpu().numpy()

    # 6) (Optional) print last-bar predictions
    if verbose:
        last = -1
        for idx, name in enumerate(target_cols):
            print(f"🔮 Pred {name}: {preds[last, idx]:.4f}   (actual {y_true[last, idx]:.4f})")

    # 7) Trading rules
    cash              = float(MODEL_PARAMS.get("initial_balance", 10_000))
    shares            = 0
    last_price        = 0.0
    highest_price     = None
    max_loss_per_trade= 0.0

    profit_target     = MODEL_PARAMS.get("profit_target", 0.05)
    trailing_stop     = MODEL_PARAMS.get("trailing_stop", 0.03)
    buy_thr           = MODEL_PARAMS.get("buying_threshold", 0.0)
    sell_thr          = MODEL_PARAMS.get("selling_threshold", 0.0)
    fee_per_share     = MODEL_PARAMS.get("buy_sell_fee_per_share", 0.01)
    min_fee           = MODEL_PARAMS.get("minimum_fee", 1.0)
    tax_rate          = MODEL_PARAMS.get("tax_rate", 0.25)

    trades = []

    for i, date in enumerate(dates[:-1]):
        price  = float(prices[i])
        signal = preds[i]
        avg_ret= float(signal.mean())

        # update trailing high
        if shares > 0:
            highest_price = price if highest_price is None or price > highest_price else highest_price

        # check stops
        stop_loss_hit    = shares > 0 and price < last_price * (1 - MODEL_PARAMS.get("stop_loss_pct", 0.03))
        trailing_hit     = shares > 0 and highest_price is not None and price < highest_price * (1 - trailing_stop)
        profit_hit       = shares > 0 and price >= last_price * (1 + profit_target)

        # compute fee/tax placeholders
        fee = float(max(fee_per_share * shares, min_fee)) if shares > 0 else 0.0

        # BUY signal
        if shares == 0 and avg_ret > buy_thr and signal[target_cols.index("Target_Next_Week")] > signal[target_cols.index("Target_Tomorrow")]:
            shares     = math.floor((cash - fee) / price)
            cash      -= shares * price + fee
            last_price = price
            highest_price = price
            trades.append({
                "Date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "Type": "BUY",
                "Price": price,
                "Portfolio": float(cash + shares * price)
            })

        # SELL signal
        elif shares > 0 and (avg_ret < sell_thr or stop_loss_hit or trailing_hit or profit_hit):
            tax        = float(tax_rate * (price - last_price) * shares)
            cash      += shares * price - fee - tax
            change_pct = (price - last_price) / last_price * 100
            max_loss_per_trade = min(max_loss_per_trade, change_pct)
            trades.append({
                "Date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "Type": "SELL",
                "Price": price,
                "Portfolio": float(cash),
                "Change_%": change_pct,
                "Tax": tax
            })
            shares = 0
            highest_price = None

    # 8) Final close‐out if still holding
    if shares > 0:
        date  = dates.iloc[-1]
        price = float(prices[-1])
        tax   = float(tax_rate * (price - last_price) * shares)
        cash += shares * price - tax
        change_pct = (price - last_price) / last_price * 100
        max_loss_per_trade = min(max_loss_per_trade, change_pct)
        trades.append({
            "Date": date.strftime("%Y-%m-%dT%H:%M:%S"),
            "Type": "SELL",
            "Price": price,
            "Portfolio": float(cash),
            "Change_%": change_pct,
            "Tax": tax
        })

    # 9) Summary
    ticker_change = (prices[-1] / prices[0] - 1) * 100
    net_profit    = (cash / MODEL_PARAMS.get("initial_balance", 10_000) - 1) * 100

    if verbose:
        print(f"📉 Ticker Change: {ticker_change:.2f}%")
        print(f"📈 Portfolio    : {net_profit:.2f}%")
        print(f"⚠️ Max Loss     : {max_loss_per_trade:.2f}%")

    return {
        "net_profit": net_profit,
        "ticker_change": ticker_change,
        "max_loss_per_trade": max_loss_per_trade,
        "trades_signals": trades
    }


def optimize_signal_params(
    request_data: UpdateIndicatorsData, param_grid: dict
) -> pd.DataFrame:
    """
    Runs a grid search over signal parameters and returns a DataFrame sorted by net_profit.

    param_grid keys: buying_threshold, selling_threshold, profit_target, trailing_stop
    """
    results = []
    # Save original params
    original = {
        k: MODEL_PARAMS.get(k)
        for k in [
            "buying_threshold",
            "selling_threshold",
            "profit_target",
            "trailing_stop",
        ]
    }

    for bt, st, pt, ts in itertools.product(
        param_grid.get("buying_threshold", [original["buying_threshold"]]),
        param_grid.get("selling_threshold", [original["selling_threshold"]]),
        param_grid.get("profit_target", [original["profit_target"]]),
        param_grid.get("trailing_stop", [original["trailing_stop"]]),
    ):
        # Override\
        MODEL_PARAMS.update(
            {
                "buying_threshold": bt,
                "selling_threshold": st,
                "profit_target": pt,
                "trailing_stop": ts,
            }
        )
        res = backtest_model(request_data, verbose=False)
        results.append(
            {
                "buying_threshold": bt,
                "selling_threshold": st,
                "profit_target": pt,
                "trailing_stop": ts,
                "net_profit": res["net_profit"],
                "ticker_change": res["ticker_change"],
                "max_loss_per_trade": res["max_loss_per_trade"],
                "num_trades": len(res["trades_signals"]),
            }
        )
    # restore original
    MODEL_PARAMS.update(original)

    df = pd.DataFrame(results)
    return df.sort_values("net_profit", ascending=False).reset_index(drop=True)
