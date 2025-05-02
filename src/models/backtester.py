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
    """
    Backtest trading strategy for the given stock and trained model.
    Returns a dict with summary metrics and a list of trade signals.
    """
    # Load model and params
    ticker = request_data.stock_ticker
    start, end = request_data.start_date, request_data.end_date
    model_type = MODEL_PARAMS.get("model_type", "LSTM")
    model, scaler, feature_cols = load_model(ticker, model_type)
    seq_len = MODEL_PARAMS.get("seq_len", 10)

    # Fetch data
    logging.info(f"📊 Fetching data for {ticker} from {start} to {end}...")
    df = get_data(request_data)
    if df is None or df.empty:
        raise ValueError("No data for backtest.")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"])

    # Targets
    df["Target_Tomorrow"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["Target_3_Days"]   = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
    df["Target_Next_Week"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df.dropna(inplace=True)

    # Prepare sequences
    if seq_len > 1:
        X = np.array([df[feature_cols].iloc[i:i+seq_len].values
                      for i in range(len(df) - seq_len)])
        dates = df["Date"].iloc[seq_len:].reset_index(drop=True)
        prices = df["Close"].iloc[seq_len:].values
        y_true = df[["Target_Tomorrow","Target_3_Days","Target_Next_Week"]].iloc[seq_len:].values
    else:
        X = df[feature_cols].values
        dates = df["Date"].reset_index(drop=True)
        prices = df["Close"].values
        y_true = df[["Target_Tomorrow","Target_3_Days","Target_Next_Week"]].values

    # Scale
    if seq_len > 1:
        n_feat = X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    # Predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        preds = model(X_tensor).cpu().numpy()

    # Show last preds
    if verbose:
        last = len(preds) - 1
        print(f"🔮 Prediction Tomorrow:     {preds[last,0]:.4f}")
        print(f"🔮 Prediction Next 3 Days:  {preds[last,1]:.4f}")
        print(f"🔮 Prediction Next Week:    {preds[last,2]:.4f}")

    # Backtest logic
    cash = float(MODEL_PARAMS.get("initial_balance", 10000))
    shares = 0
    last_price = 0.0
    max_loss_per_trade = 0.0
    profit_target = float(MODEL_PARAMS.get("profit_target", 0.05))
    trailing_stop = float(MODEL_PARAMS.get("trailing_stop", 0.03))
    highest_price = None
    trades = []

    for i in range(len(prices) - 1):
        date = dates[i]
        price = float(prices[i])
        signal = preds[i]
        avg_ret = float(signal.mean())
        stop_loss = shares > 0 and price < last_price * 0.97
        if shares > 0:
            highest_price = price if highest_price is None or price > highest_price else highest_price
        trailing_stop_hit = shares > 0 and highest_price is not None and price < highest_price * (1 - trailing_stop)
        profit_target_hit = shares > 0 and price >= last_price * (1 + profit_target)
        fee = float(max(
            MODEL_PARAMS.get("buy_sell_fee_per_share", 0.01) * shares,
            MODEL_PARAMS.get("minimum_fee", 1)
        )) if shares > 0 else 0.0

        # BUY
        if shares == 0 and avg_ret > MODEL_PARAMS.get("buying_threshold", 0.0) and signal[2] > signal[0]:
            shares = math.floor((cash - fee) / price)
            cash -= shares * price + fee
            last_price = price
            highest_price = price
            trades.append({
                "Date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "Type": "BUY",
                "Price": price,
                "Portfolio": float(cash + shares * price),
                "Change_%": None,
                "Tax": None
            })
        # SELL
        elif shares > 0 and (
            avg_ret < MODEL_PARAMS.get("selling_threshold", 0.0)
            or stop_loss
            or trailing_stop_hit
            or profit_target_hit
        ):
            tax = float(MODEL_PARAMS.get("tax_rate", 0.25) * (price - last_price) * shares)
            cash += shares * price - fee - tax
            change_pct = float((price - last_price) / last_price * 100)
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

    # Force final sell
    if shares > 0:
        final_date = dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1]
        final_price = float(prices[-1])
        tax = float(MODEL_PARAMS.get("tax_rate", 0.25) * (final_price - last_price) * shares)
        cash += shares * final_price - tax
        change_pct = float((final_price - last_price) / last_price * 100)
        max_loss_per_trade = min(max_loss_per_trade, change_pct)
        trades.append({
            "Date": final_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "Type": "SELL",
            "Price": final_price,
            "Portfolio": float(cash),
            "Change_%": change_pct,
            "Tax": tax
        })
        shares = 0

    # Summary metrics
    ticker_change = float((prices[-1] / prices[0] - 1) * 100)
    net_profit = float((cash / MODEL_PARAMS.get("initial_balance", 10000) - 1) * 100)

    # Print final metrics
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


def optimize_signal_params(request_data: UpdateIndicatorsData, param_grid: dict) -> pd.DataFrame:
    """
    Runs a grid search over signal parameters and returns a DataFrame sorted by net_profit.

    param_grid keys: buying_threshold, selling_threshold, profit_target, trailing_stop
    """
    results = []
    # Save original params
    original = {k: MODEL_PARAMS.get(k) for k in [
        "buying_threshold", "selling_threshold", "profit_target", "trailing_stop"
    ]}

    for bt, st, pt, ts in itertools.product(
        param_grid.get("buying_threshold", [original["buying_threshold"]]),
        param_grid.get("selling_threshold", [original["selling_threshold"]]),
        param_grid.get("profit_target", [original["profit_target"]]),
        param_grid.get("trailing_stop", [original["trailing_stop"]])
    ):
        # Override\        
        MODEL_PARAMS.update({
            "buying_threshold": bt,
            "selling_threshold": st,
            "profit_target": pt,
            "trailing_stop": ts
        })
        res = backtest_model(request_data, verbose=False)
        results.append({
            "buying_threshold": bt,
            "selling_threshold": st,
            "profit_target": pt,
            "trailing_stop": ts,
            "net_profit": res["net_profit"],
            "ticker_change": res["ticker_change"],
            "max_loss_per_trade": res["max_loss_per_trade"],
            "num_trades": len(res["trades_signals"])
        })
    # restore original
    MODEL_PARAMS.update(original)

    df = pd.DataFrame(results)
    return df.sort_values("net_profit", ascending=False).reset_index(drop=True)
