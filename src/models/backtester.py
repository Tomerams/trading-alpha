import math
import numpy as np
import pandas as pd
import torch
import joblib
import lightgbm as lgb
from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_data
from models.model_utilities import load_model
from config import MODEL_PARAMS


def _train_gbm_on_preds(preds: np.ndarray, true_vals: np.ndarray, target_cols: list):
    """
    Train a GBM classifier on the neural-network predictions (stacking).
    preds: array of shape (n_samples, n_targets)
    true_vals: array of true shift_targets (e.g. actual returns)
    target_cols: names of preds columns
    """
    # assemble DataFrame
    df_signals = pd.DataFrame(preds, columns=target_cols)
    # binary label: 1 if actual next-day return > 0, else 0
    df_signals["label"] = (
        true_vals[:, target_cols.index("Target_Tomorrow")] > 0
    ).astype(int)
    gbm = lgb.LGBMClassifier(**MODEL_PARAMS["gbm_params"])
    gbm.fit(df_signals[target_cols], df_signals["label"])
    joblib.dump(gbm, "files/models/gbm_signal_model.pkl")
    return gbm


def _load_gbm():
    try:
        return joblib.load("files/models/gbm_signal_model.pkl")
    except FileNotFoundError:
        return None


def backtest_model(request_data: UpdateIndicatorsData, verbose: bool = True) -> dict:
    # 1) load model and artifacts
    ticker = request_data.stock_ticker
    model_type = MODEL_PARAMS["model_type"]
    model, scaler, feature_cols = load_model(ticker, model_type)
    seq_len = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]

    # 2) fetch data
    df = get_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)

    # 3) prepare input sequences
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

    # 4) scale
    if seq_len > 1:
        n_feat = X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(X)

    # 5) neural predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        preds = model(X_tensor).cpu().numpy()

    # 6) optional GBM stacking on preds
    gbm = None
    if MODEL_PARAMS["use_gbm_signals"]:
        gbm = _load_gbm() or _train_gbm_on_preds(preds, true_vals, target_cols)
        gbm_probs = gbm.predict_proba(pd.DataFrame(preds, columns=target_cols))[:, 1]

    # 7) backtest loop
    cash = MODEL_PARAMS["initial_balance"]
    shares = 0
    last_price = 0.0
    highest_price = None
    max_loss = 0.0
    trades = []

    # thresholds
    buy_thr = (
        MODEL_PARAMS["gbm_buy_threshold"]
        if MODEL_PARAMS["use_gbm_signals"]
        else MODEL_PARAMS["buying_threshold"]
    )
    sell_thr = (
        MODEL_PARAMS["gbm_sell_threshold"]
        if MODEL_PARAMS["use_gbm_signals"]
        else MODEL_PARAMS["selling_threshold"]
    )
    profit_tgt = MODEL_PARAMS["profit_target"]
    trail_stop = MODEL_PARAMS["trailing_stop"]
    stop_pct = MODEL_PARAMS["stop_loss_pct"]
    fee_share = MODEL_PARAMS["buy_sell_fee_per_share"]
    min_fee = MODEL_PARAMS["minimum_fee"]
    tax_rate = MODEL_PARAMS["tax_rate"]

    for i, date in enumerate(dates[:-1]):
        price = float(prices[i])
        signal = preds[i]
        avg_ret = float(gbm_probs[i] if gbm else signal.mean())

        if shares:
            highest_price = max(highest_price, price)

        stop_hit = shares and price < last_price * (1 - stop_pct)
        trail_hit = shares and price < highest_price * (1 - trail_stop)
        profit_hit = shares and price >= last_price * (1 + profit_tgt)

        fee = max(shares * fee_share, min_fee) if shares else 0.0

        # BUY
        if not shares and avg_ret > buy_thr:
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
        elif shares and avg_ret < sell_thr or stop_hit or trail_hit or profit_hit:
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

    # final exit
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
            f"📉 Ticker: {ticker_ret:.2f}%  📈 Port: {net_ret:.2f}%  ⚠️ Max Loss: {max_loss:.2f}%"
        )

    return {
        "ticker_change": ticker_ret,
        "net_profit": net_ret,
        "max_loss_per_trade": max_loss,
        "trades_signals": trades,
    }
