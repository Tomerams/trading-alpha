import pandas as pd
import math
import logging
from typing import List, Dict, Any
from config import MODEL_PARAMS


def decide_action_meta(meta_model, preds, target_cols, i):
    row = pd.DataFrame([preds[i]], columns=[f"Pred_{col}" for col in target_cols])
    action = meta_model.predict(row)[0]
    return action


def simulate_trades(
    dates: pd.DatetimeIndex,
    prices: pd.Series,
    preds: Any,
    target_cols: List[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Runs the trade simulation loop and returns performance metrics and trades.

    :param dates: series of datetime indices
    :param prices: series of closing prices
    :param preds: model predictions array
    :param target_cols: list of target column names
    :param verbose: whether to log progress
    :return: dict with ticker_change, net_profit, max_loss_per_trade, trades_signals
    """
    cash = MODEL_PARAMS["initial_balance"]
    shares = 0
    last_price = 0.0
    highest_price = None
    max_loss = 0.0
    trades: List[Dict[str, Any]] = []

    stop_pct = MODEL_PARAMS["stop_loss_pct"]
    profit_tgt = MODEL_PARAMS["profit_target"]
    trail_stop = MODEL_PARAMS["trailing_stop"]
    fee_share = MODEL_PARAMS["buy_sell_fee_per_share"]
    min_fee = MODEL_PARAMS["minimum_fee"]
    tax_rate = MODEL_PARAMS["tax_rate"]

    n_steps = len(dates) - 1
    logging.info(f"Entering trade loop for {n_steps} steps")

    for i in range(n_steps):
        date = dates[i]
        price = float(prices.iloc[i])
        action = decide_action_meta(None, preds, target_cols, i)

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

    # finalize open position
    if shares:
        date = dates[-1]
        price = float(prices.iloc[-1])
        tax = tax_rate * (price - last_price) * shares
        cash += shares * price - tax
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

    ticker_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    net_ret = (cash / MODEL_PARAMS["initial_balance"] - 1) * 100

    if verbose:
        logging.info(
            f"Ticker Change: {ticker_ret:.2f}% | Portfolio Return: {net_ret:.2f}% | Max Loss: {max_loss:.2f}%"
        )

    return {
        "ticker_change": ticker_ret,
        "net_profit": net_ret,
        "max_loss_per_trade": max_loss,
        "trades_signals": trades,
    }
