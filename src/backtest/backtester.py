import logging
import time
import math
import numpy as np
import torch
import pandas as pd

from config import MODEL_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import load_model
from models.model_signals_decision_train import load_meta_model
from backtest.backtest_utilities import decide_action_meta

# Ensure root logger is configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def backtest_model(request_data, verbose=True):
    """
    Runs a backtest simulation using the base and (optional) meta models,
    with detailed logging around each step to pinpoint hangs or errors.
    """
    try:
        ticker = request_data.stock_ticker
        logging.info("▶️  backtest_model start for %s", ticker)
        t_all = time.time()

        # 1) Load base model & scaler
        logging.info("   • about to load base model & scaler")
        try:
            model_type = MODEL_PARAMS["model_type"]
            model, scaler, feature_cols = load_model(ticker, model_type)
            logging.info("   • load_model succeeded in %.2fs", time.time() - t_all)
        except Exception as e:
            logging.error("   • load_model error for %s: %s", ticker, e, exc_info=True)
            raise RuntimeError(f"Failed to load model for {ticker}")

        seq_len = MODEL_PARAMS["seq_len"]
        target_cols = MODEL_PARAMS["target_cols"]

        # 2) Fetch indicators
        logging.info("   • about to fetch indicator data")
        t0 = time.time()
        df = get_indicators_data(request_data)
        logging.info(
            "   • get_indicators_data returned in %.2fs — shape %s",
            time.time() - t0,
            df.shape,
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.dropna(inplace=True)

        # 3) Build X, dates, prices, true_vals
        logging.info("   • about to build X array")
        t1 = time.time()
        if seq_len > 1:
            X = np.stack(
                [
                    df[feature_cols].iloc[i : i + seq_len].values
                    for i in range(len(df) - seq_len)
                ]
            )
            dates = df["Date"].iloc[seq_len:].reset_index(drop=True)
            prices = df["Close"].iloc[seq_len:].reset_index(drop=True)
            true_vals = df[target_cols].iloc[seq_len:].values
        else:
            X = df[feature_cols].values
            dates = df["Date"].reset_index(drop=True)
            prices = df["Close"].reset_index(drop=True)
            true_vals = df[target_cols].values
        logging.info("   • built X in %.2fs — X.shape=%s", time.time() - t1, X.shape)

        # 4) Scale & infer
        logging.info("   • about to scale features")
        try:
            if seq_len > 1:
                n_feat = X.shape[2]
                X_flat = X.reshape(-1, n_feat)
                X_scaled = scaler.transform(X_flat).reshape(X.shape)
            else:
                X_scaled = scaler.transform(X)
            logging.info("   • scaling completed in %.2fs", time.time() - t1)
        except Exception as e:
            logging.error("   • scaling error: %s", e, exc_info=True)
            raise RuntimeError("Scaling of features failed")

        logging.info("   • about to convert to tensor & run inference")
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)

        model.eval()
        t2 = time.time()
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
        logging.info(
            "   • inference done in %.2fs — preds.shape=%s",
            time.time() - t2,
            preds.shape,
        )

        # 5) Load meta-model
        logging.info("   • about to load meta-model")
        t3 = time.time()
        try:
            meta_model = load_meta_model()
            status = "FOUND" if meta_model else "NOT FOUND"
        except Exception as e:
            logging.warning(
                "   • load_meta_model error: %s — proceeding without meta-model",
                e,
                exc_info=True,
            )
            meta_model = None
            status = "ERROR"
        logging.info(
            "   • load_meta_model returned in %.2fs — %s", time.time() - t3, status
        )

        # 6) Initialize state
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

        # 7) Trade loop
        n_steps = len(dates) - 1
        logging.info("   • entering trade loop for %d steps", n_steps)
        for i, date in enumerate(dates[:-1]):
            if i % 50 == 0:
                logging.info("      – loop step %d/%d", i, n_steps)

            price = float(prices.iloc[i])
            action = 1  # HOLD by default
            if meta_model:
                try:
                    action = decide_action_meta(meta_model, preds, target_cols, i)
                except Exception as e:
                    logging.warning(
                        "   • meta-model error at step %d: %s — defaulting to HOLD",
                        i,
                        e,
                        exc_info=True,
                    )

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

        logging.info("   • trade loop done in %.2fs", time.time() - t_all)

        # 8) Finalize any open position
        if shares:
            price = float(prices.iloc[-1])
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
        ticker_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        net_ret = (cash / MODEL_PARAMS["initial_balance"] - 1) * 100

        logging.info("✅ backtest_model complete in %.2fs", time.time() - t_all)
        return {
            "ticker_change": ticker_ret,
            "net_profit": net_ret,
            "max_loss_per_trade": max_loss,
            "trades_signals": trades,
        }

    except Exception:
        logging.exception("❌ backtest_model failed for %s", request_data.stock_ticker)
        raise
