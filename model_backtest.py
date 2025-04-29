import math
import numpy as np
import pandas as pd
import torch
import logging
from data_processing import get_data
from utilities import load_model
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
from config import MODEL_PARAMS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def backtest_model(
    stock_ticker: str,
    start_date: str,
    end_date: str,
    trained_model: torch.nn.Module,
    data_scaler,
    selected_features: list,
    use_leverage: bool = False,
    trade_ticker: str = None,
    buying_threshold: float = 0.0,
    selling_threshold: float = 0.0,
    verbose: bool = True,
    seq_len: int = 1,
    max_history: int = None,
):
    logging.info(
        f"📊 Fetching data for {stock_ticker} from {start_date} to {end_date}..."
    )
    df = get_data(stock_ticker, start_date, end_date)
    if df is None or df.empty:
        raise ValueError("No data for backtest.")
    if max_history:
        df = df.tail(max_history).reset_index(drop=True)
    logging.info(f"✅ Data loaded with shape: {df.shape}")

    df["traded_close"] = df["Close"]
    df["Target_Tomorrow"] = df["Close"].shift(-1)
    df["Target_3_Days"] = df["Close"].shift(-3)
    df["Target_Next_Week"] = df["Close"].shift(-5)
    df.dropna(inplace=True)

    if seq_len > 1:
        X = np.array(
            [
                df[selected_features].iloc[i : i + seq_len].values
                for i in range(len(df) - seq_len)
            ]
        )
        traded_close = df["traded_close"].values[seq_len:]
        dates = df["Date"].values[seq_len:]
    else:
        X = df[selected_features].values
        traded_close = df["traded_close"].values
        dates = df["Date"].values

    if seq_len > 1:
        n_feat = X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = data_scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = data_scaler.transform(X)

    trained_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1:
            X_tensor = X_tensor.unsqueeze(1)
        preds = trained_model(X_tensor).cpu().numpy()

    if verbose:
        last = len(preds) - 1
        print(f"🔮 Prediction Tomorrow:     {preds[last, 0]:.4f}")
        print(f"🔮 Prediction Next 3 Days:  {preds[last, 1]:.4f}")
        print(f"🔮 Prediction Next Week:    {preds[last, 2]:.4f}")

    cash = MODEL_PARAMS.get("initial_balance", 10000)
    shares = 0
    last_buy_price = None
    max_loss_per_trade = 0
    trade_log = []

    for i in range(len(traded_close) - 1):
        date = dates[i]
        price = traded_close[i]
        avg_ret = preds[i].mean()
        stop_loss = shares > 0 and price < last_buy_price * 0.97
        fee = (
            max(
                MODEL_PARAMS.get("buy_sell_fee_per_share", 0.01) * shares,
                MODEL_PARAMS.get("minimum_fee", 1),
            )
            if shares > 0
            else 0
        )

        if shares == 0 and avg_ret > buying_threshold and preds[i, 2] > preds[i, 0]:
            shares = math.floor((cash - fee) / price)
            cash -= shares * price + fee
            last_buy_price = price
            trade_log.append(
                [date, "BUY", price, cash + shares * price, None, None, None]
            )
        elif shares > 0 and (avg_ret < selling_threshold or stop_loss):
            tax = MODEL_PARAMS.get("tax_rate", 0.25) * (price - last_buy_price) * shares
            cash += shares * price - fee - tax
            change_pct = (price - last_buy_price) / last_buy_price * 100
            port_pct = (
                (cash - MODEL_PARAMS.get("initial_balance", 10000))
                / MODEL_PARAMS.get("initial_balance", 10000)
                * 100
            )
            max_loss_per_trade = min(max_loss_per_trade, change_pct)
            trade_log.append([date, "SELL", price, cash, change_pct, port_pct, tax])
            shares = 0

    final_val = cash + shares * traded_close[-1]
    net_profit = (final_val / MODEL_PARAMS.get("initial_balance", 10000) - 1) * 100
    ticker_change = ((traded_close[-1] / traded_close[0]) - 1) * 100

    dir_actual = np.vstack(
        [
            (df["Target_Tomorrow"].values[seq_len:] > traded_close).astype(int),
            (df["Target_3_Days"].values[seq_len:] > traded_close).astype(int),
            (df["Target_Next_Week"].values[seq_len:] > traded_close).astype(int),
        ]
    ).T
    dir_pred = (preds > 0).astype(int)

    horizons = ["Tomorrow", "3 Days", "Next Week"]
    for idx, name in enumerate(horizons):
        prec = precision_score(dir_actual[:, idx], dir_pred[:, idx], zero_division=0)
        rec = recall_score(dir_actual[:, idx], dir_pred[:, idx], zero_division=0)
        mae = mean_absolute_error(
            df[
                f'Target_{"Tomorrow" if idx==0 else "3_Days" if idx==1 else "Next_Week"}'
            ].values[seq_len:],
            preds[:, idx],
        )
        if verbose:
            print(f"🌟 Precision ({name}): {prec:.4f}")
            print(f"🔍 Recall    ({name}): {rec:.4f}")
            print(f"📊 MAE       ({name}): {mae:.4f}")
    if verbose:
        print(f"📉 Ticker Change: {ticker_change:.2f}%")
        print(f"📈 Portfolio    : {net_profit:.2f}%")
        print(f"⚠️ Max Loss     : {max_loss_per_trade:.2f}%")

    pd.DataFrame(
        trade_log,
        columns=[
            "Date",
            "Type",
            "Price",
            "Portfolio Value",
            "Price Change %",
            "Portfolio %",
            "Tax",
        ],
    ).to_csv("data/backtest_trades.csv", index=False)

    return net_profit, pd.DataFrame(trade_log), ticker_change, max_loss_per_trade


if __name__ == "__main__":
    ticker = input(
        "🔎 Signal ticker (default QQQ): "
    ).strip().upper() or MODEL_PARAMS.get("ticker", "QQQ")
    model, scaler, feature_cols, seq_len = load_model(
        ticker, MODEL_PARAMS.get("model_type", "TransformerRNN")
    )
    print(f"▶️ Loaded model seq_len={seq_len}, features={len(feature_cols)} cols")
    start = MODEL_PARAMS.get("start_date")
    end = MODEL_PARAMS.get("end_date")
    backtest_model(
        stock_ticker=ticker,
        start_date=start,
        end_date=end,
        trained_model=model,
        data_scaler=scaler,
        selected_features=feature_cols,
        buying_threshold=MODEL_PARAMS.get("buying_threshold", 0.0),
        selling_threshold=MODEL_PARAMS.get("selling_threshold", 0.0),
        seq_len=seq_len,
    )
