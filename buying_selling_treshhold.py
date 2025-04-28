import numpy as np
import pandas as pd
from model_backtest import backtest_model
from utilities import load_model
from config import MODEL_PARAMS

if __name__ == "__main__":
    ticker_to_predict = MODEL_PARAMS.get("ticker", "QQQ")
    ticker_to_trade = "TQQQ"
    start_date = MODEL_PARAMS.get("start_date", "2020-01-01")
    end_date = MODEL_PARAMS.get("end_date", "2024-01-01")
    model_type = MODEL_PARAMS.get("model_type", "TransformerRNN")

    model, scaler, best_features, seq_len = load_model(ticker_to_predict, model_type)

    best_result = None
    results = []

    buying_thresholds = np.arange(0.002, 0.015, 0.002)
    selling_thresholds = np.arange(-0.015, -0.002, 0.002)

    for buy_th in buying_thresholds:
        for sell_th in selling_thresholds:
            print(f"🚀 Backtest: buy_th={buy_th:.4f}, sell_th={sell_th:.4f}")

            net_profit, trades_df, ticker_change, max_loss_per_trade = backtest_model(
                stock_ticker=ticker_to_predict,
                start_date=start_date,
                end_date=end_date,
                trained_model=model,
                data_scaler=scaler,
                selected_features=best_features,
                use_leverage=True,
                trade_ticker=ticker_to_trade,
                buying_threshold=buy_th,
                selling_threshold=sell_th,
                verbose=False,
                seq_len=seq_len,
            )

            result = {
                "Buying Threshold": buy_th,
                "Selling Threshold": sell_th,
                "Net Profit %": net_profit,
                "Portfolio %": net_profit,
                "Ticker %": ticker_change,
                "Max Loss %": max_loss_per_trade,
                "Trades": len(trades_df),
            }
            results.append(result)

            if (
                best_result is None
                or result["Net Profit %"] > best_result["Net Profit %"]
            ):
                best_result = result

    results_df = pd.DataFrame(results)
    print("\n📊 All Results:")
    print(
        results_df.sort_values(by="Net Profit %", ascending=False).reset_index(
            drop=True
        )
    )

    print("\n🏆 Best Result:")
    print(best_result)
