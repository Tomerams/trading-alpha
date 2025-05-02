import io
import pandas as pd
import matplotlib.pyplot as plt

from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_data
from models.backtester import backtest_model


def generate_trade_plot(request_data: UpdateIndicatorsData) -> io.BytesIO:
    """
    Generate a PNG plot of stock price overlaid with BUY/SELL signals from backtest.
    Returns a BytesIO buffer containing the image.
    """
    # 1. Run backtest to get trade signals
    result = backtest_model(request_data, verbose=False)
    trades = result.get("trades_signals", [])

    # 2. Fetch historical price data
    df = get_data(request_data)
    if df is None or df.empty:
        raise ValueError("No price data available for the given dates.")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close Price", linewidth=1)

    # 4. Overlay BUY/SELL signals
    for trade in trades:
        ts = pd.to_datetime(trade["Date"])
        price = trade["Price"]
        if trade["Type"] == "BUY":
            ax.scatter(
                ts, price, marker="^", color="green", s=100, zorder=5, label="BUY"
            )
        else:
            ax.scatter(
                ts, price, marker="v", color="red", s=100, zorder=5, label="SELL"
            )

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    ax.set_title(
        f"{request_data.stock_ticker} Trades ({request_data.start_date} → {request_data.end_date})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    fig.tight_layout()

    # 5. Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf
