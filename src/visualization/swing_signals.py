import io
import math
import matplotlib.pyplot as plt
import pandas as pd

from routers.routers_entities import UpdateIndicatorsData
from data.data_utilities import get_data
from models.prediction_utilities import add_swing_column
from config import MODEL_PARAMS


def fetch_price_history(
    ticker: str, start: str = None, end: str = None
) -> pd.DataFrame:
    rd = UpdateIndicatorsData(stock_ticker=ticker, start_date=start, end_date=end)
    df = get_data(rd)

    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def simulate_swing_trades(
    df: pd.DataFrame, close_col: str = "Close", signal_col: str = "swing_signal"
) -> pd.DataFrame:
    """
    Simulate trades by buying at troughs (signal=1) and selling at peaks (signal=-1).
    Adds columns:
      - trade_signal: 1=buy, -1=sell, 0=no action
      - cash: remaining cash after each step
      - shares: number of shares held
      - portfolio: total portfolio value (cash + shares*price)
    """
    initial_cash = MODEL_PARAMS["initial_balance"]
    cash = initial_cash
    shares = 0

    df = df.copy()
    df["trade_signal"] = 0
    df["cash"] = 0.0
    df["shares"] = 0
    df["portfolio"] = 0.0

    for idx, row in df.iterrows():
        price = float(row[close_col])
        signal = row[signal_col]

        # BUY at trough
        if signal == 1 and shares == 0:
            # buy as many as possible
            shares = math.floor(cash / price)
            cash -= shares * price
            df.at[idx, "trade_signal"] = 1

        # SELL at peak
        elif signal == -1 and shares > 0:
            cash += shares * price
            df.at[idx, "trade_signal"] = -1
            shares = 0

        # record state
        df.at[idx, "cash"] = cash
        df.at[idx, "shares"] = shares
        df.at[idx, "portfolio"] = cash + shares * price

    return df


def get_visualizations(
    ticker: str,
    start: str = None,
    end: str = None,
    window: int = 5,
    prominence: float = 0.01,
) -> bytes:
    # 1) Fetch raw prices
    df = fetch_price_history(ticker, start, end)

    # 2) Inject swing signals
    df = add_swing_column(
        df,
        close_col="Close",
        signal_col="swing_signal",
        window=window,
        prominence=prominence,
    )

    # 3) Simulate buy/sell trades
    df = simulate_swing_trades(df, close_col="Close", signal_col="swing_signal")

    # 4) Plot price and trade actions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close")

    buys = df.index[df["trade_signal"] == 1]
    sells = df.index[df["trade_signal"] == -1]

    ax.scatter(buys, df.loc[buys, "Close"], marker="^", label="Buy", s=100)
    ax.scatter(sells, df.loc[sells, "Close"], marker="v", label="Sell", s=100)

    ax.legend()
    ax.set_title(f"{ticker} Swing Trade Simulation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()
