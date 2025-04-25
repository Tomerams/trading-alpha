import pandas as pd
import matplotlib.pyplot as plt
from data_processing import get_data

# Load backtest results
trade_df = pd.read_csv("data/backtest_results.csv")

trade_df["Date"] = pd.to_datetime(trade_df["Date"]).dt.normalize()  # remove hours


# Fetch stock data
ticker = "FNGA"
start_date = trade_df["Date"].min().strftime("%Y-%m-%d")
end_date = trade_df["Date"].max().strftime("%Y-%m-%d")

print(f"📅 Fetching stock data for {ticker} from {start_date} to {end_date}...")

stock_data = get_data(ticker, start_date=start_date, end_date=end_date)

stock_data["Date"] = pd.to_datetime(
    stock_data["Date"]
).dt.normalize()  # normalize as well


# Merge by date
merged_df = pd.merge(stock_data, trade_df, on="Date", how="left")

buy_trades = merged_df[merged_df["Trade Type"] == "BUY"]
sell_trades = merged_df[merged_df["Trade Type"] == "SELL"]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(merged_df["Date"], merged_df["Close"], label="Stock Price", color="blue")

plt.scatter(
    buy_trades["Date"],
    buy_trades["Close"],
    color="green",
    marker="^",
    label="BUY",
    s=100,
)
plt.scatter(
    sell_trades["Date"],
    sell_trades["Close"],
    color="red",
    marker="v",
    label="SELL",
    s=100,
)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Trading Strategy for {ticker}")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
