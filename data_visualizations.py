import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ✅ **Step 1: Load trade history**
trade_df = pd.read_csv("data/backtest_results.csv")
trade_df["Date"] = pd.to_datetime(trade_df["Date"]).dt.date  # Convert to YYYY-MM-DD format

# ✅ **Step 2: Fetch stock price data from Yahoo Finance**
ticker = "FNGU"
start_date = trade_df["Date"].min().strftime("%Y-%m-%d")
end_date = trade_df["Date"].max().strftime("%Y-%m-%d")

print(f"📅 Fetching stock data for {ticker} from {start_date} to {end_date}...")

# ✅ **Download stock data**
stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# ✅ **Fix Multi-Level Index Issue in stock_data**
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)  # Flatten multi-index columns

stock_data.reset_index(inplace=True)  # Reset index to make "Date" a column
stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date  # Convert Date to YYYY-MM-DD format

# ✅ **Step 3: Check Data Formatting Before Merge**
print("📊 First 5 Dates in stock_data:\n", stock_data[["Date", "Close"]].head())
print("📊 First 5 Dates in trade_df:\n", trade_df[["Date", "Trade Type"]].head())

# ✅ **Step 4: Merge stock price data with trade data**
merged_df = pd.merge(stock_data[["Date", "Close"]], trade_df, on="Date", how="left")

# ✅ **Step 5: Extract Buy and Sell Trades**
buy_trades = merged_df[merged_df["Trade Type"] == "BUY"]
sell_trades = merged_df[merged_df["Trade Type"] == "SELL"]

# ✅ **Step 6: Plot Stock Price**
plt.figure(figsize=(12, 6))
plt.plot(merged_df["Date"], merged_df["Close"], label="Stock Price", color="blue")

# ✅ **Step 7: Plot Buy and Sell Signals**
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

# ✅ **Step 8: Customize the Plot**
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Trading Strategy for {ticker}")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

# ✅ **Step 9: Show the Plot**
plt.show()
