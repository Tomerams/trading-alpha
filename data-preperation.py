import yfinance as yf
import requests
import time
import os

FMP_API_KEY = "YOUR_FMP_API_KEY"
DATA_DIR = "data/nasdaq100_history"

# ✅ Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# ✅ Step 1: Get NASDAQ-100 tickers for each year
def get_nasdaq100_tickers_by_year(start_year, end_year):
    historical_tickers = {}

    for year in range(start_year, end_year + 1):
        url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?date={year}-12-31&apikey={FMP_API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ API request failed for {year}, skipping...")
            continue

        data = response.json()
        tickers = [item["symbol"] for item in data if "symbol" in item]
        historical_tickers[year] = tickers

        time.sleep(1)  # ✅ Prevent rate limits

    return historical_tickers


# ✅ Step 2: Download and Save Historical Data
def fetch_and_save_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")

        if hist.empty:
            print(f"❌ No data for {ticker}, skipping...")
            return

        file_path = f"{DATA_DIR}/{ticker}.csv"
        hist.to_csv(file_path)
        print(f"✅ Saved data for {ticker}")

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")


# ✅ Step 3: Run the script
start_year = 1992
end_year = 2024

nasdaq100_by_year = get_nasdaq100_tickers_by_year(start_year, end_year)

# ✅ Flatten all tickers to a unique list
all_tickers = list(
    set(ticker for tickers in nasdaq100_by_year.values() for ticker in tickers)
)

# ✅ Download data for all tickers
for ticker in all_tickers:
    fetch_and_save_stock_data(ticker)

print("✅ All historical NASDAQ-100 stock data has been saved!")
