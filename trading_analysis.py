import yfinance as yf
import pandas as pd
import requests
import time
from pandas.tseries.offsets import QuarterEnd

FMP_API_KEY = "6WlMvS48YWI28ZBXQZvDrmmtHKpHAZIV"


def get_all_nasdaq100_tickers(start_date, end_date):
    unique_tickers = set()

    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for quarter in ["03-31", "06-30", "09-30", "12-31"]:
            formatted_date = f"{year}-{quarter}"
            url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?date={formatted_date}&apikey={FMP_API_KEY}"
            response = requests.get(url)

            if response.status_code != 200:
                print(f"❌ API request failed for {formatted_date}, skipping...")
                continue

            data = response.json()
            tickers = [item["symbol"] for item in data if "symbol" in item]
            unique_tickers.update(tickers)

            time.sleep(1)  # ✅ Prevent rate limits

    return list(unique_tickers)


def fetch_analyst_growth_bulk(all_tickers):
    print(f"📌 Fetching growth for {len(all_tickers)} tickers in one request...")

    stock_data = yf.Tickers(all_tickers)  # ✅ Batch request all tickers

    growth_data = {}
    for ticker in all_tickers:
        try:
            info = stock_data.tickers[ticker].info
            growth = info.get("earningsGrowth", 0)
            growth_data[ticker] = growth
        except Exception as e:
            print(f"❌ Error fetching growth for {ticker}: {e}")
            growth_data[ticker] = 0  # Default to 0

    print("📌 Growth Data Collected!")
    return growth_data


def fetch_quarterly_returns_bulk(tickers, quarter):
    quarter_start = quarter.strftime("%Y-%m-%d")
    quarter_end = (quarter + QuarterEnd(0)).strftime("%Y-%m-%d")  # ✅ Correct end date

    print(
        f"📌 Fetching stock data for {quarter_start} to {quarter_end} for tickers: {tickers}"
    )

    hist = yf.download(
        tickers,
        start=quarter_start,
        end=quarter_end,
        interval="3mo",
        group_by="ticker",
        threads=True,
    )

    quarterly_returns = {}
    for ticker in tickers:
        try:
            stock_data = hist[ticker] if isinstance(hist, pd.DataFrame) else hist
            close_prices = stock_data["Close"]

            if close_prices.empty or len(close_prices) < 2:
                quarterly_returns[ticker] = 0
            else:
                prev_close = close_prices.iloc[0]
                last_close = close_prices.iloc[-1]
                returns = ((last_close - prev_close) / prev_close) * 100
                quarterly_returns[ticker] = returns
        except (KeyError, IndexError):
            quarterly_returns[ticker] = 0  # ✅ Default to 0 if data is missing

    return quarterly_returns


def fetch_index_quarterly_returns(ticker, start_date, end_date):
    hist = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="3mo",
        group_by="ticker",
        threads=True,
    )

    print("📌 Quarterly Data Directly Fetched from Yahoo Finance:")
    print(hist.head())

    if hist.empty:
        print(f"❌ No valid data for {ticker}, returning default 0 series")
        return pd.Series(0, dtype="float64")

    close_column = [col for col in hist.columns if "Close" in col]

    if not close_column:
        raise KeyError(f"❌ No 'Close' column found in data for {ticker}!")

    close_prices = hist[close_column[0]]
    returns = close_prices.pct_change() * 100  # ✅ Calculate quarterly returns

    print("\n📌 Quarterly Returns Calculated:")
    print(returns)

    return returns.fillna(0)


# ✅ Process Quarterly Data & Select Top Stocks
start_date = "2024-01-01"
end_date = "2024-12-31"

all_tickers = get_all_nasdaq100_tickers(start_date, end_date)
growth_estimates = fetch_analyst_growth_bulk(all_tickers)

sp500_quarterly_returns = fetch_index_quarterly_returns("^GSPC", start_date, end_date)
nasdaq100_quarterly_returns = fetch_index_quarterly_returns(
    "^NDX", start_date, end_date
)

sp500_quarterly_returns.index = pd.to_datetime(sp500_quarterly_returns.index)
sp500_quarterly_returns = sp500_quarterly_returns[
    sp500_quarterly_returns.index >= start_date
]

all_results = []

for quarter in sp500_quarterly_returns.index:
    print(f"📌 Processing {quarter}...")

    try:
        top_growth_tickers = sorted(
            growth_estimates, key=growth_estimates.get, reverse=True
        )[:10]

        if not top_growth_tickers:
            raise Exception(f"No valid top 10 tickers found for {quarter}")

        top_returns = fetch_quarterly_returns_bulk(top_growth_tickers, quarter)

        # ✅ Fix: Ensure we only calculate mean for valid returns
        top_returns_values = list(top_returns.values())
        avg_return = sum(top_returns_values) / len(
            [x for x in top_returns_values if x != 0]
        )

        print(f"✅ Top 10 Tickers for {quarter}: {top_growth_tickers}")
        print(f"✅ Average Return for Top 10: {avg_return}")

        quarter_data = {
            "Quarter": f"Q{quarter.quarter} {quarter.year}",
            "S&P 500 Quarterly Return (%)": sp500_quarterly_returns.get(quarter, 0),
            "NASDAQ-100 Quarterly Return (%)": nasdaq100_quarterly_returns.get(
                quarter, 0
            ),
            "Top 10 Tickers": ", ".join(top_growth_tickers),
            "Average Return of Top 10 (%)": avg_return,
        }

        all_results.append(quarter_data)

    except Exception as e:
        print(f"❌ Skipping {quarter} due to error: {e}")

comparison_df = pd.DataFrame(all_results)
comparison_df.to_csv("data/quarterly_growth_top10_dynamic.csv", index=False)

print("✅ Data has been saved to 'data/quarterly_growth_top10_dynamic.csv'")
