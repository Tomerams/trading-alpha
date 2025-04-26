import itertools
import multiprocessing
import logging
import pandas as pd
from config import FEATURE_COLUMNS
from data_processing import get_data
from model_training import train_and_evaluate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_features(stock_ticker, start_date, end_date, model_type, features):
    return train_and_evaluate(
        stock_ticker, start_date, end_date, model_type, features, save_model=False
    )


def run_feature_experiments(stock_ticker, start_date, end_date, model_type):
    df = get_data(stock_ticker, start_date, end_date)
    feature_sets = [
        list(f for f in comb if f in df.columns)
        for r in range(1, len(FEATURE_COLUMNS) + 1)
        for comb in itertools.combinations(FEATURE_COLUMNS, r)
    ]
    feature_sets = [f for f in feature_sets if f]

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(
            evaluate_features,
            [
                (stock_ticker, start_date, end_date, model_type, features)
                for features in feature_sets
            ],
        )

    results_df = pd.DataFrame.from_records(results)

    # Debug: Print the first few rows and types
    print("🔍 DEBUG: Checking 'stock_profit' content before processing:")
    for index, value in results_df["stock_profit"].items():
        print(f"Row {index}: Type: {type(value)} - Value:\n{value}\n")

    results_df["stock_profit"] = results_df["stock_profit"].apply(
        lambda x: x["Portfolio Gain %"].dropna().iloc[-1]
        if isinstance(x, pd.DataFrame)
        and "Portfolio Gain %" in x.columns
        and not x["Portfolio Gain %"].dropna().empty
        else None  # Keep None for now
    )

    # Fill missing values with the average of valid stock profits
    mean_profit = results_df["stock_profit"].mean()
    results_df["stock_profit"].fillna(mean_profit, inplace=True)

    # Sort after fixing
    results_df = results_df[["features", "net_profit", "stock_profit"]].sort_values(
        by=["net_profit", "stock_profit"], ascending=[False, False]
    )

    results_df.to_csv(f"data/feature_performance_{model_type}.csv", index=False)

    return results_df


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ticker = "FNGA"
    start_date = "2020-01-01"
    end_date = "2023-05-05"
    model_type = "TransformerRNN"

    df_results = run_feature_experiments(ticker, start_date, end_date, model_type)
    print(df_results.head())
