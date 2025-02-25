import logging
from model_training import train_and_evaluate
from prediction import predict_model
from model_backtest import backtest_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_trading_pipeline(
    ticker,
    start_date,
    end_date,
    model_file,
    model_type,
    train=False,
    test=False,
    predict=False,
):
    """Runs the trading pipeline based on the selected options."""
    logging.info("Starting trading pipeline...")

    if train:
        train_and_evaluate(ticker, start_date, end_date, model_type, features=None)

    if predict:
        predict_model(ticker, start_date)

    if test:
        backtest_model(ticker, start_date, end_date, model_file)

    logging.info("Trading pipeline completed.")


if __name__ == "__main__":
    CONFIG = {
        "ticker": "FNGU",
        "start_date": "2018-06-06",
        "end_date": "2025-01-01",
        "train": True,
        "test": True,
        "predict": False,
        "model_type": "CNNLSTM",  # Choose "LSTM", "Transformer", or "CNNLSTM"
    }

    CONFIG["model_file"] = f"models/trained_model-{CONFIG['ticker']}.pkl"

    run_trading_pipeline(**CONFIG)
