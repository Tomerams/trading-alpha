import os
import torch
import torch.nn as nn
import logging
import joblib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from config import FEATURE_COLUMNS, MODEL_PARAMS
from utilities import get_model, time_based_split
from data_processing import get_data
from model_backtest import backtest_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_and_evaluate(
    stock_ticker: str,
    start_date: str,
    end_date: str,
    model_type: str,
    selected_features: list = None,
    save_model: bool = True,
):
    if selected_features is None:
        selected_features = FEATURE_COLUMNS

    indicators_data = get_data(stock_ticker, start_date, end_date)

    train_df, test_df, backtest_df = time_based_split(indicators_data)

    FEATURE_COLUMNS_FOR_TRAINING = [col for col in train_df.columns if col not in ["Date", "Target_Tomorrow", "Target_3_Days", "Target_Next_Week"]]
    TARGET_COLUMNS = ["Target_Tomorrow", "Target_3_Days", "Target_Next_Week"]

    X_train, y_train = train_df[FEATURE_COLUMNS_FOR_TRAINING], train_df[TARGET_COLUMNS]
    X_test, y_test = test_df[FEATURE_COLUMNS_FOR_TRAINING], test_df[TARGET_COLUMNS]
    X_backtest, y_backtest = backtest_df[FEATURE_COLUMNS_FOR_TRAINING], backtest_df[TARGET_COLUMNS]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_backtest = scaler.transform(X_backtest)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, X_train.shape[1])
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, X_test.shape[1])
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    X_backtest_tensor = torch.tensor(X_backtest, dtype=torch.float32).view(-1, 1, X_backtest.shape[1])
    y_backtest_tensor = torch.tensor(y_backtest.values, dtype=torch.float32)

    # Create DataLoader for Training
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    logging.info("🚀 Training a new model")
    model = get_model(input_size=len(FEATURE_COLUMNS_FOR_TRAINING), model_type=model_type, output_size=3)
    
    # Optimizer & Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_function = nn.MSELoss()

    # Training Loop
    total_epochs = MODEL_PARAMS["epochs"]
    for epoch in range(total_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.view(batch_X.shape[0], 1, batch_X.shape[2])  # Reshape for PyTorch model
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{total_epochs} - Loss: {total_loss:.4f}")

    # ✅ **Validation on Test Set**
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = loss_function(test_predictions, y_test_tensor).item()
        logging.info(f"📊 Test Loss: {test_loss:.4f}")

    # ✅ **Evaluation on Backtest Set**
    with torch.no_grad():
        backtest_predictions = model(X_backtest_tensor)
        backtest_loss = loss_function(backtest_predictions, y_backtest_tensor).item()
        logging.info(f"📊 Backtest Loss: {backtest_loss:.4f}")

    # Save Model & Scaler
    if save_model:
        os.makedirs("models", exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict()},  # ✅ Save state_dict properly
            f"models/trained_model-{stock_ticker}-{model_type}.pt",
        )
        joblib.dump(scaler, f"models/scaler-{stock_ticker}-{model_type}.pkl")
        joblib.dump(FEATURE_COLUMNS_FOR_TRAINING, f"models/features-{stock_ticker}-{model_type}.pkl")
        logging.info(f"✅ Model & Scaler saved for {stock_ticker} ({model_type})")

    # Backtest Model
    logging.info("📊 Running backtest for trained model...")
    net_profit, trade_df, ticker_change, max_loss_per_trade = backtest_model(
        stock_ticker, start_date, end_date, model, scaler, FEATURE_COLUMNS_FOR_TRAINING
    )

    print(FEATURE_COLUMNS_FOR_TRAINING)

    return {
        "features": FEATURE_COLUMNS_FOR_TRAINING,
        "test_loss": test_loss,
        "backtest_loss": backtest_loss,
        "net_profit": net_profit,
        "ticker_change": ticker_change,
        "max_loss_per_trade": max_loss_per_trade
    }

if __name__ == "__main__":
    stock_ticker = "TQQQ"
    start_date = "2011-05-05"
    end_date = "2023-01-01"
    model_type = "TransformerRNN"

    print(f"🎯 Training {model_type} model for {stock_ticker} from {start_date} to {end_date}...")
    training_result = train_and_evaluate(stock_ticker, start_date, end_date, model_type)
    print(f"✅ Training complete! Results: {training_result}")
