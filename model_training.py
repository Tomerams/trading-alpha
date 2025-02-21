import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import FEATURE_COLUMNS
from utilities import get_data, get_model
from model_backtest import backtest_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_and_evaluate(
    stock_ticker: str,
    start_date: str,
    end_date: str,
    model_type: str,
    selected_features: list = None,
    save_model: bool = True,
):
    """Train a new trading model and evaluate it using backtesting."""
    if selected_features is None:
        selected_features = FEATURE_COLUMNS

    historical_data = get_data(stock_ticker, start_date, end_date).dropna()
    existing_features = [f for f in selected_features if f in historical_data.columns]
    if not existing_features:
        logging.warning(f"Skipping due to missing features: {selected_features}")
        return {"features": selected_features, "net_profit": None, "stock_profit": None}

    X = historical_data[existing_features].values
    y = historical_data["target"].values

    print(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        X_test, dtype=torch.float32
    )
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(
        -1, 1
    ), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )

    logging.info("🚀 Training a new model (ignoring existing models)")
    model = get_model(input_size=len(existing_features), model_type=model_type)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f}")

    if save_model:
        os.makedirs("models", exist_ok=True)  # Ensure directory exists
        torch.save(
            {"model_state_dict": model.state_dict()},  # ✅ Correct format
            f"models/trained_model-{stock_ticker}-{model_type}.pkl",
        )
        joblib.dump(scaler, f"models/scaler-{stock_ticker}-{model_type}.pkl")
        joblib.dump(
            selected_features, f"models/features-{stock_ticker}-{model_type}.pkl"
        )

        print(f"✅ Model and scaler saved for {stock_ticker} ({model_type})")

    logging.info("📊 Running backtest for trained model...")

    net_profit, stock_profit = backtest_model(
        stock_ticker, start_date, end_date, model, scaler, selected_features
    )

    return {
        "features": existing_features,
        "net_profit": net_profit,
        "stock_profit": stock_profit,
    }


if __name__ == "__main__":
    stock_ticker = "FNGU"
    start_date = "2019-01-01"
    end_date = "2023-01-01"
    model_type = "TCN"

    print(
        f"🎯 Training {model_type} model for {stock_ticker} from {start_date} to {end_date}..."
    )
    training_result = train_and_evaluate(stock_ticker, start_date, end_date, model_type)
    print(f"✅ Training complete! Results: {training_result}")
