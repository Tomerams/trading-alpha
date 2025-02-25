import os
import torch
import torch.nn as nn
import logging
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import FEATURE_COLUMNS, MODEL_PARAMS
from utilities import get_model
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

    historical_data = get_data(stock_ticker, start_date, end_date).dropna()
    data = historical_data[selected_features + ["Target_Tomorrow", "Target_3_Days", "Target_Next_Week"]].dropna()
    X = data[selected_features].values
    y = data[["Target_Tomorrow", "Target_3_Days", "Target_Next_Week"]].values  # Multi-target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    logging.info("🚀 Training a new model (ignoring existing models)")
    model = get_model(input_size=len(selected_features), model_type=model_type)
    if hasattr(model, "parameters"):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_function = nn.MSELoss()
        use_pytorch = True
    else:
        model.compile(optimizer="adam", loss="mse")
        use_pytorch = False

    if use_pytorch:
        for epoch in range(MODEL_PARAMS["epochs"]):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f"Epoch {epoch+1}/{MODEL_PARAMS['epochs']} - Loss: {total_loss:.4f}")
    else:
        model.fit(X_train.numpy(), y_train.numpy(), epochs=MODEL_PARAMS["epochs"], batch_size=32)

    if save_model:
        os.makedirs("models", exist_ok=True)

        if use_pytorch:
            torch.save({"model_state_dict": model.state_dict()}, f"models/trained_model-{stock_ticker}-{model_type}.pkl")
        else:
            model.save(f"models/trained_model-{stock_ticker}-{model_type}.h5")

        joblib.dump(scaler, f"models/scaler-{stock_ticker}-{model_type}.pkl")
        joblib.dump(selected_features, f"models/features-{stock_ticker}-{model_type}.pkl")

        print(f"✅ Model and scaler saved for {stock_ticker} ({model_type})")

    logging.info("📊 Running SHAP Feature Importance Analysis...")

    # ==== SHAP FEATURE IMPORTANCE ====
    def model_wrapper(input_data):
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            return model(input_tensor).numpy()

    explainer = shap.Explainer(model_wrapper, X_test.numpy())  # Convert tensor to NumPy
    shap_values = explainer(X_test.numpy())  # SHAP expects NumPy arrays

    # Mean absolute SHAP values
    feature_importance = np.abs(shap_values.values).mean(axis=0).mean(axis=0)
    feature_names = selected_features

    # Sorting feature importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    # Plot SHAP feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[:15][::-1], sorted_importance[:15][::-1], color="blue")
    plt.xlabel("Mean |SHAP Value|")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances (SHAP)")
    plt.show()

    # Save feature importance
    importance_df = pd.DataFrame({"Feature": sorted_features, "Importance": sorted_importance})
    importance_df.to_csv(f"data/feature_importance_{stock_ticker}.csv", index=False)

    logging.info("✅ SHAP Feature Importance saved!")

    logging.info("📊 Running backtest for trained model...")

    net_profit, stock_profit = backtest_model(stock_ticker, start_date, end_date, model, scaler, selected_features)

    return {
        "features": selected_features,
        "net_profit": net_profit,
        "stock_profit": stock_profit,
    }


if __name__ == "__main__":
    stock_ticker = "TQQQ"
    start_date = "2012-01-01"
    end_date = "2023-01-01"
    model_type = "TransformerRNN"

    print(f"🎯 Training {model_type} model for {stock_ticker} from {start_date} to {end_date}...")
    training_result = train_and_evaluate(stock_ticker, start_date, end_date, model_type)
    print(f"✅ Training complete! Results: {training_result}")
