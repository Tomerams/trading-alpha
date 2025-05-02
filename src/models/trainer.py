import os
import logging
from datetime import datetime
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import MODEL_PARAMS
from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_data
from models.model_utilities import get_model, time_based_split, create_sequences


def train_single(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    """
    Train a model on data specified by request_data, save model and scaler, and return predictions.
    """
    # Prepare directories
    os.makedirs("files/models", exist_ok=True)

    # 1. Load data
    df = get_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])

    # 2. Generate targets
    df["Target_Tomorrow"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["Target_3_Days"] = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
    df["Target_Next_Week"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Determine columns
    target_cols = ["Target_Tomorrow", "Target_3_Days", "Target_Next_Week"]
    exclude_cols = ["Date", "Close"] + target_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 4. Split time-based
    train_df, val_df, test_df = time_based_split(df)

    # 5. Create sequences
    seq_len = MODEL_PARAMS.get("seq_len", 10)
    X_train, y_train = create_sequences(train_df, feature_cols, target_cols, seq_len)
    X_val, y_val = create_sequences(val_df, feature_cols, target_cols, seq_len)
    X_test, y_test = create_sequences(test_df, feature_cols, target_cols, seq_len)

    # 6. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
        X_train.shape
    )
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    # 7. DataLoaders
    batch_size = MODEL_PARAMS.get("batch_size", 32)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # 8. Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[-1]
    model_type = MODEL_PARAMS.get("model_type", "LSTM")
    model = get_model(input_size, model_type, output_size=len(target_cols)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_PARAMS["learning_rate"])
    criterion = nn.MSELoss()

    # 9. Train loop
    epochs = MODEL_PARAMS["epochs"]
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(
            f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}"
        )

    # 10. Save artifacts
    model_path = f"files/models/{request_data.stock_ticker}_{model_type}.pt"
    scaler_path = (
        f"files/models/{request_data.stock_ticker}_{model_type}_scaler.pkl"
    )
    features_path = f"files/models/{request_data.stock_ticker}_{model_type}_features.pkl"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    logging.info(
        f"Saved model to {model_path}, scaler to {scaler_path}, features to {features_path}"
    )

    # 11. Inference
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
    predictions = np.concatenate(all_preds)

    # 12. Prepare output
    result = test_df.iloc[seq_len:].copy().reset_index(drop=True)
    result = result[["Date"]]
    # Convert Timestamp to ISO-formatted string for JSON serializable
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    # Predictions and actuals for each horizon
    result["Predicted_Tomorrow"] = predictions[:, 0]
    result["Predicted_3_Days"] = predictions[:, 1]
    result["Predicted_Next_Week"] = predictions[:, 2]
    result["Actual_Tomorrow"] = y_test[:, 0]
    result["Actual_3_Days"] = y_test[:, 1]
    result["Actual_Next_Week"] = y_test[:, 2]
    result["Model_Path"] = model_path
    result["Scaler_Path"] = scaler_path

    return result
