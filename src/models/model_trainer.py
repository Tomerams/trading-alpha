import os
import logging
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import MODEL_PARAMS
from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_indicators_data
from data.data_utilities import get_exclude_from_scaling
from models.model_utilities import get_model, time_based_split, create_sequences


def train_single(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    """
    Train a model on data specified by request_data, save model and scaler,
    and return predictions as a DataFrame.
    """
    # Prepare directories
    os.makedirs("files/models", exist_ok=True)

    # Paths for artifacts
    model_type = MODEL_PARAMS.get("model_type", "LSTM")
    model_path = f"files/models/{request_data.stock_ticker}_{model_type}.pt"
    scaler_path = f"files/models/{request_data.stock_ticker}_{model_type}_scaler.pkl"
    features_path = (
        f"files/models/{request_data.stock_ticker}_{model_type}_features.pkl"
    )

    # 1) Load enriched data
    df = get_indicators_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])

    # 2) Define targets and features
    target_cols = MODEL_PARAMS.get("target_cols", [])
    exclude_cols = get_exclude_from_scaling()
    exclude_cols.update(target_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 3) Time-based split
    train_df, val_df, test_df = time_based_split(df)

    # 4) Sequence generation
    seq_len = MODEL_PARAMS.get("seq_len", 10)
    X_train, y_train = create_sequences(train_df, feature_cols, target_cols, seq_len)
    X_val, y_val = create_sequences(val_df, feature_cols, target_cols, seq_len)
    X_test, y_test = create_sequences(test_df, feature_cols, target_cols, seq_len)

    # 5) Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
        X_train.shape
    )
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    # 6) DataLoaders
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

    # 7) Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[-1]
    model = get_model(input_size, model_type, output_size=len(target_cols)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=MODEL_PARAMS.get("learning_rate", 1e-3)
    )
    criterion = nn.MSELoss()

    # 8) Training loop with validation and early stopping
    epochs = MODEL_PARAMS.get("epochs", 20)
    patience = MODEL_PARAMS.get("early_stopping_patience", None)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        logging.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(torch.load(model_path))
                    break

    # 9) Save final artifacts
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    logging.info(
        f"Saved model to {model_path}, scaler to {scaler_path}, features to {features_path}"
    )

    # 10) Inference
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    predictions = np.concatenate(preds)

    # 11) Prepare output DataFrame
    result = test_df.iloc[seq_len:].copy().reset_index(drop=True)
    # Convert 'Date' to ISO-formatted string for JSON serialization
    result["Date"] = pd.to_datetime(result["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    result = result[["Date"]]
    for idx, name in enumerate(target_cols):
        result[f"Predicted_{name}"] = predictions[:, idx]
        result[f"Actual_{name}"] = y_test[:, idx]
    result["Model_Path"] = model_path
    result["Scaler_Path"] = scaler_path

    return result
