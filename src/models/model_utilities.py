import os
import joblib
import numpy as np
import pandas as pd
import torch
from config import MODEL_PARAMS
from models.architecture import (
    LSTMModel,
    CNNLSTMModel,
    TCNModel,
    TransformerModel,
    GRUModel,
    TransformerRNNModel,
    TransformerTCN,
    TransformerTCNModel,
)
from typing import List, Tuple


def get_model(input_size: int, model_type: str, output_size: int = None):
    """Initialize and return the selected trading model with fixed input shape."""
    hidden_size = MODEL_PARAMS["hidden_size"]
    if output_size is None:
        output_size = MODEL_PARAMS["output_size"]

    model_map = {
        "LSTM": LSTMModel(input_size, hidden_size, output_size),
        "Transformer": TransformerModel(input_size, hidden_size, output_size),
        "TransformerRNN": TransformerRNNModel(input_size, hidden_size, output_size),
        "CNNLSTM": CNNLSTMModel(input_size, hidden_size, output_size),
        "GRU": GRUModel(input_size, hidden_size, output_size).model,
        "TCN": TCNModel(input_size, hidden_size, output_size).model,
        "TransformerTCN": TransformerTCNModel(input_size, hidden_size, output_size),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_map[model_type]

    # Force explicit model building for Keras-like models
    if hasattr(model, "model"):
        model.model.build((None, None, input_size))

    return model


def load_model(
    ticker: str, model_type: str
) -> Tuple[torch.nn.Module, object, List[str]]:
    """Load a saved model, scaler, and feature list for backtesting."""
    model_filename = f"files/models/{ticker}_{model_type}.pt"
    scaler_filename = f"files/models/{ticker}_{model_type}_scaler.pkl"
    features_filename = f"files/models/{ticker}_{model_type}_features.pkl"

    if not (
        os.path.exists(model_filename)
        and os.path.exists(scaler_filename)
        and os.path.exists(features_filename)
    ):
        raise FileNotFoundError(
            f"Model files for {ticker}-{model_type}  vbnot found. Train the model first."
        )

    checkpoint = torch.load(model_filename, map_location=torch.device("cpu"))
    scaler = joblib.load(scaler_filename)
    features = joblib.load(features_filename)

    model = get_model(input_size=len(features), model_type=model_type)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, scaler, features


def time_based_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train, validation, and test based on time."""
    df = df.sort_values(by="Date").reset_index(drop=True)
    total = len(df)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def create_sequences(
    df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling sequences of features and targets for model input."""
    data_X = df[feature_cols].values
    data_y = df[target_cols].values

    xs, ys = [], []
    for i in range(len(data_X) - seq_len):
        xs.append(data_X[i : i + seq_len])
        ys.append(data_y[i + seq_len])

    return np.array(xs), np.array(ys)
