import os
import joblib
import pandas as pd
import torch
from config import MODEL_PARAMS
from model_architecture import (
    LSTMModel,
    CNNLSTMModel,
    TCNModel,
    TransformerModel,
    GRUModel,
    TransformerRNNModel,
)


def get_model(input_size, model_type, output_size=1):
    """Initialize and return the selected trading model with fixed input shape."""
    hidden_size = MODEL_PARAMS["hidden_size"]
    output_size = MODEL_PARAMS["output_size"]

    model_map = {
        "LSTM": LSTMModel(input_size, hidden_size, output_size),
        "Transformer": TransformerModel(input_size, hidden_size, output_size),
        "TransformerRNN": TransformerRNNModel(input_size, hidden_size, output_size),
        "CNNLSTM": CNNLSTMModel(input_size, hidden_size, output_size),
        "GRU": GRUModel(input_size, hidden_size, output_size).model,
        "TCN": TCNModel(input_size, hidden_size, output_size).model,
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_map[model_type]

    # Force explicit model building for Keras models
    if isinstance(model, TCNModel) or isinstance(model, GRUModel):
        model.model.build((None, input_size, 1))

    return model


def load_model(ticker, model_type):
    model_filename = f"models/trained_model-{ticker}-{model_type}.pt"  
    scaler_filename = f"models/scaler-{ticker}-{model_type}.pkl"
    features_filename = f"models/features-{ticker}-{model_type}.pkl"

    if (
        not os.path.exists(model_filename)
        or not os.path.exists(scaler_filename)
        or not os.path.exists(features_filename)
    ):
        raise FileNotFoundError(f"❌ Model files for {ticker}-{model_type} not found! Train the model first.")

    checkpoint = torch.load(model_filename, map_location=torch.device("cpu"))

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"❌ Invalid model file: {model_filename}. The key 'model_state_dict' is missing!")

    features = joblib.load(features_filename)
    scaler = joblib.load(scaler_filename)

    model = get_model(input_size=len(features), model_type=model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler, features


def time_based_split(df: pd.DataFrame):

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    total_rows = len(df)
    train_size = int(total_rows * 0.7)
    test_size = int(total_rows * 0.15)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:train_size + test_size]
    backtest_df = df.iloc[train_size + test_size:]
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}, Backtest size: {len(backtest_df)}")
    
    return train_df, test_df, backtest_df