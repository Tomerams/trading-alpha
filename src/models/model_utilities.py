import logging
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from models.model_architecture import (
    LSTMModel,
    CNNLSTMModel,
    TCNModel,
    TransformerModel,
    GRUModel,
    TransformerRNNModel,
    TransformerTCNModel,
)
from typing import List, Tuple


def get_model(
    input_size: int,
    model_type: str,
    output_size: int = None,
    hidden_size: int = None,
    num_layers: int = None,
    dropout: float = None,
):
    """
    Initialize and return the selected trading model, merging in
    any overrides (hidden_size, num_layers, dropout) from tuning.
    """
    # 1) determine output size
    if output_size is None:
        output_size = MODEL_TRAINER_PARAMS.get("output_size", 3)

    # 2) merge hyperparam overrides with MODEL_PARAMS defaults
    hs = hidden_size or MODEL_TRAINER_PARAMS.get("hidden_size", 64)
    nl = num_layers or MODEL_TRAINER_PARAMS.get("num_layers", 1)
    dr = dropout or MODEL_TRAINER_PARAMS.get("dropout", 0.0)

    # 3) build the selected model
    model_map = {
        "LSTM": LSTMModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
        ),
        "Transformer": TransformerModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
            num_layers=nl,
            dropout=dr,
        ),
        "TransformerRNN": TransformerRNNModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
            num_layers=nl,
            dropout=dr,
        ),
        "CNNLSTM": CNNLSTMModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
        ),
        "GRU": GRUModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
        ).model,
        "TCN": TCNModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
        ).model,
        "TransformerTCN": TransformerTCNModel(
            input_size=input_size,
            hidden_size=hs,
            output_size=output_size,
        ),
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_map[model_type]

    # 4) for any Keras-like wrappers, force build
    if hasattr(model, "model"):
        model.model.build((None, None, input_size))

    return model


def load_model(
    ticker: str, model_type: str
) -> Tuple[torch.nn.Module, object, List[str]]:
    """
    Load a saved model, scaler, and feature list—with timing & existence checks.
    """
    logging.info("start lad moodel")
    base = f"files/models/{ticker}_{model_type}"
    mp = f"{base}.pt"
    sp = f"{base}_scaler.pkl"
    fp = f"{base}_features.pkl"

    logging.info(
        "   • [load_model] looking for:\n      model=%s\n      scaler=%s\n      feats=%s",
        mp,
        sp,
        fp,
    )

    # 1) Check that all files exist
    for path in (mp, sp, fp):
        if not os.path.exists(path):
            logging.error("   • [load_model] MISSING file: %s", path)
            raise FileNotFoundError(f"Model file not found: {path}")

    # 2) Load the checkpoint
    t0 = time.time()
    checkpoint = torch.load(mp, map_location="cpu")
    dt = time.time() - t0
    logging.info("   • [load_model] torch.load finished in %.2f s", dt)

    # 3) Load scaler & feature list
    t1 = time.time()
    scaler = joblib.load(sp)
    features = joblib.load(fp)
    dt2 = time.time() - t1
    logging.info("   • [load_model] joblib.load scaler+features in %.2f s", dt2)

    # 4) Rebuild model head to match target_cols
    output_size = len(TRAIN_TARGETS_PARAMS.get("target_cols", []))
    model = get_model(
        input_size=len(features), model_type=model_type, output_size=output_size
    )

    # 5) Load weights & eval
    model.load_state_dict(checkpoint)
    model.eval()
    logging.info("   • [load_model] model rebuilt and state_dict loaded")

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
