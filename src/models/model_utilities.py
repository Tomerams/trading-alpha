"""
models/model_utilities.py
─────────────────────────
• get_model  – returns an untrained nn.Module by name
• load_model – loads checkpoint, scaler, and feature list for inference/back-test
"""

from __future__ import annotations

import logging, time, joblib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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

logger_utils = logging.getLogger(__name__)
if not logger_utils.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger_utils.addHandler(h)
    logger_utils.setLevel(logging.INFO)


def get_model(
    input_size: int,
    model_type: str,
    output_size: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    nhead: int | None = None,
) -> nn.Module:
    t0 = time.time()
    logger_utils.info(
        "[get_model] START  input=%d  type=%s  out=%s  hs=%s  nl=%s  dr=%s  nhead=%s",
        input_size,
        model_type,
        output_size,
        hidden_size,
        num_layers,
        dropout,
        nhead,
    )

    if output_size is None:
        output_size = len(TRAIN_TARGETS_PARAMS["target_cols"])
    hidden_size = hidden_size or MODEL_TRAINER_PARAMS.get("hidden_size", 64)
    num_layers = num_layers or MODEL_TRAINER_PARAMS.get("num_layers", 1)
    dropout = (
        dropout if dropout is not None else MODEL_TRAINER_PARAMS.get("dropout", 0.0)
    )
    nhead = nhead or MODEL_TRAINER_PARAMS.get("nhead", 4)

    if model_type == "LSTM":
        model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout)
    elif model_type == "GRU":
        model = GRUModel(input_size, hidden_size, output_size, num_layers, dropout)
    elif model_type == "CNNLSTM":
        model = CNNLSTMModel(input_size, hidden_size, output_size)
    elif model_type == "TCN":
        model = TCNModel(input_size, hidden_size, output_size)
    elif model_type == "Transformer":
        model = TransformerModel(
            input_size,
            hidden_size,
            output_size,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )
    elif model_type == "TransformerRNN":
        model = TransformerRNNModel(
            input_size,
            hidden_size,
            output_size,
            num_layers=num_layers,
            num_heads=nhead,
            dropout=dropout,
        )
    elif model_type == "TransformerTCN":
        model = TransformerTCNModel(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger_utils.info(
        "[get_model] Model %s built in %.2fs", model_type, time.time() - t0
    )
    return model


def load_model(
    ticker: str,
    model_type: str,
    target_override: str | None = None,
) -> Tuple[nn.Module, StandardScaler, List[str]]:
    start_time_load_model = time.time()

    key = target_override or model_type
    model_path = Path(f"files/models/{ticker}_{key}.pt")
    scaler_path = Path(f"files/models/{ticker}_{key}_scaler.pkl")
    feats_path = Path(f"files/models/{ticker}_{key}_features.pkl")

    logger_utils.info(
        "[load_model] Expected files: %s | %s | %s", model_path, scaler_path, feats_path
    )

    for p in (model_path, scaler_path, feats_path):
        if not p.exists():
            raise FileNotFoundError(p)

    checkpoint = torch.load(model_path, map_location="cpu")
    scaler = joblib.load(scaler_path)
    features = joblib.load(feats_path)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    out_dim = state_dict["net.head.weight"].shape[0]

    model = get_model(len(features), model_type, out_dim)
    model.load_state_dict(state_dict)
    model.eval()

    logger_utils.info(
        "[load_model] Loaded %s in %.2fs",
        model_path.name,
        time.time() - start_time_load_model,
    )
    return model, scaler, features


def time_based_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    train_df = df.iloc[: int(n * 0.70)]
    val_df = df.iloc[int(n * 0.70) : int(n * 0.85)]
    test_df = df.iloc[int(n * 0.85) :]
    return train_df, val_df, test_df


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(df) - seq_len):
        xs.append(df[feature_cols].iloc[i : i + seq_len].values)
        ys.append(df[target_cols].iloc[i + seq_len].values)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
