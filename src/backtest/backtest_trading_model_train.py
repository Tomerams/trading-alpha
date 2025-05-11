import os
import logging
import numpy as np
import pandas as pd
import joblib
import torch
from lightgbm import LGBMClassifier
from config import MODEL_PARAMS
from models.model_utilities import load_model
from data.data_processing import get_indicators_data

logging.basicConfig(level=logging.INFO)


def create_meta_ai_dataset(
    preds: np.ndarray,
    true_vals: np.ndarray,
    prices: np.ndarray,
    target_cols: list,
    buy_threshold: float = None,
    sell_threshold: float = None,
) -> pd.DataFrame:
    buy_thr = (
        buy_threshold
        if buy_threshold is not None
        else MODEL_PARAMS.get("buy_threshold", 0.002)
    )
    sell_thr = (
        sell_threshold
        if sell_threshold is not None
        else MODEL_PARAMS.get("sell_threshold", -0.002)
    )

    df = pd.DataFrame(preds, columns=[f"Pred_{col}" for col in target_cols])
    actual_returns = true_vals[:, target_cols.index("Target_Tomorrow")]

    def decide_action(ret: float) -> int:
        if ret > buy_thr:
            return 2  # BUY
        elif ret < sell_thr:
            return 0  # SELL
        else:
            return 1  # HOLD

    df["Action"] = [decide_action(r) for r in actual_returns]
    return df


def train_meta_model(meta_df: pd.DataFrame, model_path: str = None) -> LGBMClassifier:
    path = model_path or MODEL_PARAMS.get(
        "meta_model_path", "files/models/meta_action_model.pkl"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logging.info(f"Training meta-model on {len(meta_df)} samples; saving to {path}")
    X = meta_df.drop(columns=["Action"])
    y = meta_df["Action"]

    params = MODEL_PARAMS.get(
        "meta_model_params", {"n_estimators": 200, "max_depth": 5}
    )
    model = LGBMClassifier(**params)
    model.fit(X, y)

    joblib.dump(model, path)
    logging.info("Meta-model training complete.")
    return model


def load_meta_model(model_path: str = None) -> LGBMClassifier | None:
    path = model_path or MODEL_PARAMS.get(
        "meta_model_path", "files/models/meta_action_model.pkl"
    )
    try:
        model = joblib.load(path)
        logging.info(f"Loaded meta-model from {path}")
        return model
    except FileNotFoundError:
        logging.warning(f"No meta-model found at {path}")
        return None


def train_meta_model_from_request(request_data) -> None:
    model, scaler, feature_cols = load_model(
        request_data.stock_ticker, MODEL_PARAMS["model_type"]
    )
    seq_len = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]

    # Fetch and prepare data
    df = get_indicators_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)

    if seq_len > 1:
        X = np.stack(
            [
                df[feature_cols].iloc[i : i + seq_len].values
                for i in range(len(df) - seq_len)
            ]
        )
        prices = df["Close"].iloc[seq_len:].values
        true_vals = df[target_cols].iloc[seq_len:].values
        n_feat = X.shape[2]
        X_flat = X.reshape(-1, n_feat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled = scaler.transform(df[feature_cols].values)
        prices = df["Close"].values
        true_vals = df[target_cols].values

    # Predict targets
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        preds = model(X_tensor).cpu().numpy()

    # Build meta-dataset and train
    meta_df = create_meta_ai_dataset(preds, true_vals, prices, target_cols)
    train_meta_model(meta_df)
