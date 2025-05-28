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
from typing import Any, List, Tuple
import torch.nn as nn

logger_utils = logging.getLogger(__name__)
if not logger_utils.handlers:
    handler_utils = logging.StreamHandler()
    formatter_utils = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler_utils.setFormatter(formatter_utils)
    logger_utils.addHandler(handler_utils)
    logger_utils.setLevel(logging.INFO)


def get_model(
    input_size: int,
    model_type: str,
    output_size: int = None,
    hidden_size: int = None,
    num_layers: int = None,  # For Transformer or LSTM/GRU layers
    dropout: float = None,
    nhead: int = None,  # Specifically for Transformer nhead, אם רלוונטי
) -> nn.Module:  # החזרת טיפוס nn.Module
    """
    Initialize and return the selected trading model.
    """
    start_time_get_model = time.time()
    logger_utils.info(
        f"[get_model] START - input_size={input_size}, model_type='{model_type}', output_size={output_size}, hs={hidden_size}, nl={num_layers}, dr={dropout}, nhead={nhead}"
    )

    effective_output_size = output_size
    if effective_output_size is None:
        effective_output_size = MODEL_TRAINER_PARAMS.get("output_size", 3)  #
        logger_utils.info(
            f"[get_model] output_size was None, using MODEL_TRAINER_PARAMS default: {effective_output_size}"
        )

    effective_hidden_size = hidden_size or MODEL_TRAINER_PARAMS.get(
        "hidden_size", 64
    )  #
    # num_layers מהקונפיגורציה יכול להיות כללי, ויש להתאימו למה שהארכיטקטורה הספציפית מצפה
    effective_num_layers = num_layers or MODEL_TRAINER_PARAMS.get("num_layers", 1)  #
    effective_dropout = (
        dropout if dropout is not None else MODEL_TRAINER_PARAMS.get("dropout", 0.0)
    )  #
    # nhead מהקונפיגורציה, עם ברירת מחדל אם לא קיים (למשל, 4)
    effective_nhead = nhead or MODEL_TRAINER_PARAMS.get("nhead", 4)  #

    logger_utils.info(
        f"[get_model] Effective model params: output_size={effective_output_size}, hidden_size={effective_hidden_size}, num_layers={effective_num_layers}, dropout={effective_dropout}, nhead={effective_nhead}"
    )

    model_instance: nn.Module
    if model_type == "LSTM":
        model_instance = LSTMModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,
            output_size=effective_output_size,
        )  # הוסף num_layers אם LSTMModel תומך
    elif model_type == "Transformer":
        model_instance = TransformerModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,
            output_size=effective_output_size,
            num_layers=effective_num_layers,
            nhead=effective_nhead,
            dropout=effective_dropout,
        )
    elif model_type == "TransformerRNN":
        model_instance = TransformerRNNModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,
            output_size=effective_output_size,
            num_layers=effective_num_layers,
            num_heads=effective_nhead,
            dropout=effective_dropout,
        )
    elif model_type == "CNNLSTM":
        model_instance = CNNLSTMModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,
            output_size=effective_output_size,
        )
    elif model_type == "GRU":
        # ודא ש-GRUModel מקבל את כל הפרמטרים הנכונים
        gru_model_wrapper = GRUModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,
            output_size=effective_output_size,
        )
        model_instance = (
            gru_model_wrapper.model
            if hasattr(gru_model_wrapper, "model")
            else gru_model_wrapper
        )
    elif model_type == "TCN":
        # ודא ש-TCNModel מקבל את כל הפרמטרים הנכונים
        tcn_model_wrapper = TCNModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,
            output_size=effective_output_size,
        )
        model_instance = (
            tcn_model_wrapper.model
            if hasattr(tcn_model_wrapper, "model")
            else tcn_model_wrapper
        )
    elif model_type == "TransformerTCN":
        # TransformerTCNModel מקבל input_size, hidden_size, output_size
        # הפרמטרים הפנימיים n_heads, n_layers ל-Transformer הפנימי קבועים בארכיטקטורה שלו
        # אם רוצים לשנות אותם, יש לעדכן את TransformerTCNModel
        logger_utils.info(
            f"[get_model] Creating TransformerTCNModel with input_size={input_size}, hidden_size={effective_hidden_size}, output_size={effective_output_size}"
        )
        model_instance = TransformerTCNModel(
            input_size=input_size,
            hidden_size=effective_hidden_size,  # משמש ל-tcn_channels ול-transformer_dim
            output_size=effective_output_size,
        )
    else:
        logger_utils.error(f"[get_model] Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

    logger_utils.info(
        f"[get_model] Model '{model_type}' instance created: {type(model_instance)}. Time: {time.time() - start_time_get_model:.2f}s"
    )
    return model_instance


def load_model(
    ticker: str, model_type: str
) -> Tuple[nn.Module, Any, List[str]]:  # החזרת טיפוס nn.Module
    """
    Load a saved model, scaler, and feature list—with timing & existence checks.
    """
    start_time_load_model = time.time()
    logger_utils.info(
        f"[load_model] START - Loading model for ticker='{ticker}', model_type='{model_type}'"
    )
    base_path = f"files/models/{ticker}_{model_type}"
    model_path = f"{base_path}.pt"
    scaler_path = f"{base_path}_scaler.pkl"
    features_path = f"{base_path}_features.pkl"

    logger_utils.info(
        f"[load_model] Expected file paths: model='{model_path}', scaler='{scaler_path}', feats='{features_path}'"
    )

    for path_to_check in (model_path, scaler_path, features_path):
        if not os.path.exists(path_to_check):
            logger_utils.error(
                f"[load_model] CRITICAL ERROR - File not found: {path_to_check}"
            )
            raise FileNotFoundError(f"Model related file not found: {path_to_check}")
    logger_utils.info("[load_model] All required files confirmed to exist.")

    # 1. Load checkpoint from .pt file
    load_checkpoint_start = time.time()
    try:
        checkpoint = torch.load(
            model_path, map_location="cpu"
        )  # Always map to CPU first
        logger_utils.info(
            f"[load_model] torch.load('{model_path}') Succeeded. Time: {time.time() - load_checkpoint_start:.2f}s"
        )
    except Exception as e:
        logger_utils.error(
            f"[load_model] CRITICAL ERROR loading checkpoint from '{model_path}': {e}",
            exc_info=True,
        )
        raise

    # 2. Load scaler and features list from .pkl files
    load_joblibs_start = time.time()
    try:
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        logger_utils.info(
            f"[load_model] joblib.load for scaler & features Succeeded. Time: {time.time() - load_joblibs_start:.2f}s. Number of features: {len(features)}"
        )
    except Exception as e:
        logger_utils.error(
            f"[load_model] CRITICAL ERROR loading scaler/features from '{scaler_path}'/'{features_path}': {e}",
            exc_info=True,
        )
        raise

    # 3. Determine model's output size based on training configuration
    if not TRAIN_TARGETS_PARAMS.get("target_cols"):  #
        logger_utils.error(
            "[load_model] CRITICAL ERROR - TRAIN_TARGETS_PARAMS['target_cols'] is not defined or empty. Cannot determine model output_size."
        )
        raise ValueError(
            "TRAIN_TARGETS_PARAMS['target_cols'] must be defined for model loading."
        )

    effective_output_size = len(TRAIN_TARGETS_PARAMS["target_cols"])  #
    logger_utils.info(
        f"[load_model] Determined model output_size: {effective_output_size} from TRAIN_TARGETS_PARAMS['target_cols']"
    )

    # 4. Rebuild model architecture
    rebuild_model_start = time.time()
    logger_utils.info(
        f"[load_model] Attempting to rebuild model architecture using get_model..."
    )
    try:
        # Pass parameters that might be needed by get_model to reconstruct the exact architecture
        model_instance = get_model(
            input_size=len(features),
            model_type=model_type,
            output_size=effective_output_size,
            hidden_size=MODEL_TRAINER_PARAMS.get("hidden_size"),  #
            num_layers=MODEL_TRAINER_PARAMS.get("num_layers"),  #
            dropout=MODEL_TRAINER_PARAMS.get("dropout"),  #
            nhead=MODEL_TRAINER_PARAMS.get("nhead"),  # Pass nhead if defined in params
        )
        logger_utils.info(
            f"[load_model] Model instance re-created via get_model in {time.time() - rebuild_model_start:.2f}s. Type: {type(model_instance)}"
        )
    except Exception as e:
        logger_utils.error(
            f"[load_model] CRITICAL ERROR during get_model call: {e}", exc_info=True
        )
        raise

    # 5. Load weights into the re-created model structure
    load_weights_start = time.time()
    logger_utils.info(
        f"[load_model] Attempting to load state_dict into model instance..."
    )
    try:
        # Ensure model is on CPU before loading state_dict, as checkpoint was mapped to CPU
        model_instance.to("cpu")

        # Check if checkpoint is a dictionary and contains 'model_state_dict'
        # This is a common pattern when saving more than just the state_dict (e.g., optimizer state, epoch)
        state_dict_to_load = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict_to_load = checkpoint["model_state_dict"]
            logger_utils.info(
                "[load_model] Using 'model_state_dict' key from checkpoint."
            )
        else:
            logger_utils.info(
                "[load_model] Checkpoint does not contain 'model_state_dict' key, assuming checkpoint IS the state_dict."
            )

        model_instance.load_state_dict(state_dict_to_load)
        logger_utils.info(
            f"[load_model] model.load_state_dict() Succeeded. Time: {time.time() - load_weights_start:.2f}s"
        )
    except RuntimeError as e:
        logger_utils.error(
            f"[load_model] CRITICAL RuntimeError during load_state_dict. Possible architecture mismatch or corrupted file: {e}",
            exc_info=True,
        )
        logger_utils.info("--- Keys in checkpoint state_dict_to_load: ---")
        if isinstance(state_dict_to_load, dict):
            for key in state_dict_to_load.keys():
                logger_utils.info(f"  - Checkpoint Key: {key}")
        else:
            logger_utils.info("   State_dict_to_load is not a dictionary.")
        logger_utils.info("--- Keys in current model_instance.state_dict(): ---")
        for key in model_instance.state_dict().keys():
            logger_utils.info(f"  - Model Key: {key}")
        raise
    except Exception as e:
        logger_utils.error(
            f"[load_model] CRITICAL ERROR during load_state_dict: {e}", exc_info=True
        )
        raise

    model_instance.eval()
    logger_utils.info("[load_model] Model set to eval() mode.")
    total_load_time = time.time() - start_time_load_model
    logger_utils.info(
        f"[load_model] END - Model '{ticker}_{model_type}' loaded successfully. Total time: {total_load_time:.2f}s"
    )
    return model_instance, scaler, features


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
