import logging
import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from optuna.integration import LightGBMTunerCV
from config import MODEL_PARAMS
from models.model_utilities import load_model
from data.data_processing import get_indicators_data
from routers.routers_entities import UpdateIndicatorsData
import torch
from scipy.signal import find_peaks

# Logging setup
ologging = logging.getLogger()
if not ologging.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("meta_model_training.log"), logging.StreamHandler()],
    )


def create_meta_ai_dataset(
    preds: np.ndarray,
    true_vals: np.ndarray,
    target_cols: list[str],
    buy_threshold: float = None,
    sell_threshold: float = None,
    peak_window: int = 5,
    peak_prominence: float = 0.01,
) -> pd.DataFrame:
    buy_thr = buy_threshold or MODEL_PARAMS.get("buy_threshold", 0.002)
    sell_thr = sell_threshold or MODEL_PARAMS.get("sell_threshold", -0.002)
    df = pd.DataFrame(preds, columns=[f"Pred_{c}" for c in target_cols])
    actual = true_vals[:, target_cols.index("Target_Tomorrow")]
    df["Action"] = np.where(
        actual > buy_thr, 2,
        np.where(actual < sell_thr, 0, 1)
    ).astype(int)

    series = pd.Series(actual).reset_index(drop=True)
    w = peak_window
    df["is_local_max"] = (series == series.rolling(window=2*w+1, center=True).max()).astype(int)
    df["is_local_min"] = (series == series.rolling(window=2*w+1, center=True).min()).astype(int)
    peaks, _ = find_peaks(series.values, distance=w, prominence=peak_prominence)
    troughs, _ = find_peaks(-series.values, distance=w, prominence=peak_prominence)
    df["is_peak"] = 0
    df.loc[peaks, "is_peak"] = 1
    df["is_trough"] = 0
    df.loc[troughs, "is_trough"] = 1
    return df


def train_meta_model(
    meta_df: pd.DataFrame,
    model_path: str | None = None
) -> dict:
    path = model_path or MODEL_PARAMS.get(
        "meta_model_path_txt", "files/models/meta_action_model.txt"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    X = meta_df.drop(columns="Action")
    y = meta_df["Action"].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=MODEL_PARAMS.get("val_size", 0.2),
        stratify=y,
        random_state=42
    )

    # Base LightGBM parameters
    base_params = {
        "objective": "multiclass",
        "num_class": len(y.unique()),
        "metric": "multi_logloss",
        "verbosity": -1,
        "force_col_wise": True,
        "n_jobs": 1
    }

    # 1) Create Dataset with pre-filter disabled via params dict
    dtrain = lgb.Dataset(
        X_tr,
        label=y_tr,
        params={"feature_pre_filter": False}
    )

    # 2) Initial CV to find best number of rounds
    cv_res = lgb.cv(
        base_params,
        dtrain,
        num_boost_round=MODEL_PARAMS.get("num_boost_round", 1000),
        nfold=MODEL_PARAMS.get("cv_folds", 5),
        stratified=True,
        seed=42,
        callbacks=[
            lgb.early_stopping(MODEL_PARAMS.get("early_stopping_rounds", 10)),
            lgb.log_evaluation(period=0),
        ],
    )
    best_rounds = len(cv_res[next(iter(cv_res))])

    # 3) Skip hyperparameter tuning to avoid feature_pre_filter issues
    #    Use the base parameters with the best iteration count only
    tuned = {**base_params, "n_estimators": best_rounds, "n_jobs": 1}

    # 4) Train final classifier
    clf = lgb.LGBMClassifier(**tuned)
    clf.fit(X_tr, y_tr)

    # 5) Validation report
    try:
        preds = clf.predict(X_val)
        report = classification_report(y_val, preds, digits=4)
    except Exception:
        report = ""

    # 6) Save model
    try:
        clf.booster_.save_model(path)
    except Exception:
        pass

    return {
        "model_path": path,
        "best_iteration": best_rounds,
        "val_report": report,
    }


def load_meta_model(model_path: str | None = None):
    """
    Load the LightGBM meta-model from a text file, caching the Booster in memory.
    Uses model_str to avoid file I/O hangs, and limits threads to 1.
    """
    path_txt = model_path or MODEL_PARAMS.get(
        "meta_model_path_txt", "files/models/meta_action_model.txt"
    )

    # Module-level cache
    global _cached_meta_model, _cached_meta_path
    try:
        _cached_meta_model
    except NameError:
        _cached_meta_model = None
        _cached_meta_path = None

    # Return cache if same path
    if _cached_meta_model is not None and _cached_meta_path == path_txt:
        logging.info(f"    > using cached meta-model from {path_txt}")
        return _cached_meta_model

    if not os.path.exists(path_txt):
        logging.info(f"    > meta-model file not found at {path_txt}")
        return None

    logging.info(f"    > reading meta-model file into memory to avoid hangs: {path_txt}")
    t0 = time.time()
    try:
        with open(path_txt, 'r') as f:
            model_txt = f.read()
        logging.info(f"    > file read in {time.time() - t0:.2f}s; loading Booster via model_str")
        booster = lgb.Booster(model_str=model_txt, params={"nthread":1})
    except Exception as e:
        logging.error(f"    > failed to load meta-model via model_str: {e}", exc_info=True)
        return None
    elapsed = time.time() - t0
    logging.info(f"    > Booster loaded in {elapsed:.2f}s with nthread=1")

    # Cache and return
    _cached_meta_model = booster
    _cached_meta_path = path_txt
    return booster



def train_meta_model_from_request(request_data: UpdateIndicatorsData) -> dict:
    base_model, scaler, feature_cols = load_model(
        request_data.stock_ticker,
        MODEL_PARAMS["model_type"]
    )
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    seq_len     = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]
    arr = df[feature_cols].values
    if seq_len > 1:
        X = np.stack([arr[i : i + seq_len] for i in range(len(arr) - seq_len)])
    else:
        X = arr[np.newaxis, :, :].transpose(1, 0, 2)
    flat   = X.reshape(-1, X.shape[-1])
    scaled = scaler.transform(flat).reshape(X.shape)
    t_in   = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        preds = base_model(t_in).cpu().numpy()
    true_vals = df[target_cols].iloc[seq_len:].values
    meta_df   = create_meta_ai_dataset(preds, true_vals, target_cols)
    return train_meta_model(meta_df)
