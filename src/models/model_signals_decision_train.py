import logging
import os
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
    df = pd.DataFrame(preds, columns=[f"Pred_{c}" for c in target_cols])
    actual = true_vals[:, target_cols.index("Target_Tomorrow")]
    df["Action"] = np.where(
        actual > buy_thr, 2, np.where(actual < sell_thr, 0, 1)
    ).astype(int)
    series = pd.Series(actual).reset_index(drop=True)
    window = peak_window
    df["rolling_max"] = series.rolling(window=2 * window + 1, center=True).max()
    df["is_local_max"] = (series == df["rolling_max"]).astype(int)
    df["rolling_min"] = series.rolling(window=2 * window + 1, center=True).min()
    df["is_local_min"] = (series == df["rolling_min"]).astype(int)
    vals = series.values
    peaks, _ = find_peaks(vals, distance=window, prominence=peak_prominence)
    troughs, _ = find_peaks(-vals, distance=window, prominence=peak_prominence)
    df["is_peak"] = 0
    df.loc[peaks, "is_peak"] = 1
    df["is_trough"] = 0
    df.loc[troughs, "is_trough"] = 1
    df = df.drop(columns=["rolling_max", "rolling_min"])
    return df


def train_meta_model(meta_df: pd.DataFrame, model_path: str | None = None) -> dict:
    path = model_path or MODEL_PARAMS.get(
        "meta_model_path_txt", "files/models/meta_action_model.txt"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X = meta_df.drop(columns="Action")
    y = meta_df["Action"].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=MODEL_PARAMS.get("val_size", 0.2), stratify=y, random_state=42
    )
    base_params = {
        "objective": "multiclass",
        "num_class": len(y.unique()),
        "metric": "multi_logloss",
        "verbosity": -1,
        "force_col_wise": True,
        "n_jobs": 1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
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
    try:
        tuner = LightGBMTunerCV(
            base_params,
            dtrain,
            num_boost_round=best_rounds,
            nfold=MODEL_PARAMS.get("tuner_cv_folds", 3),
            seed=42,
            callbacks=[
                lgb.early_stopping(MODEL_PARAMS.get("early_stopping_rounds", 10)),
                lgb.log_evaluation(period=0),
            ],
        )
        tuner.run()
        tuned = tuner.best_params
        tuned.update({"n_estimators": best_rounds, "n_jobs": 1})
    except Exception:
        tuned = {**base_params, "n_estimators": best_rounds, "n_jobs": 1}
    clf = lgb.LGBMClassifier(**tuned)
    clf.fit(X_tr, y_tr)
    report = ""
    try:
        preds = clf.predict(X_val)
        report = classification_report(y_val, preds, digits=4)
    except Exception:
        report = ""
    try:
        clf.booster_.save_model(path)
    except Exception:
        pass
    return {"model_path": path, "best_iteration": best_rounds, "val_report": report}


def load_meta_model(model_path: str | None = None) -> lgb.Booster | None:
    path = model_path or MODEL_PARAMS.get(
        "meta_model_path_txt", "files/models/meta_action_model.txt"
    )
    if not os.path.exists(path):
        return None
    return lgb.Booster(model_file=path)


def train_meta_model_from_request(request_data: UpdateIndicatorsData) -> dict:
    base_model, scaler, feature_cols = load_model(
        request_data.stock_ticker, MODEL_PARAMS["model_type"]
    )
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    seq_len = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]
    arr = df[feature_cols].values
    if seq_len > 1:
        X = np.stack([arr[i : i + seq_len] for i in range(len(arr) - seq_len)])
    else:
        X = arr[np.newaxis, :, :].transpose(1, 0, 2)
    flat = X.reshape(-1, X.shape[-1])
    scaled = scaler.transform(flat).reshape(X.shape)
    t_in = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        preds = base_model(t_in).cpu().numpy()
    true_vals = df[target_cols].iloc[seq_len:].values
    meta_df = create_meta_ai_dataset(preds, true_vals, target_cols)
    return train_meta_model(meta_df)
