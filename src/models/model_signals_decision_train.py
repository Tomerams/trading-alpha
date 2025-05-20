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
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# Logging setup
ologging = logging.getLogger()
if not ologging.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("meta_model_training.log"),
            logging.StreamHandler(),
        ],
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
        actual > buy_thr, 2, np.where(actual < sell_thr, 0, 1)
    ).astype(int)

    series = pd.Series(actual).reset_index(drop=True)
    w = peak_window
    df["is_local_max"] = (
        series == series.rolling(window=2 * w + 1, center=True).max()
    ).astype(int)
    df["is_local_min"] = (
        series == series.rolling(window=2 * w + 1, center=True).min()
    ).astype(int)
    peaks, _ = find_peaks(series.values, distance=w, prominence=peak_prominence)
    troughs, _ = find_peaks(-series.values, distance=w, prominence=peak_prominence)
    df["is_peak"] = 0
    df.loc[peaks, "is_peak"] = 1
    df["is_trough"] = 0
    df.loc[troughs, "is_trough"] = 1
    return df


def train_meta_model(meta_df: pd.DataFrame, model_path: str | None = None) -> dict:
    # --- prepare paths & logging ---
    path = model_path or MODEL_PARAMS.get(
        "meta_model_path_pkl", "files/models/meta_action_model.pkl"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.info(f"[meta] stacking ensemble → saving to {path}")

    # --- split into X/y and train/val ---
    X = meta_df.drop(columns="Action")
    y = meta_df["Action"].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=MODEL_PARAMS.get("val_size", 0.2), stratify=y, random_state=42
    )
    logging.info(f"[meta] train shape {X_tr.shape}, val shape {X_val.shape}")

    # --- define base learners ---
    base_learners = [
        ("lgbm", lgb.LGBMClassifier(n_estimators=200, max_depth=5, random_state=42)),
        (
            "xgb",
            xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_estimators=200,
                max_depth=4,
                random_state=42,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=200,
                early_stopping=True,
                random_state=42,
            ),
        ),
    ]

    # --- build the stacking meta-classifier ---
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=500
        ),
        cv=MODEL_PARAMS.get("stack_cv_folds", 5),
        n_jobs=-1,
        passthrough=False,  # if True, original X features are also fed to meta-learner
        verbose=1,
    )

    # --- fit & time it ---
    t0 = time.time()
    stack.fit(X_tr, y_tr)
    logging.info(f"[meta] stacking fit completed in {time.time() - t0:.2f}s")

    # --- evaluate on validation set ---
    preds = stack.predict(X_val)
    report = classification_report(y_val, preds, digits=4)
    logging.info(f"[meta] validation report:\n{report}")

    # --- save the whole pipeline to disk with joblib ---
    joblib.dump(stack, path)
    logging.info(f"[meta] stacked model saved to {path}")

    return {
        "model_path": path,
        "val_report": report,
    }


_cached_meta_model = None
_cached_meta_path = None


def load_meta_model(
    model_path: str = None, use_cache: bool = True
) -> lgb.Booster | None:
    global _cached_meta_model, _cached_meta_path

    path_txt = model_path or MODEL_PARAMS.get(
        "meta_model_path_txt", "files/models/meta_action_model.txt"
    )

    if use_cache and _cached_meta_model is not None and _cached_meta_path == path_txt:
        logging.info(f"    > returning cached meta-model from: {path_txt}")
        return _cached_meta_model

    logging.info(f"    > Attempting to load meta-model from: {path_txt}")
    if not os.path.exists(path_txt):
        logging.error(f"    > Meta-model file NOT FOUND: {path_txt}")
        return None

    t0 = time.time()
    booster_instance = None
    try:
        logging.info(f"    > Reading meta-model file into memory: {path_txt}")
        with open(path_txt, "r") as f:
            model_txt = f.read()
        read_duration = time.time() - t0
        logging.info(
            f"    > File read in {read_duration:.2f}s; length: {len(model_txt)} chars."
        )

        logging.info(f"    > Loading Booster from model_str...")
        t_load_start = time.time()
        # נסה לטעון ישירות מהמחרוזת, בדרך כלל אין צורך בפרמטר nthread כאן.
        # LightGBM יקבע את מספר ה-threads באופן אוטומטי או לפי הגדרות גלובליות.
        booster_instance = lgb.Booster(model_str=model_txt)
        load_duration = time.time() - t_load_start
        logging.info(f"    > Booster loaded from model_str in {load_duration:.2f}s.")

    except FileNotFoundError:
        logging.error(f"    > Meta-model file NOT FOUND (checked again): {path_txt}")
        return None
    except Exception as e:
        # כאן אתה צריך את הלוג המפורט מה-try-except שהוספת, הוא תפס את השגיאה
        logging.error(
            f"    > load_meta_model encountered an unexpected error: {e}", exc_info=True
        )
        return None  # החזר None במקרה של שגיאה

    total_elapsed = time.time() - t0
    if booster_instance:
        logging.info(f"    • load_meta_model succeeded in {total_elapsed:.2f}s")
        if use_cache:
            _cached_meta_model = booster_instance
            _cached_meta_path = path_txt
        return booster_instance
    else:
        # ההודעה שלך מגיעה לכאן "load_meta_model returned in ...s — NOT FOUND"
        # כי booster_instance נשאר None
        logging.info(
            f"    • load_meta_model returned in {total_elapsed:.2f}s — MODEL NOT LOADED (remained None)"
        )
        return None


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
