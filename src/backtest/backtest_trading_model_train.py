import os
import logging
import time
import numpy as np
import pandas as pd
import joblib
import torch
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from config import MODEL_PARAMS
from models.model_utilities import load_model
from data.data_processing import get_indicators_data
from routers.routers_entities import UpdateIndicatorsData

logging.basicConfig(level=logging.INFO)


def create_meta_ai_dataset(
    preds: np.ndarray,
    true_vals: np.ndarray,
    prices: np.ndarray,
    target_cols: list,
    buy_threshold: float = None,
    sell_threshold: float = None,
) -> pd.DataFrame:
    buy_thr = buy_threshold if buy_threshold is not None else MODEL_PARAMS.get("buy_threshold", 0.002)
    sell_thr = sell_threshold if sell_threshold is not None else MODEL_PARAMS.get("sell_threshold", -0.002)

    df = pd.DataFrame(preds, columns=[f"Pred_{c}" for c in target_cols])
    actual_returns = true_vals[:, target_cols.index("Target_Tomorrow")]

    def decide_action(ret: float) -> int:
        if ret > buy_thr:    return 2  # BUY
        if ret < sell_thr:   return 0  # SELL
        return 1  # HOLD

    df["Action"] = [decide_action(r) for r in actual_returns]
    return df


def train_meta_model(meta_df: pd.DataFrame, model_path: str = None) -> LGBMClassifier:
    # 1) Prepare path
    path = model_path or MODEL_PARAMS.get("meta_model_path", "files/models/meta_action_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2) Extract X,y and log imbalance
    X = meta_df.drop(columns=["Action"])
    y = meta_df["Action"]
    logging.info(f"Meta‐dataset size: {len(y)}; class distribution:\n{y.value_counts()}")

    # 3) Quick stratified 5-fold CV on F1-macro (fresh model each fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average="macro")
    cv_scores = []
    early = MODEL_PARAMS.get("early_stopping_rounds", 10)

    for train_idx, test_idx in cv.split(X, y):
        est = LGBMClassifier(
            random_state=42,
            force_col_wise=True,
            verbosity=1,            # only errors
            min_gain_to_split=0.001,
            min_child_samples=5,
        )
        est.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=early)]
        )
        preds = est.predict(X.iloc[test_idx])
        cv_scores.append(f1_score(y.iloc[test_idx], preds, average="macro"))

    logging.info("5-fold CV F1-macro: %.4f ± %.4f", np.mean(cv_scores), np.std(cv_scores))

    # 4) Split out a hold-out set for final validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=MODEL_PARAMS.get("val_size", 0.2),
        stratify=y,
        random_state=42,
    )
    logging.info("→ Final Train: %d, Val: %d", len(y_tr), len(y_val))

    # 5) Compute sample weights
    sample_weights = compute_sample_weight("balanced", y_tr)

    # 6) Hyperparameter search
    param_dist = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [5, 10, 20],
    }
    search = RandomizedSearchCV(
        LGBMClassifier(random_state=42, force_col_wise=True, verbosity=1),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring=scorer,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    logging.info("Starting hyperparameter search…")
    search.fit(X_tr, y_tr, sample_weight=sample_weights)
    best_params = search.best_params_
    logging.info("  • Best params: %s", best_params)

    # 7) Final fit on best params
    model = LGBMClassifier(**best_params, random_state=42,
                           force_col_wise=True, verbosity=1)
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=early)]
    )
    logging.info("  • Final fit completed in %.2fs", time.time() - t0)

    # 8) Final validation report
    preds_val = model.predict(X_val)
    report = classification_report(y_val, preds_val, digits=4)
    logging.info("Final validation report:\n%s", report)

    # 9) Persist without compression
    joblib.dump(model, path, compress=0)
    logging.info("Meta‐model saved to %s (uncompressed)", path)

    return model


def load_meta_model(model_path: str = None) -> LGBMClassifier | None:
    path = model_path or MODEL_PARAMS.get("meta_model_path", "files/models/meta_action_model.pkl")
    logging.info(" • [load_meta_model] path = %s", path)

    if not os.path.exists(path):
        logging.warning(" • [load_meta_model] file NOT FOUND")
        return None

    t0 = time.time()
    model = joblib.load(path)
    logging.info(" • [load_meta_model] loaded in %.2fs", time.time() - t0)
    return model


def train_meta_model_from_request(request_data: UpdateIndicatorsData) -> None:
    logging.info("▶️  train_meta_model_from_request start for %s", request_data.stock_ticker)

    base_model, scaler, feature_cols = load_model(
        request_data.stock_ticker, MODEL_PARAMS["model_type"]
    )
    seq_len     = MODEL_PARAMS["seq_len"]
    target_cols = MODEL_PARAMS["target_cols"]

    df = get_indicators_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(inplace=True)

    if seq_len > 1:
        X = np.stack([
            df[feature_cols].iloc[i : i + seq_len].values
            for i in range(len(df) - seq_len)
        ])
        prices    = df["Close"].iloc[seq_len:].values
        true_vals = df[target_cols].iloc[seq_len:].values
        n_feat    = X.shape[2]
        X_flat    = X.reshape(-1, n_feat)
        X_scaled  = scaler.transform(X_flat).reshape(X.shape)
    else:
        X_scaled  = scaler.transform(df[feature_cols].values)
        prices    = df["Close"].values
        true_vals = df[target_cols].values

    base_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if seq_len == 1 and X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1)
        preds = base_model(X_tensor).cpu().numpy()

    meta_df = create_meta_ai_dataset(preds, true_vals, prices, target_cols)
    train_meta_model(meta_df)

    logging.info("   ✅  train_meta_model_from_request complete")