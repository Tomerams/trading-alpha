import logging
import os
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from config.meta_data_config import META_PARAMS
from config.model_trainer_config import TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.meta_dataset import create_meta_ai_dataset
from models.model_utilities import load_model
from routers.routers_entities import UpdateIndicatorsData


def get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

log = get_logger()


def prepare_meta_dataset(request_data: UpdateIndicatorsData):
    model, scaler, features = load_model(
        request_data.stock_ticker,
        META_PARAMS["model_type"]
    )
    seq_len = META_PARAMS["seq_len"]
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    arr = df[features].values
    if len(arr) < seq_len:
        raise ValueError(f"Data length {len(arr)} < seq_len {seq_len}")
    windows = len(arr) - seq_len
    X = np.stack([arr[i : i + seq_len] for i in range(windows)])
    flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(flat).reshape(X.shape)
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).cpu().numpy()
    targets = df[TRAIN_TARGETS_PARAMS["target_cols"]].iloc[seq_len:].values
    preds = preds[: len(targets)]
    return create_meta_ai_dataset(preds, targets, TRAIN_TARGETS_PARAMS["target_cols"])


def train_base_learners(X_tr, y_tr, X_va, y_va):
    base = {}
    if META_PARAMS.get("use_lgbm", True):
        params = {k: v for k, v in META_PARAMS["lgbm"].items() if k != "early_stopping_rounds"}
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                early_stopping(
                    stopping_rounds=META_PARAMS["lgbm"]["early_stopping_rounds"]
                ),
                log_evaluation(period=50),
            ],
        )
        base["lgbm"] = clf
    if META_PARAMS.get("use_xgb", False):
        params = META_PARAMS["xgb"].copy()
        estop = params.pop("early_stopping_rounds", None)
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=estop,
            verbose=True,
        )
        base["xgb"] = clf
    if META_PARAMS.get("use_rf", False):
        clf = RandomForestClassifier(**META_PARAMS["rf"])
        clf.fit(X_tr, y_tr)
        base["rf"] = clf
    if META_PARAMS.get("use_mlp", False):
        clf = MLPClassifier(**META_PARAMS["mlp"])
        clf.fit(X_tr, y_tr)
        base["mlp"] = clf
    return base


def build_meta_features(base, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in base.values()])


def train_meta_model_from_request(request_data: UpdateIndicatorsData) -> dict:
    meta_df = prepare_meta_dataset(request_data)
    feat_cols = [c for c in meta_df.columns if c.startswith("Pred_")]
    X, y = meta_df[feat_cols], meta_df["Action"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=META_PARAMS.get("val_size", 0.2),
        stratify=y,
        random_state=META_PARAMS.get("random_state", 42),
    )
    base = train_base_learners(X_tr, y_tr, X_va, y_va)
    train_meta_X = build_meta_features(base, X_tr)
    val_meta_X = build_meta_features(base, X_va)
    meta_clf = LogisticRegression(**META_PARAMS["final_estimator"]).fit(train_meta_X, y_tr)
    report = classification_report(y_va, meta_clf.predict(val_meta_X), digits=4)
    path = META_PARAMS["meta_model_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"base": base, "meta": meta_clf}, path)
    return {"model_path": path, "val_report": report}
