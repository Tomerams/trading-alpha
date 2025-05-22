from sklearn import set_config

# disable metadata routing
set_config(enable_metadata_routing=False)

import logging
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import torch

from config.meta_data_config import META_PARAMS
from config.model_trainer_config import TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.meta_dataset import create_meta_ai_dataset
from models.model_utilities import load_model
from routers.routers_entities import UpdateIndicatorsData

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def prepare_meta_dataset(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    model, scaler, feature_cols = load_model(
        request_data.stock_ticker, META_PARAMS["model_type"]
    )
    seq_len = META_PARAMS["seq_len"]

    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    arr = df[feature_cols].values

    if seq_len > 1:
        X3 = np.stack([arr[i : i + seq_len] for i in range(len(arr) - seq_len)])
    else:
        X3 = arr[np.newaxis, :, :].transpose(1, 0, 2)

    flat = X3.reshape(-1, X3.shape[-1])
    scaled = scaler.transform(flat).reshape(X3.shape)

    with torch.no_grad():
        preds_base = model(torch.tensor(scaled, dtype=torch.float32)).cpu().numpy()

    true_vals = df[TRAIN_TARGETS_PARAMS["target_cols"]].iloc[seq_len:].values
    return create_meta_ai_dataset(
        preds_base, true_vals, TRAIN_TARGETS_PARAMS["target_cols"]
    )


def train_meta_model_from_request(request_data: UpdateIndicatorsData) -> dict:
    # 1. Prepare meta dataset
    meta_df = prepare_meta_dataset(request_data)
    X, y = meta_df.drop(columns="Action"), meta_df["Action"]

    # 2. Split train/val
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=META_PARAMS.get("val_size", 0.2),
        stratify=y,
        random_state=META_PARAMS.get("random_state", 42),
    )

    # 3. Initialize and train base learners manually
    base_models = {}
    if META_PARAMS.get("use_lgbm", True):
        log.info("🚀 Training LightGBM base learner")
        # remove early_stopping_rounds from constructor
        lgbm_params = {
            k: v for k, v in META_PARAMS["lgbm"].items() if k != "early_stopping_rounds"
        }
        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(
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
        base_models["lgbm"] = lgbm

    if META_PARAMS.get("use_xgb", False):
        log.info("🚀 Training XGBoost base learner")
        xgbc = xgb.XGBClassifier(**META_PARAMS["xgb"])
        xgbc.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=META_PARAMS["xgb"]["early_stopping_rounds"],
            verbose=True,
        )
        base_models["xgb"] = xgbc

    if META_PARAMS.get("use_rf", True):
        log.info("🚀 Training RandomForest base learner")
        rf = RandomForestClassifier(**META_PARAMS["rf"])
        rf.fit(X_tr, y_tr)
        base_models["rf"] = rf

    if META_PARAMS.get("use_mlp", False):
        log.info("🚀 Training MLPClassifier base learner")
        mlp = MLPClassifier(**META_PARAMS["mlp"])
        mlp.fit(X_tr, y_tr)
        base_models["mlp"] = mlp

    # 4. Build meta features
    train_meta_X = np.column_stack(
        [m.predict_proba(X_tr)[:, 1] for m in base_models.values()]
    )
    val_meta_X = np.column_stack(
        [m.predict_proba(X_va)[:, 1] for m in base_models.values()]
    )

    # 5. Train final meta-learner
    log.info("🚀 Training final LogisticRegression meta-learner")
    meta_clf = LogisticRegression(**META_PARAMS["final_estimator"]).fit(
        train_meta_X, y_tr
    )

    # 6. Evaluate
    preds_va = meta_clf.predict(val_meta_X)
    report = classification_report(y_va, preds_va, digits=4)
    log.info("Validation classification report:\n%s", report)

    # 7. Persist combined model dict
    path = META_PARAMS["meta_model_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"base": base_models, "meta": meta_clf}, path)
    log.info("Saved base+meta models to %s", path)

    return {"model_path": path, "val_report": report}
