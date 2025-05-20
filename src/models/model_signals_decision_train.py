import logging
import os
import time
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb
import torch
import xgboost as xgb

from config.meta_data_config import META_PARAMS
from data.data_processing import get_indicators_data
from models.meta_dataset import create_meta_ai_dataset
from models.model_utilities import load_model
from routers.routers_entities import UpdateIndicatorsData


log = logging.getLogger(__name__)


def prepare_meta_dataset(request_data: UpdateIndicatorsData):
    # 1) load the base model + artifacts
    model_type = META_PARAMS["model_type"]
    model, scaler, feature_cols, seq_len = load_model(
        request_data.stock_ticker,
        model_type,
    )

    # 2) fetch the indicator‐enriched DataFrame
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)

    # 3) build the 3D input array for your sequence model
    arr = df[feature_cols].values
    if seq_len > 1:
        X3 = np.stack([arr[i : i + seq_len] for i in range(len(arr) - seq_len)])
    else:
        # the loader will tell you seq_len=1 if it’s a non‐sequential model
        X3 = arr[np.newaxis, :, :].transpose(1, 0, 2)

    # 4) scale + reshape back to the right shape for inference
    flat = X3.reshape(-1, X3.shape[-1])
    scaled = scaler.transform(flat).reshape(X3.shape)

    # 5) run your model
    with torch.no_grad():
        preds = model(torch.tensor(scaled, dtype=torch.float32)).cpu().numpy()

    # 6) get the true future returns matrix
    true_vals = df[META_PARAMS["target_cols"]].iloc[seq_len:].values

    # 7) turn those into your meta‐dataset
    return create_meta_ai_dataset(preds, true_vals, META_PARAMS["target_cols"])


def build_base_learners():
    """Instantiates the base learners from your config."""
    return [
        ("lgbm", lgb.LGBMClassifier(**META_PARAMS["lgbm"])),
        ("xgb", xgb.XGBClassifier(**META_PARAMS["xgb"])),
        ("rf", RandomForestClassifier(**META_PARAMS["rf"])),
        ("mlp", MLPClassifier(**META_PARAMS["mlp"])),
    ]


def build_stacker():
    """Builds the StackingClassifier using base learners + a final estimator."""
    return StackingClassifier(
        estimators=build_base_learners(),
        final_estimator=LogisticRegression(**META_PARAMS["final_estimator"]),
        cv=META_PARAMS["stack_cv_folds"],
        n_jobs=META_PARAMS["n_jobs"],
        passthrough=META_PARAMS["passthrough"],
        verbose=META_PARAMS["verbose"],
    )


def train_and_persist(meta_df: pd.DataFrame, path: str):
    """Runs train/test split, fits the stacker, logs metrics, and saves the pipeline."""
    X, y = meta_df.drop(columns="Action"), meta_df["Action"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=META_PARAMS["val_size"],
        stratify=y,
        random_state=META_PARAMS["random_state"],
    )
    log.info(f"Training meta-model on {X_tr.shape}, validating on {X_va.shape}")

    stack = build_stacker()
    t0 = time.time()
    stack.fit(X_tr, y_tr)
    log.info(f"→ Stack fit completed in {time.time()-t0:.1f}s")

    preds = stack.predict(X_va)
    report = classification_report(y_va, preds, digits=4)
    log.info(f"Validation report:\n{report}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(stack, path)
    log.info(f"Stacked model saved to {path}")
    return {"model_path": path, "val_report": report}


def train_meta_model_from_request(request_data: UpdateIndicatorsData):
    """Orchestrator: prepares data, trains the stacker, returns results."""
    path = "/files/models"
    meta_df = prepare_meta_dataset(request_data)
    return train_and_persist(meta_df, path)
