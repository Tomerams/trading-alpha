from sklearn import set_config
set_config(enable_metadata_routing=True)

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
from config.model_trainer_config import TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.meta_dataset import create_meta_ai_dataset
from models.model_utilities import load_model
from routers.routers_entities import UpdateIndicatorsData

log = logging.getLogger(__name__)

def prepare_meta_dataset(request_data: UpdateIndicatorsData):
    model, scaler, feature_cols = load_model(request_data.stock_ticker, META_PARAMS["model_type"])
    seq_len = META_PARAMS["seq_len"]
    df = get_indicators_data(request_data).dropna().reset_index(drop=True)
    arr = df[feature_cols].values
    if seq_len > 1:
        X3 = np.stack([arr[i:i+seq_len] for i in range(len(arr)-seq_len)])
    else:
        X3 = arr[np.newaxis,:,:].transpose(1,0,2)
    flat = X3.reshape(-1, X3.shape[-1])
    scaled = scaler.transform(flat).reshape(X3.shape)
    with torch.no_grad():
        preds = model(torch.tensor(scaled, dtype=torch.float32)).cpu().numpy()
    true_vals = df[TRAIN_TARGETS_PARAMS["target_cols"]].iloc[seq_len:].values
    return create_meta_ai_dataset(preds, true_vals, TRAIN_TARGETS_PARAMS["target_cols"])

def build_base_learners():
    return [
        ("lgbm", lgb.LGBMClassifier(**META_PARAMS["lgbm"])),
        ("xgb", xgb.XGBClassifier(**META_PARAMS["xgb"])),
        ("rf", RandomForestClassifier(**META_PARAMS["rf"])),
        ("mlp", MLPClassifier(**META_PARAMS["mlp"]))
    ]

def build_stacker():
    return StackingClassifier(
        estimators=build_base_learners(),
        final_estimator=LogisticRegression(**META_PARAMS["final_estimator"]),
        cv=META_PARAMS["stack_cv_folds"],
        n_jobs=META_PARAMS["n_jobs"],
        passthrough=META_PARAMS["passthrough"],
        verbose=META_PARAMS["verbose"]
    )

def train_and_persist(meta_df: pd.DataFrame, path: str):
    X, y = meta_df.drop(columns="Action"), meta_df["Action"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=META_PARAMS["val_size"],
        stratify=y,
        random_state=META_PARAMS["random_state"]
    )
    stack = build_stacker()
    fit_params = {
        "lgbm__eval_set": [(X_va, y_va)],
        "lgbm__early_stopping_rounds": META_PARAMS["lgbm"]["early_stopping_rounds"],
        "xgb__eval_set": [(X_va, y_va)],
        "xgb__early_stopping_rounds": META_PARAMS["xgb"]["early_stopping_rounds"]
    }
    t0 = time.time()
    stack.fit(X_tr, y_tr, **fit_params)
    preds = stack.predict(X_va)
    report = classification_report(y_va, preds, digits=4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(stack, path)
    return {"model_path": path, "val_report": report}

def train_meta_model_from_request(request_data: UpdateIndicatorsData):
    path = META_PARAMS["meta_model_path"]
    meta_df = prepare_meta_dataset(request_data)
    return train_and_persist(meta_df, path)
