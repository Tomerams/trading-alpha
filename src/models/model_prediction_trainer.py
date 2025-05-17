import os
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import MODEL_PARAMS
from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_indicators_data
from data.data_utilities import get_exclude_from_scaling
from models.model_utilities import get_model, time_based_split, create_sequences


def train_single(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    os.makedirs("files/models", exist_ok=True)
    ticker = request_data.stock_ticker
    model_type = MODEL_PARAMS.get("model_type", "LSTM")
    model_path = f"files/models/{ticker}_{model_type}.pt"
    scaler_path = f"files/models/{ticker}_{model_type}_scaler.pkl"
    feats_path = f"files/models/{ticker}_{model_type}_features.pkl"

    # 1) Load data
    df = get_indicators_data(request_data)
    df["Date"] = pd.to_datetime(df["Date"])
    target_cols = MODEL_PARAMS["target_cols"]
    exclude = get_exclude_from_scaling() | set(target_cols)
    feature_cols = [c for c in df.columns if c not in exclude]

    # 2) Split & sequences
    train_df, val_df, test_df = time_based_split(df)
    seq_len = MODEL_PARAMS.get("seq_len", 10)
    X_tr, y_tr = create_sequences(train_df, feature_cols, target_cols, seq_len)
    X_va, y_va = create_sequences(val_df, feature_cols, target_cols, seq_len)
    X_te, y_te = create_sequences(test_df, feature_cols, target_cols, seq_len)

    # 3) Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_va = scaler.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
    X_te = scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

    # 4) DataLoaders (shuffle only train)
    bs = MODEL_PARAMS.get("batch_size", 32)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float()),
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float()),
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 5) Model, optimizer, scaler, scheduler, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(len(feature_cols), model_type, output_size=len(target_cols)).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=MODEL_PARAMS.get("learning_rate", 1e-3)
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    scaler_amp = GradScaler()
    criterion = nn.MSELoss()

    # 6) Training loop with early stopping
    best_val = float("inf")
    epochs_no_improve = 0
    patience = MODEL_PARAMS.get("early_stopping_patience", 5)
    epochs = MODEL_PARAMS.get("epochs", 50)

    for epoch in range(1, epochs + 1):
        # –– train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # –– validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast():
                    val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch}/{epochs} ─ Train: {train_loss:.4f}, Val: {val_loss:.4f}"
        )

        # early stopping & checkpoint
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                model.load_state_dict(torch.load(model_path))
                break

    # 7) Final save & inference
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, feats_path)
    logging.info(f"Artifacts saved: {model_path}, {scaler_path}, {feats_path}")

    # 8) Test-time predictions + Monte Carlo Dropout confidence
    model.train()  # keep dropout active
    mc_runs = MODEL_PARAMS.get("mc_dropout_runs", 20)
    all_preds = []
    for _ in range(mc_runs):
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
        all_preds.append(np.concatenate(preds))
    all_preds = np.stack(all_preds, axis=0)  # shape (mc_runs, N, T)
    mean_preds = all_preds.mean(axis=0)
    std_preds = all_preds.std(axis=0)  # confidence measure

    # 9) Build result DataFrame
    result = test_df.iloc[seq_len:].reset_index(drop=True)
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for i, name in enumerate(target_cols):
        result[f"Pred_{name}"] = mean_preds[:, i]
        result[f"Std_{name}"] = std_preds[:, i]
        result[f"Actual_{name}"] = y_te[:, i]
    result["Model_Path"] = model_path
    result["Scaler_Path"] = scaler_path

    return result
