import optuna
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from config.optimizations_config import OPTUNA_PARAMS
from config.model_trainer_config import MODEL_TRAINER_PARAMS, TRAIN_TARGETS_PARAMS
from data.data_processing import get_indicators_data
from models.model_utilities import get_model, time_based_split, create_sequences
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_loaders(request_data, seq_len, batch_size):
    df = get_indicators_data(request_data)
    train_df, val_df, _ = time_based_split(df)
    feature_cols = [
        c
        for c in df.columns
        if c not in ("Date", "Close", *TRAIN_TARGETS_PARAMS["target_cols"])
    ]
    X_tr, y_tr = create_sequences(
        train_df, feature_cols, TRAIN_TARGETS_PARAMS["target_cols"], seq_len
    )
    X_val, y_val = create_sequences(
        val_df, feature_cols, TRAIN_TARGETS_PARAMS["target_cols"], seq_len
    )

    scaler = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    )

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds, batch_size, shuffle=False),
    )


def objective(trial, request_data):
    # 1) sample hyperparameters
    #   – num_heads chosen from allowable set
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])

    #   – hidden_size as multiple of num_heads
    multiplier = trial.suggest_int("hidden_mult", 1, 8)
    hidden_size = num_heads * multiplier

    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    seq_len = trial.suggest_int("sequence_length", 5, 30)

    # 2) build data loaders
    train_loader, val_loader = _prepare_loaders(
        request_data, seq_len, OPTUNA_PARAMS["batch_size"]
    )

    # 3) instantiate model (pass num_heads into MODEL_PARAMS or directly to get_model)
    model = get_model(
        input_size=train_loader.dataset.tensors[0].shape[-1],
        model_type=MODEL_TRAINER_PARAMS["model_type"],
        output_size=len(TRAIN_TARGETS_PARAMS["target_cols"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout_rate,
        num_heads=num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(OPTUNA_PARAMS["max_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        val_loss = sum(
            loss_fn(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_loader
        ) / len(val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= OPTUNA_PARAMS["early_stopping_patience"]:
                break

    return best_val


def run_optuna(request_data):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=OPTUNA_PARAMS["warmup_steps"]
        ),
    )
    study.optimize(
        lambda t: objective(t, request_data),
        n_trials=OPTUNA_PARAMS["n_trials"],
        timeout=OPTUNA_PARAMS["timeout_seconds"],
    )
    # persist for later inspection
    Path("files/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(study, "files/models/optuna_study.pkl")
    return study
