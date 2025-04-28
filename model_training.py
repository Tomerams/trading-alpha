import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
import optuna
import shap

from config import MODEL_PARAMS
from utilities import get_model, time_based_split, add_noise, create_sequences
from data_processing import get_data
from model_backtest import backtest_model


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def train_single(stock_ticker, start_date, end_date, model_type, params, device):
    df = get_data(stock_ticker, start_date, end_date)
    skip = {"Date", "Close", "Target_Tomorrow", "Target_3_Days", "Target_Next_Week"}
    feature_cols = [c for c in df.columns if c not in skip]
    target_cols = ["Target_Tomorrow", "Target_3_Days", "Target_Next_Week"]

    train_df, val_df, back_df = time_based_split(df)
    for split in (train_df, val_df, back_df):
        split.dropna(subset=feature_cols + target_cols, inplace=True)

    seq_len = params["seq_len"]
    X_train, y_train = create_sequences(train_df, feature_cols, target_cols, seq_len)
    X_val, y_val = create_sequences(val_df, feature_cols, target_cols, seq_len)

    scaler = StandardScaler()
    n_feat = len(feature_cols)
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)

    if params.get("noise_level", 0) > 0:
        X_train = add_noise(X_train, params["noise_level"])

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32).to(device)

    X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)
    X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params.get("num_workers", 4),
    )

    model = get_model(
        input_size=n_feat, model_type=model_type, output_size=len(target_cols)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    scaler_amp = GradScaler()
    loss_fn = nn.MSELoss()

    best_loss, wait = float("inf"), 0
    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            with autocast():
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch+1}/{params['epochs']} Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seq_len": seq_len,
                    "feature_cols": feature_cols,
                    "buy_th": params.get("buying_threshold", 0.0),
                    "sell_th": params.get("selling_threshold", 0.0),
                },
                f"models/trained_model-{stock_ticker}-{model_type}.pt",
            )
        else:
            wait += 1
            if wait >= params.get("early_stop_patience", 10):
                break

    checkpoint = torch.load(
        f"models/trained_model-{stock_ticker}-{model_type}.pt", map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cpu()

    net_profit, _, _, _ = backtest_model(
        stock_ticker=stock_ticker,
        start_date=start_date,
        end_date=end_date,
        trained_model=model,
        data_scaler=scaler,
        selected_features=feature_cols,
        seq_len=checkpoint.get("seq_len"),
        buying_threshold=checkpoint.get("buy_th"),
        selling_threshold=checkpoint.get("sell_th"),
    )
    return model, scaler, net_profit


def objective(trial):
    params = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "weight_decay": trial.suggest_loguniform("wd", 1e-6, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "seq_len": trial.suggest_int("seq_len", 10, 50),
        "noise_level": trial.suggest_uniform("noise", 0.0, 0.05),
        "buying_threshold": trial.suggest_uniform("buy_th", -0.01, 0.01),
        "selling_threshold": trial.suggest_uniform("sell_th", -0.01, 0.01),
        "epochs": MODEL_PARAMS["epochs"],
        "early_stop_patience": 10,
        "num_workers": 4,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, profit = train_single(
        MODEL_PARAMS.get("ticker", "QQQ"),
        MODEL_PARAMS["start_date"],
        MODEL_PARAMS["end_date"],
        MODEL_PARAMS.get("model_type", "TransformerRNN"),
        params,
        device,
    )
    return profit


def run_hyperopt():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    logging.info(f"Best hyperparams: {study.best_trial.params}")
    return study.best_trial.params


def explain_model(model, X_sample: np.ndarray):
    explainer = shap.DeepExplainer(model, torch.tensor(X_sample, dtype=torch.float32))
    shap_vals = explainer.shap_values(torch.tensor(X_sample, dtype=torch.float32))
    shap.summary_plot(shap_vals, X_sample)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=MODEL_PARAMS.get("model_type", "TransformerRNN")
    )
    parser.add_argument("--ticker", default=MODEL_PARAMS.get("ticker", "QQQ"))
    parser.add_argument("--start", default=MODEL_PARAMS.get("start_date", "2020-01-01"))
    parser.add_argument("--end", default=MODEL_PARAMS.get("end_date", "2024-01-01"))
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    if args.optuna:
        run_hyperopt()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = {
            "lr": 0.001,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "seq_len": MODEL_PARAMS.get("seq_len", 20),
            "noise_level": 0.0,
            "buying_threshold": 0.0,
            "selling_threshold": 0.0,
            "epochs": MODEL_PARAMS["epochs"],
            "early_stop_patience": 10,
            "num_workers": 4,
        }
        model, scaler, profit = train_single(
            args.ticker, args.start, args.end, args.model, params, device
        )
        logging.info(f"Backtest net profit: {profit:.2f}%")
        if args.explain:
            df = get_data(args.ticker, args.start, args.end)
            feat_cols = [
                c
                for c in df.columns
                if c
                not in (
                    "Date",
                    "Close",
                    "Target_Tomorrow",
                    "Target_3_Days",
                    "Target_Next_Week",
                )
            ]
            X, _ = create_sequences(
                df,
                feat_cols,
                ["Target_Tomorrow", "Target_3_Days", "Target_Next_Week"],
                params["seq_len"],
            )
            explain_model(model, X[:100])


if __name__ == "__main__":
    main()
