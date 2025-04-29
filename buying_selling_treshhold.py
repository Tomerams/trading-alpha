import optuna
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data_processing import get_data
from model_architecture import TransformerRNNModel
from model_backtest import backtest_model
from config import FEATURE_COLUMNS, BACKTEST_PARAMS

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

def objective(trial):
    # 1. הגדרת פרמטרים לחיפוש
    params = {
        "hidden_size":        trial.suggest_int("hidden_size",        64, 512),
        "num_heads":          trial.suggest_int("num_heads",          2,   8),
        "num_layers":         trial.suggest_int("num_layers",         1,   4),
        "dropout":            trial.suggest_float("dropout",           0.0, 0.5),
        "learning_rate":      trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "batch_size":         trial.suggest_categorical("batch_size",   [32, 64, 128]),
        "seq_len":            trial.suggest_int("seq_len",            1,  10),
        "epochs":             20,  # אפשר להעלות/להוריד
        "buying_threshold":   trial.suggest_float("buying_threshold",  0.0,  0.02),
        "selling_threshold":  trial.suggest_float("selling_threshold", -0.02, 0.0),
    }

    # 2. טוען ומעבד נתונים
    df = get_data("QQQ", "2020-01-01", "2024-01-01")
    X = df[FEATURE_COLUMNS].values
    y = df["Target_Tomorrow"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_tr, y_tr = create_sequences(X_train, y_train, params["seq_len"])
    X_te, y_te = create_sequences(X_test,  y_test,  params["seq_len"])

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False)

    # 3. בונה את המודל
    model = TransformerCNN(
        input_size=X.shape[1],
        hidden_size=params["hidden_size"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        output_size=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn   = nn.SmoothL1Loss()

    # 4. לולאת אימון בסיסית
    for _ in range(params["epochs"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss  = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    # 5. backtest עם thresholds וחישוב ביצועים
    perf = backtest_model(
        stock_ticker="TQQQ",
        start_date="2020-01-01",
        end_date="2024-01-01",
        trained_model=model,
        data_scaler=None,  # אם אתה משתמש ב-scaler, העבר אותו כאן
        selected_features=FEATURE_COLUMNS,
        use_leverage=False,
        trade_ticker=None,
        buying_threshold=params["buying_threshold"],
        selling_threshold=params["selling_threshold"],
        verbose=False,
        seq_len=params["seq_len"],
        max_history=None,
        **BACKTEST_PARAMS
    )

    # נרצה **למקסם** Net Profit %, לכן מפיקים שלילי (Optuna בוחרת במינימום)
    return -perf["Net Profit %"]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best parameters found:")
    print(study.best_params)
