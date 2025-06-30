TRAIN_TARGETS_PARAMS = {
    "target_type": "log",
    "target_cols": [
        "Target_Tomorrow",
        "Target_2_Days",
        "Target_3_Days",
        "Target_5_Days",
        "Target_8_Days",
        "Target_13_Days",
        "Target_21_Days",
        "NextLocalMaxPct",
        "NextLocalMinPct",
        "BarsToNextLocalMax",
        "BarsToNextLocalMin",
    ],
    "shift_targets": [
        {"name": "Tomorrow", "shift": -1},
        {"name": "2_Days", "shift": -2},
        {"name": "3_Days", "shift": -3},
        {"name": "5_Days", "shift": -5},
        {"name": "8_Days", "shift": -8},
        {"name": "13_Days", "shift": -13},
        {"name": "21_Days", "shift": -21},
    ],
    "extrema_window": 10,
}


MODEL_TRAINER_PARAMS = {
    # ───────────── בסיס ─────────────
    "model_type": "TransformerTCN",
    "epochs": 25,  # 25–30 עם Early-Stopping
    "batch_size": 256,
    "seq_len": 60,
    # ─── היפר-פרמטרים שה-Optuna מצא ───
    "model_kwargs": {  # <-- מועבר אל get_model(...)
        "hidden_size": 192,
        "num_layers": 4,
        "dropout": 0.10,
        "lr": 0.0022173,
        # אם למודל יש num_heads:
        # "num_heads":     4,
    },
    # ─────────── רגולריזציה / לימוד ───────────
    "weight_decay": 1e-4,
    "lr_patience": 3,  # Reduce-LR אם val לא משתפר 3 epochs
    "lr_factor": 0.5,
    "min_lr": 1e-6,
    "early_stopping_patience": 6,  # עצור אחרי 6 epochs בלי שיפור
    # ─────────── חלוקה ל־Train/Val ───────────
    "val_ratio": 0.2,  # 20 % אחרונים כ-validation
    # ─────────── Grid-Trading Parameters ───────────
    "grid_buying_threshold": [0.00, 0.01, 0.02, 0.05],
    "grid_selling_threshold": [0.00, 0.005, 0.01, 0.02],
    "grid_profit_target": [0.02, 0.05, 0.10],
    "grid_trailing_stop": [0.02, 0.03, 0.05],
    # ─────────── סוג Target (לוגריתמי) ───────────
    "target_type": "log",
}
