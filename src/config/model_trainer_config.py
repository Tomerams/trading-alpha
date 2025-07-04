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
    "extrema_window": 10
}

MODEL_TRAINER_PARAMS = {
    "model_type": "TransformerTCN",
    "epochs": 60,
    "batch_size": 256,
    "seq_len": 60,
    "seq_len_map": {1: 40, 2: 60, 3: 60, 5: 80, 8: 120, 13: 120, 21: 120},
    "model_kwargs": {
        "hidden_size": 384,
        "num_layers": 6,
        "dropout": 0.15
    },
    "weight_decay": 1e-4,
    "warmup_pct": 0.20,
    "early_stopping_patience": 8,
    "val_ratio": 0.2
}