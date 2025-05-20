OPTUNA_PARAMS = {
    "n_trials": 50,
    "timeout_seconds": 3600,
    "max_epochs": 100,
    "batch_size": 64,
    "early_stopping_patience": 5,
}

BACKTEST_OPTIMIZATIONS_PARAMS = {
    "grid_buying_threshold": [0.0, 0.01, 0.02, 0.05],
    "grid_selling_threshold": [0.0, 0.005, 0.01, 0.02],
    "grid_profit_target": [0.02, 0.05, 0.1],
    "grid_trailing_stop": [0.02, 0.03, 0.05],
}
