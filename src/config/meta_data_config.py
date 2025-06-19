import os

THIS_DIR = os.path.dirname(__file__)
PROJECT_SRC = os.path.abspath(os.path.join(THIS_DIR, os.pardir))


META_PARAMS = {
    "meta_model_path": "files/models/meta_action_model.pkl",
    "base_targets": [
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
    # פרמטרי Pivot-Action
    "pivot_lookback": 7,
    "pivot_prominence_pct": 0.006,
    "pivot_min_spacing": 10,
    # פרמטרי אימון / ולידציה
    "val_size": 0.2,
    "random_state": 42,
    "lgb_params": {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "force_col_wise": True,
    },
    "label_cols": [
        "Target_Tomorrow",
        "Target_2_Days",
        "Target_3_Days",
        "Target_5_Days",
        "Target_7_Days",
        "Target_8_Days",
        "Target_13_Days",
        "Target_21_Days",
    ],}
