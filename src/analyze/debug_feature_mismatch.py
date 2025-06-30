# debug_feature_mismatch.py  – הרץ פעם אחת
from datetime import date, timedelta
from pathlib import Path
import pandas as pd, numpy as np, joblib, torch, json
from routers.routers_entities import UpdateIndicatorsData
from data.data_processing import get_indicators_data

TICKER  = "QQQ"
TARGET  = "Target_Tomorrow"
SEQ_LEN = 60
stem    = Path("src/files/models") / f"{TICKER}_{TARGET}"

feats   = joblib.load(stem.parent / f"{stem.name}_features.pkl")
scaler  = joblib.load(stem.parent / f"{stem.name}_scaler.pkl")

req = UpdateIndicatorsData(
    stock_ticker=TICKER,
    start_date =(date.today() - timedelta(days=365*20)).isoformat(),
    end_date   = date.today().isoformat(),
    scale=True,
)
df = get_indicators_data(req)

# 1) האם חסרים או מיותרים פיצ'רים?
missing = [f for f in feats if f not in df.columns]
extra   = [c for c in df.columns if c in feats and c not in feats]
print("Missing features:", missing)
print("Extra   features:", extra)

# 2) ערכי סטטיסטיקה לפני ואחרי scaler
X_raw = np.lib.stride_tricks.sliding_window_view(
            df[feats].values, (SEQ_LEN, len(feats))
        )[:-1].reshape(-1, len(feats))
print("\nRaw  mean ± std:", X_raw.mean(), X_raw.std())

X_z   = scaler.transform(X_raw)
print("Scaled mean ± std:", X_z.mean(), X_z.std())
print("Scaled min / max :", X_z.min(),  X_z.max())