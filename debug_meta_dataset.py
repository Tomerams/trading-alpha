# debug_meta_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path

path = Path("./src/files/datasets/meta_dataset.pkl")  # הנתיב הקיים אצלך
df = pd.read_pickle(path)

# אם Action לא קיים – צור אותו בדומה לקוד שלך
if "Action" not in df.columns:
    from src.models.model_meta_trainer import _derive_action

    df["Action"] = _derive_action(df)
print("\n🧩 feature columns:", [c for c in df.columns if c.startswith("Pred_")])

print("⏹  meta_dataset shape:", df.shape)
print("\n🔎 Action value counts:")
print(df["Action"].value_counts(dropna=False))

print("\n📊 Return_3d describe:")
print(df["Return_3d"].describe())
