#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUC per V_target  (One-Vs-Rest)
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

META_PKL = Path("src/files/datasets/meta_dataset_scalar.pkl")  # <- עדכן אם צריך

if not META_PKL.exists():
    raise SystemExit(f"❌ {META_PKL} not found. הרץ קודם meta_dataset!")

df = pd.read_pickle(META_PKL)

y = df["action_label"].to_numpy()


def auc_ovr(y_true, y_score, pos):
    return roc_auc_score((y_true == pos).astype(int), y_score)


print("\nAUC per base scalar")
print("-" * 46)
print(f"{'scalar':<25}  BUY   SELL   max")

for col in df.filter(like="V_").columns:
    buy = auc_ovr(y, df[col], 2)
    sell = auc_ovr(y, df[col], 0)
    best = max(buy, sell)
    print(f"{col:<25}  {buy:5.3f}  {sell:5.3f}  {best:5.3f}")
