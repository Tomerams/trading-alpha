#!/usr/bin/env python3
"""
מדפיס BUY-AUC ו-SELL-AUC עבור כל V_target
(ROC One-Vs-Rest; זהה לשיטה של Meta-Model)
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

META_PKL = Path("src/files/datasets/meta_dataset_scalar.pkl")  # עדכן אם שינית
df = pd.read_pickle(META_PKL)

# אותה פונקציית derive_action
y_raw = df["Return_3d"].to_numpy()
q_hi, q_lo = np.quantile(y_raw, [0.67, 0.33])
y = np.where(y_raw >= q_hi, 2, np.where(y_raw <= q_lo, 0, 1)).astype(int)


def auc_ovr(y_true, y_score, pos_label):
    """ROC-AUC One-Vs-Rest"""
    y_bin = (y_true == pos_label).astype(int)
    return roc_auc_score(y_bin, y_score)


print("\nAUC per base scalar")
print("-" * 41)
print(f"{'scalar':<25}  BUY  SELL  max")

for col in df.filter(like="V_").columns:
    buy_auc = auc_ovr(y, df[col], 2)
    sell_auc = auc_ovr(y, df[col], 0)
    best = max(buy_auc, sell_auc)
    print(f"{col:<25}  {buy_auc:.3f}  {sell_auc:.3f}  {best:.3f}")
