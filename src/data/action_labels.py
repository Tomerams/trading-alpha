import numpy as np


# ── BUY / HOLD / SELL ע"פ קוונטילים ─────────────────────────────────────────
def make_action_label_quantile(df, horizon: str = "Target_3_Days", q: float = 0.20):
    ret = df[horizon]
    high = np.quantile(ret, 1 - q)
    low = np.quantile(ret, q)
    return np.where(ret >= high, 2, np.where(ret <= low, 0, 1))
