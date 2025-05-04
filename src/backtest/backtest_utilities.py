import pandas as pd


def decide_action_meta(meta_model, preds, target_cols, i):
    row = pd.DataFrame([preds[i]], columns=[f'Pred_{col}' for col in target_cols])
    action = meta_model.predict(row)[0]
    return action  # 0 = SELL, 1 = HOLD, 2 = BUY
