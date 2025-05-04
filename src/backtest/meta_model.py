import pandas as pd
import joblib
from lightgbm import LGBMClassifier

def create_meta_ai_dataset(preds, true_vals, prices, target_cols, buy_threshold=0.002, sell_threshold=-0.002):
    df = pd.DataFrame(preds, columns=[f'Pred_{col}' for col in target_cols])
    actual_returns = true_vals[:, target_cols.index('Target_Tomorrow')]

    def decide_action(ret):
        if ret > buy_threshold:
            return 2  # BUY
        elif ret < sell_threshold:
            return 0  # SELL
        else:
            return 1  # HOLD

    df['Action'] = [decide_action(r) for r in actual_returns]
    return df

def train_meta_model(meta_df):
    X = meta_df.drop(columns=['Action'])
    y = meta_df['Action']
    model = LGBMClassifier(n_estimators=200, max_depth=5)
    model.fit(X, y)
    joblib.dump(model, 'files/models/meta_action_model.pkl')
    return model

def load_meta_model():
    try:
        return joblib.load('files/models/meta_action_model.pkl')
    except FileNotFoundError:
        return None
