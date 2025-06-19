import numpy as np
import pandas as pd
from data.action_labels import make_action_label_clean
from data.features import calculate_features
from data.targets import calculate_targets
from routers.routers_entities import UpdateIndicatorsData
from data.data_utilities import get_data, get_exclude_from_scaling


def get_indicators_data(request_data: UpdateIndicatorsData) -> pd.DataFrame:
    # 1) Load and prepare raw price data
    df = get_data(
        stock_ticker=request_data.stock_ticker,
        start_date=request_data.start_date,
        end_date=request_data.end_date,
    )
    df = df.rename_axis("Date").reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = calculate_features(df)
    df = calculate_targets(df)
    df["action_label"] = make_action_label_clean(
            df,
            up_thr = 0.015,          # BUY אם יש “בשר” של -1.5 %
            dn_thr = -0.015,         # SELL אם ירידה ≥-1.5 %
            atr_mult = 1.0           # וגם Δ ≥ ATR ממוצע
    )    
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = get_exclude_from_scaling()
    to_scale = [c for c in numeric if c not in exclude]
    df[to_scale] = df[to_scale].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df
