import numpy as np
import pandas as pd
from data.action_labels import make_action_label_quantile
from data.features import calculate_features
from data.targets import calculate_targets, make_action_label
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
    df["action_label"] = make_action_label_quantile(df, horizon="Target_3_Days", q=0.20)

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = get_exclude_from_scaling()
    to_scale = [c for c in numeric if c not in exclude]
    df[to_scale] = df[to_scale].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

