from config.backtest_config import BACKTEST_PARAMS
from routers.routers_entities import UpdateIndicatorsData
import pandas as pd
import itertools


def optimize_signal_params(
    request_data: UpdateIndicatorsData, param_grid: dict
) -> pd.DataFrame:
    """
    Runs a grid search over signal parameters and returns a DataFrame sorted by net_profit.

    param_grid keys should match MODEL_PARAMS entries (e.g. 'buying_threshold', etc.).
    """
    param_keys = list(param_grid.keys())
    original = {k: BACKTEST_PARAMS.get(k) for k in param_keys}

    results = []
    grids = [param_grid[k] for k in param_keys]
    for combo in itertools.product(*grids):
        override = dict(zip(param_keys, combo))
        BACKTEST_PARAMS.update(override)
        res = backtest_model(request_data, verbose=False)
        entry = {
            **override,
            "net_profit": res["net_profit"],
            "ticker_change": res["ticker_change"],
            "max_loss_per_trade": res["max_loss_per_trade"],
            "num_trades": len(res["trades_signals"]),
        }
        results.append(entry)

    BACKTEST_PARAMS.update(original)
    df = pd.DataFrame(results)
    return df.sort_values("net_profit", ascending=False).reset_index(drop=True)
