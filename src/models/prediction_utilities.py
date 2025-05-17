from scipy.signal import find_peaks
import pandas as pd

def generate_swing_signals(
    close: pd.Series,
    window: int = 5,
    prominence: float = 0.01
) -> pd.Series:
    peaks, _ = find_peaks(close.values, distance=window, prominence=prominence)
    troughs, _ = find_peaks(-close.values, distance=window, prominence=prominence)
    events = sorted(
        [(i, 'buy') for i in troughs] + [(i, 'sell') for i in peaks],
        key=lambda x: x[0]
    )
    state = 'buy'
    trades = []
    for idx, action in events:
        if action == state:
            trades.append((idx, action))
            state = 'sell' if state == 'buy' else 'buy'
    signals = pd.Series(0, index=close.index)
    for idx, action in trades:
        signals.iloc[idx] = 1 if action == 'buy' else -1
    return signals

def add_swing_column(
    df: pd.DataFrame,
    close_col: str = "Close",
    signal_col: str = "swing_signal",
    window: int = 5,
    prominence: float = 0.01
) -> pd.DataFrame:
    signals = generate_swing_signals(df[close_col], window=window, prominence=prominence)
    # מחזיר DataFrame חדש עם העמודה החדשה
    return df.assign(**{signal_col: signals})
