from config import BinaryIndicator, DateFeatures, ExternalDerivedFeatures, Pattern


exclude_from_scaling = set(
    ["Date", "Target_Tomorrow", "Target_3_Days", "Target_Next_Week", "Close"]
    + [p.value for p in Pattern]
    + [
        ExternalDerivedFeatures.SPY_VIX_RATIO.value,
        ExternalDerivedFeatures.YIELD_SPREAD_10Y_2Y.value,
    ]
    + [i.value for i in BinaryIndicator]
    + [d.value for d in DateFeatures]
)
