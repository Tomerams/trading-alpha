import numpy as np
from config import MODEL_PARAMS
from data.features import Pattern, ExternalDerivedFeatures, BinaryIndicator, DateFeatures


def get_exclude_from_scaling() -> set:
    """
    Build and return a set of column names to exclude from feature scaling,
    based on MODEL_PARAMS and enums from data.features.
    """
    base = ["Date", "Close"]
    # shift-based targets
    targets = MODEL_PARAMS.get("target_cols", [])
    # pattern features
    pattern_cols = [p.value for p in Pattern]
    # external derived features
    external_cols = [
        ExternalDerivedFeatures.SPY_VIX_RATIO.value,
        ExternalDerivedFeatures.YIELD_SPREAD_10Y_2Y.value,
    ]
    # binary indicators
    binary_cols = [i.value for i in BinaryIndicator]
    # date-related features
    datefeature_cols = [d.value for d in DateFeatures]

    return set(base + targets + pattern_cols + external_cols + binary_cols + datefeature_cols)
