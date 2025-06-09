# backtest/meta_loader.py
import logging
import os
import joblib

logger = logging.getLogger("meta_loader")


def load_meta_pipeline(path: str, timeout: float = 5.0):
    if not os.path.exists(path):
        msg = f"Meta pipeline file not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"[meta_loader] Loading pipeline from: {path}")
    try:
        pipeline = joblib.load(path)
        logger.info("[meta_loader] Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Error loading meta pipeline: {e}")
        raise

    # Validate pipeline structure
    if (
        not isinstance(pipeline, dict)
        or "base" not in pipeline
        or "meta" not in pipeline
    ):
        msg = f"Invalid pipeline format: expected dict with 'base' and 'meta' keys"
        logger.error(msg)
        raise ValueError(msg)

    return pipeline
