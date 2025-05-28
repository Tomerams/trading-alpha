# src/models/meta_loader.py
"""
Load & cache the meta-action model (LightGBM / any sklearn pipeline).

Usage:
    from models.meta_loader import load_meta_model
    meta = load_meta_model()                     # uses default path
    meta = load_meta_model("files/…/my.pkl")     # custom path
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib

# ────────────────────────────────────
# Environment tweaks to avoid thread dead-locks
# ────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")        # OpenMP → single thread
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")  # fixes macOS + MKL clash

logger = logging.getLogger(__name__)

# Cache (module-level singletons)
_cached_meta_path: Optional[Path] = None
_cached_meta_model: Optional[Any] = None

# Default artefact location (relative to repo root)
DEFAULT_META_PATH = Path("files/models/meta_action_model.pkl")


def load_meta_model(model_path: str | os.PathLike | None = None,
                    *, use_cache: bool = True) -> Any:
    """
    Load the meta-model (pickled with joblib) and return the instance.

    Parameters
    ----------
    model_path : str | Path | None
        Path to the `.pkl` file.  If None ⇒ `DEFAULT_META_PATH`.
    use_cache : bool, default=True
        If True and the requested model is already in memory, return it instead
        of re-loading from disk.

    Returns
    -------
    Any
        The deserialised model / pipeline (usually LightGBM, Stacking, etc.).
    """
    global _cached_meta_model, _cached_meta_path

    path = Path(model_path or DEFAULT_META_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"[meta_loader] model not found: {path}")

    if use_cache and _cached_meta_model is not None and _cached_meta_path == path:
        logger.info(f"[meta_loader] ⚡ returning cached meta-model → {path}")
        return _cached_meta_model

    logger.info(f"[meta_loader] ⏳ loading meta-model from: {path}")
    # mmap_mode="r" ⇒ joblib maps the file into memory; no full RAM copy
    meta = joblib.load(path, mmap_mode="r")
    logger.info("[meta_loader] ✅ meta-model loaded")

    if use_cache:
        _cached_meta_model = meta
        _cached_meta_path = path

    return meta
