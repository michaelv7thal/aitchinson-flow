"""Shared AUROC helpers for benchmark tasks and plotting utilities.

Provides a single NaN-safe AUROC implementation used by every consumer so
benchmark metrics and diagnostic plots stay numerically aligned.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_1d_finite(arr: Any) -> np.ndarray:
    if arr is None:
        return np.empty(0, dtype=np.float32)
    flat = np.asarray(arr).ravel()
    return flat[np.isfinite(flat)]


def safe_auroc(valid: Any, invalid: Any) -> float:
    """AUROC with ``valid=0`` / ``invalid=1`` labels; returns NaN if a class is empty.

    Inputs may be lists, numpy arrays, or ``None``. Non-finite entries are
    dropped before scoring. When the concatenated score vector still contains
    non-finite values, they are replaced with ``0.0`` so that sklearn does not
    raise — matching the historical behavior of the benchmark helpers.
    """
    v = _to_1d_finite(valid)
    i = _to_1d_finite(invalid)
    if v.size == 0 or i.size == 0:
        return float("nan")

    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    labels = np.concatenate([np.zeros(v.size), np.ones(i.size)])
    scores = np.concatenate([v, i])
    if not np.isfinite(scores).all():
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return float(roc_auc_score(labels, scores))


__all__ = ["safe_auroc"]
