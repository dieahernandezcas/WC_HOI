"""Descriptive measures of simulated and recorded activity."""

from __future__ import annotations

import numpy as np


def functional_connectivity(signals: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of signals with shape (T, N)."""
    return np.corrcoef(np.asarray(signals, dtype=float), rowvar=False)


def upper_triangle(m: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(m.shape[0], k=1)
    return m[i, j]


def fc_similarity(fc_model: np.ndarray, fc_data: np.ndarray) -> float:
    """Pearson correlation between the off-diagonal entries of two FC matrices."""
    a, b = upper_triangle(fc_model), upper_triangle(fc_data)
    if np.std(a) == 0 or np.std(b) == 0 or np.any(~np.isfinite(a)):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
