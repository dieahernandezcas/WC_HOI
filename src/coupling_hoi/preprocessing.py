"""Signal preprocessing applied identically to recorded and simulated data.

Steps, along the time axis: zero-phase Butterworth high-pass filter, removal
of a linear trend, and z-scoring.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt

from .config import PreprocessingParams

_EPS = 1e-10


def preprocess(signals: np.ndarray,
               params: PreprocessingParams = PreprocessingParams(),
               axis: int = -1) -> np.ndarray:
    """Return high-pass filtered, detrended and z-scored signals.

    ``signals`` may have any shape; the operations act along ``axis``.
    """
    x = np.moveaxis(np.asarray(signals, dtype=float), axis, -1)
    nyquist = params.sampling_rate_hz / 2.0
    b, a = butter(params.highpass_order, params.highpass_cutoff_hz / nyquist, btype="high")
    x = filtfilt(b, a, x, axis=-1)
    x = detrend(x, axis=-1)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + _EPS)
    return np.moveaxis(x, -1, axis)
