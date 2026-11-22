"""Surrogate signals for null hypotheses.

``covariance_preserving``: Gaussian signals with the mean and the covariance
between regions of the recording. They keep pairwise dependencies only.

``phase_randomised``: the amplitude spectrum of each region is kept and the
Fourier phases are redrawn independently for each region. Temporal structure
within regions is kept; temporal relations between regions are destroyed.
The marginal distributions become approximately Gaussian.

``circular_shift``: each region is rotated in time by an independent random
lag. The marginal distribution and the autocorrelation of every region are
kept exactly; only the relations between regions are destroyed. This is the
appropriate null for the nearest-neighbour O-information estimator, whose
finite-sample bias depends on the marginal distributions.
"""

from __future__ import annotations

import numpy as np


def covariance_preserving(signals: np.ndarray, seed: int) -> np.ndarray:
    """Gaussian surrogate of ``signals`` with shape (T, N)."""
    x = np.asarray(signals, dtype=float)
    rng = np.random.default_rng(seed)
    cov = np.cov(x, rowvar=False) + 1e-10 * np.eye(x.shape[1])
    return rng.multivariate_normal(x.mean(axis=0), cov, size=x.shape[0])


def phase_randomised(signals: np.ndarray, seed: int) -> np.ndarray:
    """Independent phase randomisation of each column of ``signals`` (T, N)."""
    x = np.asarray(signals, dtype=float)
    t = x.shape[0]
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(x, axis=0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.shape)
    phases[0] = 0.0
    if t % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n=t, axis=0)


def circular_shift(signals: np.ndarray, seed: int, min_shift: int = 50) -> np.ndarray:
    """Rotate each column of ``signals`` (T, N) by an independent random lag.

    Lags are drawn uniformly in [min_shift, T - min_shift), so that no region
    stays close to its original alignment.
    """
    x = np.asarray(signals, dtype=float)
    t = x.shape[0]
    if t <= 2 * min_shift:
        raise ValueError("series too short for the requested minimum shift")
    rng = np.random.default_rng(seed)
    lags = rng.integers(min_shift, t - min_shift, size=x.shape[1])
    out = np.empty_like(x)
    for j, lag in enumerate(lags):
        out[:, j] = np.roll(x[:, j], lag)
    return out
