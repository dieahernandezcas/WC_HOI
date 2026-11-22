r"""O-information and its estimation.

For a set of variables X = (X_1, ..., X_n),

    Omega(X) = (n - 2) H(X) + sum_i H(X_i) - sum_i H(X_{-i}),

where X_{-i} is X without X_i. Omega > 0 indicates redundancy-dominated and
Omega < 0 synergy-dominated dependencies. For n = 3 this reduces to

    Omega = sum_i H(X_i) - sum_{i<j} H(X_i, X_j) + H(X_1, X_2, X_3).

Entropies are estimated with the Kozachenko-Leonenko nearest-neighbour
estimator using the maximum (Chebyshev) norm,

    H = psi(T) - psi(k) + d log 2 + (d / T) sum_t log eps_t,

where eps_t is the distance to the k-th nearest neighbour. The term d log 2
is the log-volume of the unit ball of the maximum norm and is only consistent
with distances measured in that norm.

A Gaussian-copula estimator is provided as an independent check: each
variable is mapped to a standard normal through its ranks and Omega is
evaluated in closed form from the covariance of the transformed data.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import norm, rankdata

_MIN_DISTANCE = 1e-300


def entropy_kl(x: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko entropy (nats) of samples ``x`` with shape (T, d)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    t, d = x.shape
    if t <= k:
        raise ValueError(f"need more samples than neighbours (T={t}, k={k})")
    dist, _ = cKDTree(x).query(x, k=k + 1, p=np.inf)
    eps = np.maximum(dist[:, k], _MIN_DISTANCE)
    return float(digamma(t) - digamma(k) + d * np.log(2.0) + d * np.mean(np.log(eps)))


def omega_from_entropy(x: np.ndarray, entropy) -> float:
    """O-information of the columns of ``x`` for a given entropy functional."""
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    idx = np.arange(n)
    h_full = entropy(x)
    h_single = sum(entropy(x[:, [i]]) for i in range(n))
    h_minus = sum(entropy(x[:, np.delete(idx, i)]) for i in range(n))
    return float((n - 2) * h_full + h_single - h_minus)


def omega_ksg(x: np.ndarray, k: int = 5) -> float:
    """O-information with the Kozachenko-Leonenko estimator."""
    return omega_from_entropy(x, lambda y: entropy_kl(y, k))


def _gaussian_logdet(cov: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.atleast_2d(cov))
    if sign <= 0:
        raise np.linalg.LinAlgError("covariance is not positive definite")
    return float(logdet)


def omega_gaussian(x: np.ndarray) -> float:
    """O-information of Gaussian variables in closed form (nats)."""
    x = np.asarray(x, dtype=float)
    cov = np.cov(x, rowvar=False)
    n = cov.shape[0]
    idx = np.arange(n)
    ld = lambda s: _gaussian_logdet(cov[np.ix_(s, s)])
    h = lambda s: 0.5 * ld(s)   # constants cancel in Omega
    return float((n - 2) * h(idx) + sum(h([i]) for i in idx)
                 - sum(h(np.delete(idx, i)) for i in idx))


def copula_normalise(x: np.ndarray) -> np.ndarray:
    """Map each column to a standard normal through its empirical ranks."""
    x = np.asarray(x, dtype=float)
    t = x.shape[0]
    ranks = np.apply_along_axis(rankdata, 0, x)
    return norm.ppf((ranks - 0.5) / t)


def omega_gaussian_copula(x: np.ndarray) -> float:
    """O-information with the Gaussian-copula estimator."""
    return omega_gaussian(copula_normalise(x))


def omega_triplets(signals: np.ndarray, triplets: Iterable[Iterable[int]],
                   k: int = 5) -> np.ndarray:
    """Kozachenko-Leonenko O-information of many triplets of one recording.

    ``signals`` has shape (T, N). Single-variable and pairwise entropies are
    computed once and reused across triplets, which gives exactly the same
    values as evaluating every triplet independently.
    """
    x = np.asarray(signals, dtype=float)
    trip = np.asarray(list(triplets), dtype=int)
    singles = {int(i): entropy_kl(x[:, [i]], k) for i in np.unique(trip)}
    pairs: dict[tuple[int, int], float] = {}
    out = np.empty(len(trip))
    for m, (a, b, c) in enumerate(trip):
        h2 = 0.0
        for p in combinations(sorted((int(a), int(b), int(c))), 2):
            if p not in pairs:
                pairs[p] = entropy_kl(x[:, list(p)], k)
            h2 += pairs[p]
        h3 = entropy_kl(x[:, [a, b, c]], k)
        out[m] = singles[int(a)] + singles[int(b)] + singles[int(c)] - h2 + h3
    return out


def omega_triplets_copula(signals: np.ndarray, triplets: Iterable[Iterable[int]]) -> np.ndarray:
    """Gaussian-copula O-information of many triplets of one recording."""
    z = copula_normalise(signals)
    cov = np.cov(z, rowvar=False)
    out = []
    for a, b, c in triplets:
        s = [a, b, c]
        sub = cov[np.ix_(s, s)]
        ld3 = _gaussian_logdet(sub)
        ld2 = sum(_gaussian_logdet(sub[np.ix_(p, p)]) for p in ([0, 1], [0, 2], [1, 2]))
        ld1 = sum(np.log(sub[i, i]) for i in range(3))
        out.append(0.5 * (ld1 - ld2 + ld3))
    return np.asarray(out)
