r"""Coupling inputs between network nodes.

For node ``i`` with activity ``x_i`` and incoming weights ``A[i, j]``:

Pairwise (second-order) inputs
    additive   C_i = sum_j A_ij x_j
    diffusive  C_i = sum_j A_ij (x_j - x_i) = sum_j A_ij x_j - d_i x_i

Fan (third-order) inputs, over ordered pairs of distinct neighbours j != k
    additive   F_i = sum_{j != k} A_ij A_ik x_j x_k
    diffusive  F_i = sum_{j != k} A_ij A_ik (x_j - x_i)(x_k - x_i)

The fan sums are evaluated in O(N^2) with the identity
``sum_{j != k} w_j w_k y_j y_k = (sum_j w_j y_j)^2 - sum_j w_j^2 y_j^2``.

All functions accept a batch of states with shape ``(B, N)`` and return an
array of the same shape, so that many simulations can be advanced at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CouplingOperator:
    """Precomputed matrices for fast evaluation of all coupling inputs."""

    a_t: np.ndarray        # A transposed, for right-multiplication of (B, N) states
    a2_t: np.ndarray       # elementwise square of A, transposed
    d: np.ndarray          # in-strength, shape (N,)
    d2: np.ndarray         # row sums of A * A, shape (N,)

    @classmethod
    def from_adjacency(cls, adjacency: np.ndarray) -> "CouplingOperator":
        a = np.asarray(adjacency, dtype=float)
        if np.any(np.diag(a) != 0):
            raise ValueError("adjacency must have a zero diagonal")
        a2 = a * a
        return cls(a_t=np.ascontiguousarray(a.T), a2_t=np.ascontiguousarray(a2.T),
                   d=a.sum(axis=1), d2=a2.sum(axis=1))

    # ------------------------------------------------------------------ #
    def pairwise(self, x: np.ndarray, form: str) -> np.ndarray:
        ax = x @ self.a_t
        if form == "additive":
            return ax
        if form == "diffusive":
            return ax - self.d * x
        raise ValueError(f"unknown coupling form: {form!r}")

    def fan(self, x: np.ndarray, form: str) -> np.ndarray:
        if form == "additive":
            s1 = x @ self.a_t
            s2 = (x * x) @ self.a2_t
            return s1 * s1 - s2
        if form == "diffusive":
            s1 = x @ self.a_t - self.d * x
            a2x = x @ self.a2_t
            s2 = (x * x) @ self.a2_t - 2.0 * x * a2x + self.d2 * x * x
            return s1 * s1 - s2
        raise ValueError(f"unknown coupling form: {form!r}")

    def input(self, x: np.ndarray, form: str, order: str, fan_ratio: float = 20.0) -> np.ndarray:
        if order == "pairwise":
            return self.pairwise(x, form)
        if order == "fan":
            return self.fan(x, form)
        if order == "pairwise+fan":
            return self.pairwise(x, form) + fan_ratio * self.fan(x, form)
        raise ValueError(f"unknown coupling order: {order!r}")


# --------------------------------------------------------------------------- #
# Reference implementations (used by the test suite)
# --------------------------------------------------------------------------- #

def pairwise_reference(a: np.ndarray, x: np.ndarray, form: str) -> np.ndarray:
    """Direct double-loop evaluation of the pairwise input for one state."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            y = x[j] if form == "additive" else x[j] - x[i]
            out[i] += a[i, j] * y
    return out


def fan_reference(a: np.ndarray, x: np.ndarray, form: str) -> np.ndarray:
    """Direct triple-loop evaluation of the fan input for one state."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if j == k:
                    continue
                if form == "additive":
                    out[i] += a[i, j] * a[i, k] * x[j] * x[k]
                else:
                    out[i] += a[i, j] * a[i, k] * (x[j] - x[i]) * (x[k] - x[i])
    return out
