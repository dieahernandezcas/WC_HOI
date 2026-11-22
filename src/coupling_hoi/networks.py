"""Network construction and connectome normalisation.

Convention: ``A[i, j]`` is the weight of the link from node ``j`` to node
``i`` (row = receiving node). The in-strength of node ``i`` is
``d_i = sum_j A[i, j]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMALISATIONS = ("global", "row")


@dataclass(frozen=True)
class Network:
    """A named weighted adjacency matrix with zero diagonal."""

    name: str
    adjacency: np.ndarray

    def __post_init__(self) -> None:
        a = self.adjacency
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if np.any(np.diag(a) != 0):
            raise ValueError("adjacency must have a zero diagonal")
        if np.any(a < 0):
            raise ValueError("adjacency must be non-negative")

    @property
    def n_nodes(self) -> int:
        return self.adjacency.shape[0]

    @property
    def in_strength(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)


# --------------------------------------------------------------------------- #
# Synthetic networks
# --------------------------------------------------------------------------- #

def three_node_all_to_all() -> Network:
    """Three nodes, every pair linked with unit weight."""
    a = np.ones((3, 3)) - np.eye(3)
    return Network("three_node", a)


def ring_with_shortcuts(n: int = 10, reach: int = 2,
                        shortcuts: tuple[tuple[int, int], ...] = ((0, 5), (2, 7))) -> Network:
    """Ring lattice in which each node links to its ``reach`` nearest
    neighbours on each side, plus symmetric long-range shortcuts."""
    a = np.zeros((n, n))
    for i in range(n):
        for off in range(1, reach + 1):
            for j in ((i + off) % n, (i - off) % n):
                a[i, j] = a[j, i] = 1.0
    for i, j in shortcuts:
        a[i, j] = a[j, i] = 1.0
    np.fill_diagonal(a, 0.0)
    return Network("ring", a)


def erdos_renyi(n: int = 10, p: float = 0.45, seed: int = 2025) -> Network:
    """Undirected Erdos-Renyi graph with unit weights.

    A node left without links is connected to one randomly chosen node, so
    that every node receives input. The draw order reproduces the random
    graph used in earlier simulations of this project for the same seed.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                a[i, j] = a[j, i] = 1.0
    for i in range(n):
        if a[i].sum() == 0:
            j = int(rng.integers(0, n))
            while j == i:
                j = int(rng.integers(0, n))
            a[i, j] = a[j, i] = 1.0
    return Network("random", a)


def synthetic_networks() -> dict[str, Network]:
    """The three synthetic networks of the study, in a fixed order."""
    nets = [three_node_all_to_all(), ring_with_shortcuts(), erdos_renyi()]
    return {net.name: net for net in nets}


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def normalise(adjacency: np.ndarray, scheme: str) -> np.ndarray:
    """Rescale a connectome.

    ``"global"``: divide every weight by the mean in-strength. Relative
    differences in in-strength between nodes are preserved and the mean
    in-strength becomes 1, so that a given coupling strength has a comparable
    meaning across networks.

    ``"row"``: divide each row by its own sum, so that every node has
    in-strength 1. Differences in in-strength between nodes are removed;
    in this case the diffusive self-coupling term ``-K d_i x_i`` reduces to
    the same decay ``-K x_i`` at every node.

    Nodes without incoming links (possible in directed connectomes) keep a
    zero row under both schemes.
    """
    a = np.asarray(adjacency, dtype=float)
    d = a.sum(axis=1)
    if np.any(d < 0) or not np.any(d > 0):
        raise ValueError("in-strengths must be non-negative and not all zero")
    if scheme == "global":
        return a / d.mean()
    if scheme == "row":
        safe = np.where(d > 0, d, 1.0)
        return a / safe[:, None]
    raise ValueError(f"unknown normalisation: {scheme!r}; expected one of {NORMALISATIONS}")


def in_strength_heterogeneity(adjacency: np.ndarray) -> float:
    """Coefficient of variation of the in-strength (0 for regular networks)."""
    d = np.asarray(adjacency, dtype=float).sum(axis=1)
    return float(d.std() / d.mean())
