"""Loading of the larval zebrafish recordings and structural connectome.

Raw files (65 regions per hemisphere, 130 nodes in total):

    timeseries_65.npy      (fish, regions, time) calcium signals
    SC_undirected_65.npy   symmetric structural connectome
    SC_directed_65.npy     asymmetric structural connectome

A region is kept when its signal is non-constant in every fish. The triplets
of regions analysed are drawn once, uniformly without replacement, with a
fixed seed, and are shared by the data, the surrogates and all models.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .config import RAW_ZEBRAFISH_DIR

FLAT_STD_THRESHOLD = 1e-10
SYMMETRY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ZebrafishDataset:
    """Recordings and connectome restricted to the usable regions."""

    timeseries: np.ndarray        # (fish, regions, time), raw signals
    sc_undirected: np.ndarray     # (regions, regions), zero diagonal
    sc_directed: np.ndarray       # (regions, regions), zero diagonal
    region_index: np.ndarray      # indices of the kept regions in the raw files

    @property
    def n_fish(self) -> int:
        return self.timeseries.shape[0]

    @property
    def n_regions(self) -> int:
        return self.timeseries.shape[1]

    @property
    def n_samples(self) -> int:
        return self.timeseries.shape[2]


def usable_regions(timeseries: np.ndarray) -> np.ndarray:
    """Boolean mask of regions whose signal is non-constant in every fish."""
    flat = timeseries.std(axis=2) < FLAT_STD_THRESHOLD
    return ~flat.any(axis=0)


def _clean_connectome(sc: np.ndarray, keep: np.ndarray) -> np.ndarray:
    a = np.array(sc, dtype=float)
    np.fill_diagonal(a, 0.0)
    return a[np.ix_(keep, keep)]


def load_zebrafish(data_dir: str | Path = RAW_ZEBRAFISH_DIR) -> ZebrafishDataset:
    """Load the raw files and restrict them to the usable regions."""
    data_dir = Path(data_dir)
    ts = np.load(data_dir / "timeseries_65.npy")
    sc_u = np.load(data_dir / "SC_undirected_65.npy")
    sc_d = np.load(data_dir / "SC_directed_65.npy")
    if ts.ndim != 3 or sc_u.shape != (ts.shape[1],) * 2 or sc_d.shape != sc_u.shape:
        raise ValueError("unexpected shapes in the raw zebrafish files")
    if not np.allclose(sc_u, sc_u.T):
        raise ValueError("the undirected connectome is not symmetric")
    keep = usable_regions(ts)
    return ZebrafishDataset(
        timeseries=ts[:, keep, :],
        sc_undirected=_clean_connectome(sc_u, keep),
        sc_directed=_clean_connectome(sc_d, keep),
        region_index=np.flatnonzero(keep),
    )


def sample_triplets(n_regions: int, n_triplets: int, seed: int) -> np.ndarray:
    """Draw distinct region triplets uniformly without replacement."""
    all_triplets = list(combinations(range(n_regions), 3))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_triplets), size=n_triplets, replace=False)
    return np.asarray([all_triplets[i] for i in idx], dtype=int)


def symmetric_triplets(sc_directed: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """True for triplets whose three links have equal weight in both directions."""
    diff = np.abs(sc_directed - sc_directed.T)
    return np.array([diff[a, b] < SYMMETRY_TOLERANCE and diff[a, c] < SYMMETRY_TOLERANCE
                     and diff[b, c] < SYMMETRY_TOLERANCE for a, b, c in triplets])
