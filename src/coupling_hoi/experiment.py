"""Running grids of simulations and summarising each simulation.

A *condition* is a network, a normalisation scheme and a coupling
specification. For every condition the model is simulated on a grid of
parameter values with several noise replicates. The noise seed of a
simulation depends on the network, normalisation, coupling order, parameter
values and replicate, but not on the coupling form: additive and diffusive
simulations of the same condition share their noise and can be compared as
paired samples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from .config import (CouplingSpec, InformationParams, IntegrationParams, PreprocessingParams,
                     WilsonCowanParams, stable_seed)
from .coupling import CouplingOperator
from .information import omega_triplets
from .metrics import fc_similarity, functional_connectivity, upper_triangle
from .model import simulate
from .networks import normalise
from .preprocessing import preprocess


@dataclass(frozen=True)
class GridPoint:
    drive: float
    coupling: float
    replicate: int
    gain: float = 1.0


def grid_points(drives, couplings, n_replicates, gains=(1.0,)) -> list[GridPoint]:
    return [GridPoint(float(p), float(k), r, float(g))
            for g in gains for p in drives for k in couplings for r in range(n_replicates)]


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def condition_seed(base_seed: int, network: str, normalisation: str, spec: CouplingSpec,
                   point: GridPoint) -> int:
    """Noise seed of one simulation; independent of the coupling form."""
    keys = [network, normalisation, spec.order, point.drive, point.coupling]
    if point.gain != 1.0:
        keys.append(point.gain)
    return stable_seed(base_seed, *keys, point.replicate)


def run_condition(adjacency: np.ndarray,
                  network: str,
                  normalisation: str,
                  spec: CouplingSpec,
                  points: list[GridPoint],
                  triplets: np.ndarray,
                  base_seed: int,
                  model: WilsonCowanParams = WilsonCowanParams(),
                  integration: IntegrationParams = IntegrationParams(),
                  preprocessing: PreprocessingParams = PreprocessingParams(),
                  information: InformationParams = InformationParams(),
                  fc_reference: np.ndarray | None = None,
                  batch_size: int = 64,
                  keep_signals: bool = False) -> dict[str, np.ndarray]:
    """Simulate every point of one condition and summarise each run.

    Returns arrays aligned with ``points``:
        omega          (n_points, n_triplets)  triplet O-information
        mean_activity  mean excitatory activity (raw)
        mean_std       mean temporal standard deviation (raw)
        mean_fc        mean off-diagonal correlation (preprocessed)
        fc_similarity  correlation with ``fc_reference`` (NaN if not given)
        seeds          noise seed of each run
        signals        raw excitatory activity, float32 (n_points, T, N);
                       only when ``keep_signals`` is True
    """
    op = CouplingOperator.from_adjacency(normalise(adjacency, normalisation))
    seeds = [condition_seed(base_seed, network, normalisation, spec, p) for p in points]
    n = len(points)
    res = {
        "omega": np.empty((n, len(triplets))),
        "mean_activity": np.empty(n), "mean_std": np.empty(n),
        "mean_fc": np.empty(n), "fc_similarity": np.full(n, np.nan),
    }
    if keep_signals:
        res["signals"] = np.empty((n, integration.n_records, op.d.shape[0]), dtype=np.float32)
    for idx in _batches(list(range(n)), batch_size):
        raw = simulate(op, spec, [points[i].drive for i in idx], [points[i].coupling for i in idx],
                       [seeds[i] for i in idx], model, integration,
                       gain=[points[i].gain for i in idx])
        if keep_signals:
            res["signals"][idx] = raw.astype(np.float32)
        for j, i in enumerate(idx):
            x = raw[j]                                        # (time, nodes)
            y = preprocess(x, preprocessing, axis=0)
            fc = functional_connectivity(y)
            res["omega"][i] = omega_triplets(y, triplets, information.k_neighbours)
            res["mean_activity"][i] = x.mean()
            res["mean_std"][i] = x.std(axis=0).mean()
            res["mean_fc"][i] = np.nanmean(upper_triangle(fc))
            if fc_reference is not None:
                res["fc_similarity"][i] = fc_similarity(fc, fc_reference)
    res["seeds"] = np.asarray(seeds, dtype=np.int64)
    return res


def deterministic_amplitude(adjacency: np.ndarray, normalisation: str, spec: CouplingSpec,
                            points: list[GridPoint],
                            model: WilsonCowanParams = WilsonCowanParams(),
                            integration: IntegrationParams = IntegrationParams(),
                            batch_size: int = 256) -> np.ndarray:
    """Oscillation amplitude of the noise-free network at each point.

    Returns the temporal standard deviation of the excitatory activity,
    averaged over nodes, after the transient. Values above a small threshold
    indicate that the deterministic network settles on an oscillation rather
    than on a fixed point.
    """
    op = CouplingOperator.from_adjacency(normalise(adjacency, normalisation))
    quiet = replace(model, sigma=0.0)
    amp = np.empty(len(points))
    for idx in _batches(list(range(len(points))), batch_size):
        raw = simulate(op, spec, [points[i].drive for i in idx], [points[i].coupling for i in idx],
                       [7] * len(idx), quiet, integration, gain=[points[i].gain for i in idx])
        amp[idx] = raw.std(axis=1).mean(axis=1)
    return amp
