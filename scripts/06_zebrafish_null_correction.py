"""Null-corrected O-information of the zebrafish recordings and models.

The nearest-neighbour estimator has a positive finite-sample bias whose size
depends on the marginal distribution and autocorrelation of the signals. To
compare recordings and models on an equal footing, every recording and every
simulation is compared with its own circular-shift surrogates, which keep
these properties and destroy only the relations between regions:

    Delta Omega_t = Omega_t(signals) - mean_r Omega_t(surrogate r),

for each triplet t. A recording or simulation is summarised by the median of
Delta Omega over the triplets of a group.

Inputs: results/zebrafish_data/ (script 01) and results/main/zebrafish_final/
(script 04). Output: results/main/zebrafish_null_corrected.npz

Usage:
    python scripts/06_zebrafish_null_correction.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RESULTS_DIR, ExperimentConfig, stable_seed  # noqa: E402
from coupling_hoi.data import load_zebrafish  # noqa: E402
from coupling_hoi.information import omega_triplets  # noqa: E402
from coupling_hoi.preprocessing import preprocess  # noqa: E402
from coupling_hoi.surrogates import circular_shift  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main_experiment.json"
FINAL = RESULTS_DIR / "main" / "zebrafish_final"
OUT = RESULTS_DIR / "main" / "zebrafish_null_corrected.npz"
N_SURROGATES = 3


def null_omega(job: tuple) -> tuple[str, np.ndarray]:
    """Mean circular-shift O-information per triplet for a stack of recordings.

    ``job`` = (key, preprocessed signals (n, T, N), triplets, base seed, k).
    """
    key, signals, triplets, base_seed, k = job
    out = np.empty((len(signals), len(triplets)))
    for m, x in enumerate(signals):
        reps = [omega_triplets(circular_shift(x, stable_seed(base_seed, "circular", key, m, r)), triplets, k)
                for r in range(N_SURROGATES)]
        out[m] = np.mean(reps, axis=0)
    return key, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_json(CONFIG)
    ex = cfg.extra["zebrafish"]
    k = cfg.information.k_neighbours
    triplets = np.load(RESULTS_DIR / "zebrafish_data" / "triplets.npy")

    observed: dict[str, np.ndarray] = {}
    jobs = []
    ds = load_zebrafish()
    data = np.moveaxis(preprocess(ds.timeseries, cfg.preprocessing), 1, 2)     # (fish, T, N)
    observed["data"] = np.load(RESULTS_DIR / "zebrafish_data" / "omega_real.npy")
    jobs.append(("data", data, triplets, cfg.base_seed, k))

    keys = [f"{sc}__linear" for sc in ex["connectomes"]]
    for sc in ex["connectomes"]:
        for norm in ex["normalisations"]:
            for order in ex["orders"]:
                for form in ex["forms"]:
                    keys.append(f"{sc}__{norm}__{order}-{form}__at-{form}-optimum")
    for key in keys:
        d = np.load(FINAL / f"{key}.npz")
        observed[key] = d["omega"]
        sim = np.stack([preprocess(s.astype(float), cfg.preprocessing, axis=0) for s in d["signals"]])
        jobs.append((key, sim, triplets, cfg.base_seed, k))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        nulls = dict(pool.map(null_omega, jobs))
    arrays = {}
    for key, obs in observed.items():
        arrays[f"{key}__omega"] = obs
        arrays[f"{key}__null"] = nulls[key]
        arrays[f"{key}__delta"] = obs - nulls[key]
    arrays["triplets"] = triplets
    arrays["symmetric_mask"] = np.load(RESULTS_DIR / "zebrafish_data" / "symmetric_mask.npy")
    np.savez_compressed(OUT, **arrays)
    summary = {key: float(np.median(np.median(arrays[f"{key}__delta"], axis=1))) for key in observed}
    (OUT.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
