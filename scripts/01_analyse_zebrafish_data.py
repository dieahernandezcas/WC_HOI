"""O-information of the zebrafish recordings and of their surrogates.

Outputs (results/zebrafish_data/):
    omega_real.npy, omega_covariance.npy, omega_phase.npy   (fish, triplets)
    triplets.npy, symmetric_mask.npy, fc_mean.npy
    summary.json

Usage:
    python scripts/01_analyse_zebrafish_data.py [--check-against DIR]

With ``--check-against`` the recomputed real-data values are compared with a
previously stored ``omega_real.npy`` and the script fails if they differ.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RESULTS_DIR, ExperimentConfig, stable_seed  # noqa: E402
from coupling_hoi.data import load_zebrafish, sample_triplets, symmetric_triplets  # noqa: E402
from coupling_hoi.information import omega_triplets  # noqa: E402
from coupling_hoi.metrics import functional_connectivity  # noqa: E402
from coupling_hoi.preprocessing import preprocess  # noqa: E402
from coupling_hoi.surrogates import covariance_preserving, phase_randomised  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "zebrafish_data.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-against", type=Path, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(CONFIG)
    out_dir = RESULTS_DIR / "zebrafish_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_zebrafish()
    triplets = sample_triplets(ds.n_regions, cfg.extra["n_triplets"], cfg.extra["triplet_seed"])
    sym = symmetric_triplets(ds.sc_directed, triplets)
    signals = preprocess(ds.timeseries, cfg.preprocessing)          # (fish, regions, time)
    k = cfg.information.k_neighbours

    t0 = time.time()
    omega = {"real": [], "covariance": [], "phase": []}
    fcs = []
    for f in range(ds.n_fish):
        x = signals[f].T                                               # (time, regions)
        fcs.append(functional_connectivity(x))
        omega["real"].append(omega_triplets(x, triplets, k))
        cov_s = covariance_preserving(x, stable_seed(cfg.base_seed, "covariance", f))
        omega["covariance"].append(omega_triplets(cov_s, triplets, k))
        ph_s = phase_randomised(x, stable_seed(cfg.base_seed, "phase", f))
        omega["phase"].append(omega_triplets(ph_s, triplets, k))
    omega = {key: np.asarray(v) for key, v in omega.items()}

    for key, arr in omega.items():
        np.save(out_dir / f"omega_{key}.npy", arr)
    np.save(out_dir / "triplets.npy", triplets)
    np.save(out_dir / "symmetric_mask.npy", sym)
    np.save(out_dir / "fc_mean.npy", np.mean(fcs, axis=0))

    summary = {
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "n_fish": ds.n_fish, "n_regions": ds.n_regions, "n_samples": ds.n_samples,
        "n_triplets": int(len(triplets)), "n_symmetric_triplets": int(sym.sum()),
        "median_of_fish_medians": {
            key: {"directed_group": float(np.median(np.median(arr[:, ~sym], axis=1))),
                  "undirected_group": float(np.median(np.median(arr[:, sym], axis=1)))}
            for key, arr in omega.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["median_of_fish_medians"], indent=2))

    if args.check_against is not None:
        ref = np.load(args.check_against / "omega_real.npy")
        diff = float(np.abs(ref - omega["real"]).max())
        print(f"max |difference| with stored values: {diff:.3e}")
        if diff > 1e-10:
            raise SystemExit("recomputed values differ from the stored reference")


if __name__ == "__main__":
    main()
