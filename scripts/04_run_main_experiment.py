"""Main simulation experiment.

Parts (run in this order; each is resumable and skips finished conditions):

  synthetic_drive   three synthetic networks, drive x coupling grid
  synthetic_gain    ten-node networks, response gain x coupling grid
  zebrafish_scan    zebrafish connectome, coarse drive x coupling grid
  zebrafish_select  choose, for every condition, the operating point whose
                    functional connectivity best matches the recordings
  zebrafish_final   36 simulations per form at the selected operating points,
                    plus the linear reference model

Each condition is written to its own compressed file under
results/main/<part>/, with every per-run summary (triplet O-information,
functional connectivity, activity statistics, noise seed). The raw simulated
signals of the final zebrafish simulations are stored as well; any other run
can be regenerated exactly from its stored seed and parameters. Conditions are independent and can be computed in
parallel. The selection is based on functional connectivity only, never on
the O-information, so that the O-information of the selected models is an
independent comparison with the data. Final simulations use replicate indices
that do not overlap with those of the scan.

Usage:
    python scripts/04_run_main_experiment.py --part synthetic_drive [--workers 4] [--max-tasks 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# One BLAS thread per process: conditions are parallelised across processes.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RESULTS_DIR, CouplingSpec, ExperimentConfig, stable_seed  # noqa: E402
from coupling_hoi.data import load_zebrafish, sample_triplets  # noqa: E402
from coupling_hoi.experiment import (GridPoint, deterministic_amplitude, grid_points,  # noqa: E402
                                     run_condition)
from coupling_hoi.information import omega_triplets  # noqa: E402
from coupling_hoi.metrics import fc_similarity, functional_connectivity  # noqa: E402
from coupling_hoi.model import simulate_linear  # noqa: E402
from coupling_hoi.networks import synthetic_networks  # noqa: E402
from coupling_hoi.preprocessing import preprocess  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "main_experiment.json"
OUT = RESULTS_DIR / "main"
FINAL_REPLICATE_OFFSET = 1000
PARTS = ("synthetic_drive", "synthetic_gain", "zebrafish_scan", "zebrafish_select", "zebrafish_final")


def _range(spec) -> np.ndarray:
    lo, hi, n = spec
    return np.round(np.linspace(lo, hi, int(n)), 6)


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _config() -> ExperimentConfig:
    return ExperimentConfig.from_json(CONFIG)


# --------------------------------------------------------------------------- #
# Zebrafish inputs (cached per process)
# --------------------------------------------------------------------------- #

_ZF_CACHE: dict = {}


def _zebrafish_inputs(cfg: ExperimentConfig):
    if not _ZF_CACHE:
        ex = cfg.extra["zebrafish"]
        ds = load_zebrafish()
        triplets = sample_triplets(ds.n_regions, ex["n_triplets"], ex["triplet_seed"])
        signals = preprocess(ds.timeseries, cfg.preprocessing)
        fc_data = np.mean([functional_connectivity(x.T) for x in signals], axis=0)
        _ZF_CACHE.update(triplets=triplets, fc_data=fc_data,
                         connectomes={"undirected": ds.sc_undirected, "directed": ds.sc_directed})
    return _ZF_CACHE["triplets"], _ZF_CACHE["fc_data"], _ZF_CACHE["connectomes"]


# --------------------------------------------------------------------------- #
# Tasks: each task computes one condition and writes one file
# --------------------------------------------------------------------------- #

def task_synthetic(part: str, name: str, norm: str, order: str, form: str, path: str) -> str:
    cfg = _config()
    ex = cfg.extra[part]
    adjacency = synthetic_networks()[name].adjacency
    triplets = np.array(list(combinations(range(adjacency.shape[0]), 3)))
    couplings = _range(ex["coupling_range"])
    if part == "synthetic_drive":
        drives, gains = _range(ex["drive_range"]), np.array([1.0])
    else:
        drives, gains = np.array([ex["drive"]]), _range(ex["gain_range"])
    spec = CouplingSpec(form, order, cfg.extra["fan_ratio"])
    t0 = time.time()
    pts = grid_points(drives, couplings, ex["n_replicates"], gains)
    res = run_condition(adjacency, name, norm, spec, pts, triplets, cfg.base_seed,
                        cfg.model, cfg.integration, cfg.preprocessing, cfg.information)
    det = grid_points(drives, couplings, 1, gains)
    amp = deterministic_amplitude(adjacency, norm, spec, det, cfg.model, cfg.integration)
    _save(Path(path), drive=[p.drive for p in pts], coupling=[p.coupling for p in pts],
          gain=[p.gain for p in pts], replicate=[p.replicate for p in pts], triplets=triplets,
          det_drive=[p.drive for p in det], det_coupling=[p.coupling for p in det],
          det_gain=[p.gain for p in det], det_amplitude=amp, **res)
    return f"{Path(path).name}: {len(pts)} runs in {time.time() - t0:.0f}s"


def task_zebrafish_scan(sc: str, norm: str, order: str, form: str, path: str) -> str:
    cfg = _config()
    ex = cfg.extra["zebrafish"]
    triplets, fc_data, connectomes = _zebrafish_inputs(cfg)
    pts = grid_points(ex["scan_drives"], ex["scan_couplings"], ex["scan_replicates"])
    spec = CouplingSpec(form, order, cfg.extra["fan_ratio"])
    t0 = time.time()
    res = run_condition(connectomes[sc], f"zebrafish_{sc}", norm, spec, pts, triplets, cfg.base_seed,
                        cfg.model, cfg.integration, cfg.preprocessing, cfg.information,
                        fc_reference=fc_data)
    _save(Path(path), drive=[p.drive for p in pts], coupling=[p.coupling for p in pts],
          replicate=[p.replicate for p in pts], triplets=triplets, **res)
    return f"{Path(path).name}: {len(pts)} runs in {time.time() - t0:.0f}s"


def task_zebrafish_final(sc: str, norm: str, order: str, form: str, drive: float, coupling: float,
                         path: str) -> str:
    cfg = _config()
    ex = cfg.extra["zebrafish"]
    triplets, fc_data, connectomes = _zebrafish_inputs(cfg)
    reps = range(FINAL_REPLICATE_OFFSET, FINAL_REPLICATE_OFFSET + ex["final_replicates"])
    pts = [GridPoint(drive, coupling, r) for r in reps]
    spec = CouplingSpec(form, order, cfg.extra["fan_ratio"])
    t0 = time.time()
    res = run_condition(connectomes[sc], f"zebrafish_{sc}", norm, spec, pts, triplets, cfg.base_seed,
                        cfg.model, cfg.integration, cfg.preprocessing, cfg.information,
                        fc_reference=fc_data, keep_signals=True)
    _save(Path(path), drive=drive, coupling=coupling, **res)
    return f"{Path(path).name}: {len(pts)} runs in {time.time() - t0:.0f}s"


def task_zebrafish_linear(sc: str, path: str) -> str:
    cfg = _config()
    ex = cfg.extra["zebrafish"]
    lin = ex["linear_model"]
    triplets, fc_data, connectomes = _zebrafish_inputs(cfg)
    reps = range(FINAL_REPLICATE_OFFSET, FINAL_REPLICATE_OFFSET + ex["final_replicates"])
    seeds = [stable_seed(cfg.base_seed, "linear", sc, r) for r in reps]
    raw = simulate_linear(connectomes[sc], seeds, cfg.integration.n_records,
                          lin["n_burn"], lin["lam"], lin["sigma"])
    om, fcs = [], []
    for x in raw:
        y = preprocess(x, cfg.preprocessing, axis=0)
        om.append(omega_triplets(y, triplets, cfg.information.k_neighbours))
        fcs.append(fc_similarity(functional_connectivity(y), fc_data))
    _save(Path(path), omega=np.asarray(om), fc_similarity=np.asarray(fcs), seeds=np.asarray(seeds),
          signals=raw.astype(np.float32))
    return f"{Path(path).name}"


# --------------------------------------------------------------------------- #
# Task lists
# --------------------------------------------------------------------------- #

def pending_tasks(cfg: ExperimentConfig, part: str) -> list[tuple]:
    tasks = []
    if part in ("synthetic_drive", "synthetic_gain"):
        ex = cfg.extra[part]
        for name in ex["networks"]:
            for norm in ex["normalisations"]:
                for order in ex["orders"]:
                    for form in ex["forms"]:
                        label = CouplingSpec(form, order).label
                        path = OUT / part / f"{name}__{norm}__{label}.npz"
                        if not path.exists():
                            tasks.append((task_synthetic, (part, name, norm, order, form, str(path))))
    elif part == "zebrafish_scan":
        ex = cfg.extra["zebrafish"]
        for sc in ex["connectomes"]:
            for norm in ex["normalisations"]:
                for order in ex["orders"]:
                    for form in ex["forms"]:
                        label = CouplingSpec(form, order).label
                        path = OUT / "zebrafish_scan" / f"{sc}__{norm}__{label}.npz"
                        if not path.exists():
                            tasks.append((task_zebrafish_scan, (sc, norm, order, form, str(path))))
    elif part == "zebrafish_final":
        ex = cfg.extra["zebrafish"]
        selection = json.loads((OUT / "zebrafish_selection.json").read_text())
        for sc in ex["connectomes"]:
            path = OUT / "zebrafish_final" / f"{sc}__linear.npz"
            if not path.exists():
                tasks.append((task_zebrafish_linear, (sc, str(path))))
        for sc in ex["connectomes"]:
            for norm in ex["normalisations"]:
                for order in ex["orders"]:
                    for at_form in ex["forms"]:
                        op = selection[f"{sc}__{norm}__{order}-{at_form}"]
                        for form in ex["forms"]:
                            label = CouplingSpec(form, order).label
                            path = OUT / "zebrafish_final" / f"{sc}__{norm}__{label}__at-{at_form}-optimum.npz"
                            if not path.exists():
                                tasks.append((task_zebrafish_final, (sc, norm, order, form, op["drive"],
                                                                     op["coupling"], str(path))))
    return tasks


def select_operating_points(cfg: ExperimentConfig) -> None:
    """Operating point with the highest mean FC similarity, per condition."""
    metric = cfg.extra["zebrafish"]["selection_metric"]
    selection = {}
    for path in sorted((OUT / "zebrafish_scan").glob("*.npz")):
        d = np.load(path)
        keys = sorted(set(zip(d["drive"].tolist(), d["coupling"].tolist())))
        scores = []
        for p, k in keys:
            sel = (d["drive"] == p) & (d["coupling"] == k)
            scores.append((float(np.nanmean(d[metric][sel])), p, k,
                           float(np.mean(d["mean_fc"][sel])), float(np.mean(d["omega"][sel]))))
        best = max(scores, key=lambda s: s[0])
        selection[path.stem] = {"drive": best[1], "coupling": best[2], metric: best[0],
                                "mean_fc": best[3], "mean_omega": best[4]}
    out = OUT / "zebrafish_selection.json"
    out.write_text(json.dumps(selection, indent=2))
    _log(f"selection of {len(selection)} conditions written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="compute at most this many conditions in this call")
    args = parser.parse_args()

    cfg = _config()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    if args.part == "zebrafish_select":
        select_operating_points(cfg)
        return

    tasks = pending_tasks(cfg, args.part)
    _log(f"{args.part}: {len(tasks)} conditions pending")
    tasks = tasks[:args.max_tasks] if args.max_tasks else tasks
    if args.workers <= 1:
        for fn, a in tasks:
            _log(fn(*a))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for msg in pool.map(_call, tasks):
                _log(msg)


def _call(task):
    fn, a = task
    return fn(*a)


if __name__ == "__main__":
    main()
