"""Map the dynamical regimes of the Wilson-Cowan network model.

For every network, normalisation scheme and coupling specification, the model
is simulated on a (drive, coupling) grid. Each run is summarised by its mean
activity, fluctuation amplitude, mean functional connectivity, similarity to
the empirical zebrafish FC (zebrafish connectome only) and triplet
O-information. A deterministic scan of an isolated node locates the drive
values at which a single node oscillates.

Outputs (results/regimes/):
    isolated_node.npz
    <network>__<normalisation>__<order>-<form>.npz
    config.json

Usage:
    python scripts/02_explore_dynamical_regimes.py [--networks three_node ring ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RESULTS_DIR, CouplingSpec, ExperimentConfig  # noqa: E402
from coupling_hoi.coupling import CouplingOperator  # noqa: E402
from coupling_hoi.data import load_zebrafish, sample_triplets  # noqa: E402
from coupling_hoi.experiment import grid_points, run_condition  # noqa: E402
from coupling_hoi.metrics import functional_connectivity  # noqa: E402
from coupling_hoi.model import simulate  # noqa: E402
from coupling_hoi.networks import synthetic_networks  # noqa: E402
from coupling_hoi.preprocessing import preprocess  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "regime_exploration.json"


def isolated_node_scan(cfg: ExperimentConfig, out_dir: Path) -> None:
    """Deterministic response of a single uncoupled node to the drive."""
    drives = np.round(np.arange(0.0, 1.0001, 0.01), 3)
    params = replace(cfg.model, sigma=0.0)
    op = CouplingOperator.from_adjacency(np.zeros((1, 1)))
    x = simulate(op, CouplingSpec("additive", "pairwise"), drives, np.zeros_like(drives),
                 list(range(len(drives))), params, cfg.integration)[:, :, 0]
    np.savez(out_dir / "isolated_node.npz", drive=drives, mean=x.mean(1), std=x.std(1),
             minimum=x.min(1), maximum=x.max(1))
    osc = drives[x.std(1) > 1e-6]
    print(f"isolated node oscillates for drive in [{osc.min():.2f}, {osc.max():.2f}]"
          if len(osc) else "isolated node never oscillates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--networks", nargs="*", default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(CONFIG)
    ex = cfg.extra
    out_dir = RESULTS_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    isolated_node_scan(cfg, out_dir)

    networks = {name: net.adjacency for name, net in synthetic_networks().items()}
    triplet_sets = {name: np.array(list(combinations(range(a.shape[0]), 3)))
                    for name, a in networks.items()}
    fc_refs: dict[str, np.ndarray | None] = {name: None for name in networks}
    if "zebrafish_undirected" in ex["networks"]:
        ds = load_zebrafish()
        networks["zebrafish_undirected"] = ds.sc_undirected
        triplet_sets["zebrafish_undirected"] = sample_triplets(
            ds.n_regions, ex["n_zebrafish_triplets"], ex["triplet_seed"])
        fc_refs["zebrafish_undirected"] = np.mean(
            [functional_connectivity(x.T) for x in preprocess(ds.timeseries, cfg.preprocessing)], axis=0)

    selected = args.networks or ex["networks"]
    for name in selected:
        for norm in ex["normalisations"]:
            for order in ex["orders"]:
                points = grid_points(ex["drives"], ex["couplings"][order], ex["n_replicates"])
                for form in ex["forms"]:
                    spec = CouplingSpec(form, order)
                    path = out_dir / f"{name}__{norm}__{spec.label}.npz"
                    if path.exists():
                        continue
                    t0 = time.time()
                    res = run_condition(networks[name], name, norm, spec, points, triplet_sets[name],
                                        cfg.base_seed, cfg.model, cfg.integration, cfg.preprocessing,
                                        cfg.information, fc_refs[name])
                    np.savez_compressed(
                        path, drive=[p.drive for p in points], coupling=[p.coupling for p in points],
                        replicate=[p.replicate for p in points], triplets=triplet_sets[name], **res)
                    print(f"{path.name}: {len(points)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
