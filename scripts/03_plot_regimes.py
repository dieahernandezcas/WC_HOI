"""Figures summarising the dynamical-regime exploration.

For each network and normalisation, one figure with four columns (pairwise
additive, pairwise diffusive, fan additive, fan diffusive) and three rows:
mean functional connectivity, mean triplet O-information and, for the
zebrafish connectome, similarity to the empirical FC. Values are averaged
over noise replicates.

Usage:
    python scripts/03_plot_regimes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RESULTS_DIR  # noqa: E402

REGIMES = RESULTS_DIR / "regimes"
FIG_DIR = RESULTS_DIR / "figures"
SPECS = ["pairwise-additive", "pairwise-diffusive", "fan-additive", "fan-diffusive"]
TITLES = {"pairwise-additive": "Pairwise, additive", "pairwise-diffusive": "Pairwise, diffusive",
          "fan-additive": "Fan, additive", "fan-diffusive": "Fan, diffusive"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6})


def grid_mean(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    drives, couplings = np.unique(d["drive"]), np.unique(d["coupling"])
    values = d[field].mean(axis=1) if d[field].ndim == 2 else d[field]
    out = np.full((len(drives), len(couplings)), np.nan)
    for i, p in enumerate(drives):
        for j, k in enumerate(couplings):
            sel = (d["drive"] == p) & (d["coupling"] == k)
            out[i, j] = np.nanmean(values[sel])
    return out, drives, couplings


def draw(ax, z, drives, couplings, cmap, vmin, vmax):
    im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(couplings)))
    ax.set_xticklabels([f"{c:g}" for c in couplings], rotation=90)
    ax.set_yticks(range(len(drives)))
    ax.set_yticklabels([f"{p:.2f}" for p in drives])
    return im


def plot_network(network: str, norm: str) -> None:
    files = {s: REGIMES / f"{network}__{norm}__{s}.npz" for s in SPECS}
    if not all(f.exists() for f in files.values()):
        return
    rows = [("mean_fc", "Mean FC", "viridis", 0, 0.8),
            ("omega", "Mean triplet $\\Omega$ (nats)", "RdBu_r", None, None)]
    if network.startswith("zebrafish"):
        rows.append(("fc_similarity", "FC similarity to data", "magma", 0, 0.4))
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.2, 2.3 * len(rows)), squeeze=False,
                             constrained_layout=True)
    for r, (field, label, cmap, vmin, vmax) in enumerate(rows):
        maps = {s: grid_mean(files[s], field) for s in SPECS}
        if field == "omega":
            lim = np.nanmax([np.nanmax(np.abs(m[0])) for m in maps.values()])
            vmin, vmax = -lim, lim
        for c, s in enumerate(SPECS):
            z, drives, couplings = maps[s]
            im = draw(axes[r, c], z, drives, couplings, cmap, vmin, vmax)
            if r == 0:
                axes[r, c].set_title(TITLES[s])
            axes[r, c].set_xlabel("Coupling $K$")
            if c == 0:
                axes[r, c].set_ylabel("Drive $P$")
            else:
                axes[r, c].set_yticklabels([])
        fig.colorbar(im, ax=axes[r, :], shrink=0.9, label=label)
    fig.suptitle(f"{network.replace('_', ' ')}  |  {norm} normalisation", fontsize=9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"regimes_{network}_{norm}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    for network in ["three_node", "ring", "random", "zebrafish_undirected"]:
        for norm in ["global", "row"]:
            plot_network(network, norm)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
