"""Zebrafish figure: recordings, null models and network models.

Layout (2 x 2):
  columns  Directed SC (triplets with at least one asymmetric link; models on
           the directed connectome) and Undirected SC (triplets whose links are
           symmetric; models on the undirected connectome).
  row 1    O-information of the recordings and of three null models
           (covariance-preserving, phase-randomised, circular shift), one dot
           per fish = median over the triplets of the column.
  row 2    Null-corrected O-information (each recording or simulation minus
           its own circular-shift surrogates) for the recordings, the linear
           model and the Wilson-Cowan model with pairwise (M2) or pairwise plus
           fan (M3) coupling, additive or diffusive, each at its own operating
           point. Percentages give the share of triplet values with Omega < 0.

Tests: recordings against null models, Wilcoxon signed-rank (paired by fish);
additive against diffusive, Mann-Whitney U (independent simulations).

Usage:
    python scripts/07_figure_zebrafish.py [--normalisation global|row]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde, mannwhitneyu, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RESULTS_DIR  # noqa: E402

DATA = RESULTS_DIR / "zebrafish_data"
NULLC = RESULTS_DIR / "main" / "zebrafish_null_corrected.npz"
FIG = RESULTS_DIR / "figures"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6})
C = dict(real="#2166AC", cov="#D6604D", phase="#4DAC26", shift="#8c6bb1", emp="#1a1a2e",
         m1="#e76f51", m2a="#1b7f73", m2d="#7fc8e8", m3a="#7b2fe0", m3d="#c9a0ff")
TITLES = {"directed": "Directed SC", "undirected": "Undirected SC"}
RNG = np.random.default_rng(1)


def per_fish(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.median(a[:, mask], axis=1)


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def violin(ax, x, d, colour, w=0.36):
    yg = np.linspace(d.min(), d.max(), 300)
    k = gaussian_kde(d, bw_method=0.35)(yg)
    k = k / k.max() * w
    ax.fill_betweenx(yg, x - k, x + k, color=colour, alpha=0.75, lw=0, zorder=3)
    q1, m, q3 = np.percentile(d, [25, 50, 75])
    ax.vlines(x, q1, q3, color="white", lw=2.4, zorder=6)
    ax.plot(x, m, "o", ms=3.2, mfc="white", mec=colour, mew=1.0, zorder=8)
    ax.scatter(x + RNG.uniform(-0.07, 0.07, len(d)), d, s=3, color=colour, alpha=0.5, lw=0, zorder=5)


def bracket(ax, x1, x2, y, txt, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, color="#333", zorder=9)
    ax.text((x1 + x2) / 2, y + h * 1.05, txt, ha="center", va="bottom", fontsize=6.5, color="#333")


def style(ax):
    ax.axhline(0, color="#888", lw=0.7, ls="--", zorder=1)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(bottom=False)


def make(normalisation: str) -> dict:
    nc = np.load(NULLC)
    sym = nc["symmetric_mask"]
    groups = {"directed": ~sym, "undirected": sym}
    raw = {"real": np.load(DATA / "omega_real.npy"), "cov": np.load(DATA / "omega_covariance.npy"),
           "phase": np.load(DATA / "omega_phase.npy"), "shift": nc["data__null"]}

    fig, axs = plt.subplots(2, 2, figsize=(6.2, 3.0), gridspec_kw=dict(hspace=0.75, wspace=0.34))
    numbers: dict = {}
    for ci, sc in enumerate(["directed", "undirected"]):
        m = groups[sc]
        # ---------------- row 1: recordings and null models (raw Omega) ----------------
        ax = axs[0, ci]
        series = [per_fish(raw[k], m) for k in ("real", "cov", "phase", "shift")]
        for x, (d, col) in enumerate(zip(series, (C["real"], C["cov"], C["phase"], C["shift"]))):
            violin(ax, x, d, col, w=0.34)
        style(ax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Real\ndata", "Covariance-\npreserving", "Phase-\nrandomised", "Circular\nshift"],
                           fontsize=5.6)
        ax.set_xlim(-0.6, 3.6)
        top = max(d.max() for d in series)
        h = 0.05 * (top - min(d.min() for d in series))
        pvals = [wilcoxon(series[0], series[j]).pvalue for j in (1, 2, 3)]
        for j, p in enumerate(pvals, start=1):
            bracket(ax, 0, j, top + (2 + 4 * (j - 1)) * h, stars(p), h)
        ax.set_ylim(ax.get_ylim()[0], top + 15 * h)
        ax.set_ylabel(r"Median $\Omega$ (nats)")
        ax.set_title(f"{TITLES[sc]} ({int(m.sum())} triplets)", fontsize=8)
        ax.text(-0.02, 1.16, "AB"[ci], transform=ax.transAxes, fontsize=10, fontweight="bold")
        numbers[f"{sc}_recordings"] = {
            "median_raw": {k: float(np.median(s)) for k, s in zip(("real", "cov", "phase", "shift"), series)},
            "p_real_vs": dict(zip(("cov", "phase", "shift"), map(float, pvals)))}

        # ---------------- row 2: null-corrected O-information ----------------
        ax = axs[1, ci]
        conds = [("Real\ndata", "data", C["emp"]),
                 ("M1\nlinear", f"{sc}__linear", C["m1"]),
                 ("M2\nadd.", f"{sc}__{normalisation}__pairwise-additive__at-additive-optimum", C["m2a"]),
                 ("M2\ndiff.", f"{sc}__{normalisation}__pairwise-diffusive__at-diffusive-optimum", C["m2d"]),
                 ("M3\nadd.", f"{sc}__{normalisation}__pairwise+fan-additive__at-additive-optimum", C["m3a"]),
                 ("M3\ndiff.", f"{sc}__{normalisation}__pairwise+fan-diffusive__at-diffusive-optimum", C["m3d"])]
        deltas = [per_fish(nc[f"{key}__delta"], m) for _, key, _ in conds]
        shares = [float(np.mean(nc[f"{key}__omega"][:, m] < 0)) for _, key, _ in conds]
        for x, (d, (_, _, col)) in enumerate(zip(deltas, conds)):
            violin(ax, x, d, col, w=0.38)
        style(ax)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([f"{n}\n{100 * s:.0f}%" for (n, _, _), s in zip(conds, shares)], fontsize=6)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        top = max(d.max() for d in deltas)
        h = 0.05 * (top - min(d.min() for d in deltas))
        p_m2 = mannwhitneyu(deltas[2], deltas[3]).pvalue
        p_m3 = mannwhitneyu(deltas[4], deltas[5]).pvalue
        bracket(ax, 2, 3, top + 2 * h, stars(p_m2), h)
        bracket(ax, 4, 5, top + 2 * h, stars(p_m3), h)
        ax.set_ylim(ax.get_ylim()[0], top + 7 * h)
        ax.set_ylabel(r"Median $\Delta\Omega$ (nats)")
        ax.text(-0.02, 1.10, "CD"[ci], transform=ax.transAxes, fontsize=10, fontweight="bold")
        numbers[f"{sc}_null_corrected"] = {
            n.replace("\n", " "): {"median_delta": float(np.median(d)), "share_omega_negative": s,
                                   "p_vs_data": (None if i == 0 else float(mannwhitneyu(deltas[0], d).pvalue)),
                                   "p_delta_vs_zero": float(wilcoxon(d).pvalue)}
            for i, ((n, _, _), d, s) in enumerate(zip(conds, deltas, shares))}
        numbers[f"{sc}_null_corrected"]["p_M2_add_vs_diff"] = float(p_m2)
        numbers[f"{sc}_null_corrected"]["p_M3_add_vs_diff"] = float(p_m3)

    FIG.mkdir(parents=True, exist_ok=True)
    name = "zf_combined_results" if normalisation == "global" else f"zf_combined_results_{normalisation}"
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (FIG / f"{name}_numbers.json").write_text(json.dumps(numbers, indent=1))
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normalisation", choices=("global", "row"), default=None)
    args = parser.parse_args()
    for norm in ([args.normalisation] if args.normalisation else ["global", "row"]):
        print(json.dumps(make(norm), indent=1))


if __name__ == "__main__":
    main()
