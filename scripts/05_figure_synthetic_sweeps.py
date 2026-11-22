"""Parameter-sweep figures for the synthetic networks.

Figure "combined_simulation_3panels": drive sweep. One block per network
(3 nodes, ring, random). Within a block, the upper row uses additive and the
lower row diffusive coupling; the left pair of maps uses globally normalised
and the right pair row-normalised connectomes; within each pair the left map
uses pairwise and the right map pairwise plus fan coupling.

Figure "supplementary_wc_aK_combined": response-gain sweep on the three
synthetic networks (global normalisation), additive and diffusive rows, pairwise and
pairwise-plus-fan columns.

Maps show the mean triplet O-information over noise replicates. Dashed lines
enclose the region in which the noise-free network oscillates (temporal
standard deviation of the excitatory activity above 1e-3 after the transient;
the mask is unchanged with a five-fold longer transient or a different initial
condition).

Usage:
    python scripts/05_figure_synthetic_sweeps.py
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
from coupling_hoi.networks import synthetic_networks  # noqa: E402
from coupling_hoi.plotting import (C_ADD, C_DIF, C_GROUP, STYLE, draw_network, heat_panel,  # noqa: E402
                                   horizontal_colourbar, smooth, symmetric_limit)

MAIN = RESULTS_DIR / "main"
FIG = RESULTS_DIR / "figures"
AMPLITUDE_THRESHOLD = 1e-3
TITLES = {"three_node": "3-node  all-to-all", "ring": "10-node  ring small-world",
          "random": "10-node  Erdős–Rényi random"}
ORDERS = [("pairwise", "2nd order"), ("pairwise+fan", "2nd + 3rd order")]


def load_map(path: Path, x_key: str):
    """Mean O-information and oscillation indicator on the parameter grid.

    Returns (omega, osc, x_values, y_values) with arrays of shape (len(y), len(x)).
    For the drive sweep x = coupling, y = drive; for the gain sweep x = gain,
    y = coupling.
    """
    d = np.load(path)
    if x_key == "coupling":
        xs, ys, xk, yk, dxk, dyk = np.unique(d["coupling"]), np.unique(d["drive"]), "coupling", "drive", \
            "det_coupling", "det_drive"
    else:
        xs, ys, xk, yk, dxk, dyk = np.unique(d["gain"]), np.unique(d["coupling"]), "gain", "coupling", \
            "det_gain", "det_coupling"
    omega_run = d["omega"].mean(axis=1)
    om = np.full((len(ys), len(xs)), np.nan)
    osc = np.zeros_like(om)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            om[i, j] = omega_run[(d[yk] == yv) & (d[xk] == xv)].mean()
            osc[i, j] = float(d["det_amplitude"][(d[dyk] == yv) & (d[dxk] == xv)][0] > AMPLITUDE_THRESHOLD)
    return om, osc, xs, ys


def block_figure(part: str, networks: list[str], groups: list[tuple[str, str]], x_key: str,
                 xlabel: str, ylabel: str, fname: str) -> None:
    """Blocks of maps; ``groups`` is a list of (normalisation, header). The layout is
    set in inches so that text keeps its size relative to the maps."""
    plt.rcParams.update(STYLE)
    n_groups = len(groups)
    # horizontal layout (inches)
    topo_w, lab_w, hm_w, gap_hm, gap_mid, margin = 2.6, 1.75, 2.35, 0.16, 0.85, 0.35
    fw = topo_w + lab_w + n_groups * (2 * hm_w + gap_hm) + (n_groups - 1) * gap_mid + margin
    # vertical layout (inches)
    head_h, title_h, row_h = 1.15, 0.62, 1.8
    cb_gap, cb_h, cb_lab = 0.16, 0.17, 0.62
    row_gap, xlab_h, blk_gap = 0.22, 0.78, 0.38
    blk_h = title_h + row_h + cb_gap + cb_h + cb_lab + row_gap + row_h + xlab_h + cb_h + cb_lab + blk_gap
    fh = head_h + len(networks) * blk_h
    fig = plt.figure(figsize=(fw, fh))

    def rect(x, y_top, w, h):
        """Axes rectangle from inches measured from the left and from the top."""
        return [x / fw, 1 - (y_top + h) / fh, w / fw, h / fh]

    x0 = topo_w + lab_w
    col_x = []
    for g in range(n_groups):
        base = x0 + g * (2 * hm_w + gap_hm + gap_mid)
        col_x.append((base, base + hm_w + gap_hm))

    fig.text((topo_w / 2 + 0.1) / fw, 1 - 0.12 / fh, "Topology", ha="center", va="top",
             fontsize=24, fontweight="bold", color="#2a2a2a")
    for g, (_, header) in enumerate(groups):
        a, b = col_x[g]
        fig.text((a + b + hm_w) / 2 / fw, 1 - 0.12 / fh, header, ha="center", va="top", fontsize=24,
                 fontweight="bold", color=C_GROUP[g % 2])
        for xc, (_, olabel) in zip((a, b), ORDERS):
            fig.text((xc + hm_w / 2) / fw, 1 - 0.64 / fh, olabel, ha="center", va="top",
                     fontsize=19, color="#444444", style="italic")

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    nets = synthetic_networks()
    for blk, name in enumerate(networks):
        y = head_h + blk * blk_h
        add_top = y + title_h
        cb_add_top = add_top + row_h + cb_gap
        dif_top = cb_add_top + cb_h + cb_lab + row_gap
        cb_dif_top = dif_top + row_h + xlab_h

        mid_rows = (add_top + dif_top + row_h) / 2
        ax_t = fig.add_axes(rect(0.1, mid_rows - topo_w / 2, topo_w, topo_w))
        draw_network(ax_t, nets[name].adjacency)
        for top_y, txt, col in ((add_top, "Additive", C_ADD), (dif_top, "Diffusive", C_DIF)):
            yc = 1 - (top_y + row_h / 2) / fh
            fig.text((topo_w + 0.3) / fw, yc, txt, ha="center", va="center", rotation="vertical",
                     fontsize=22, fontweight="bold", color=col)
            fig.text((topo_w + 0.78) / fw, yc, ylabel, ha="center", va="center", rotation="vertical",
                     fontsize=20, color="#333333")

        for g, (norm, _) in enumerate(groups):
            maps = {}
            for form in ("additive", "diffusive"):
                for order, _ in ORDERS:
                    maps[(form, order)] = load_map(MAIN / part / f"{name}__{norm}__{order}-{form}.npz", x_key)
            for form, top_y, cb_top in (("additive", add_top, cb_add_top), ("diffusive", dif_top, cb_dif_top)):
                sm = {o: smooth(maps[(form, o)][0]) for o, _ in ORDERS}
                vlim = symmetric_limit(*sm.values())
                im = None
                for c, (order, _) in enumerate(ORDERS):
                    om, osc, xs, ys = maps[(form, order)]
                    ax = fig.add_axes(rect(col_x[g][c], top_y, hm_w, row_h))
                    im = heat_panel(ax, sm[order], osc, xs, ys, vlim,
                                    xlabel=xlabel if form == "diffusive" else "",
                                    label=next(letters), show_x=(form == "diffusive"))
                    if not (g == 0 and c == 0):
                        ax.tick_params(labelleft=False)
                horizontal_colourbar(fig, im, rect(col_x[g][0], cb_top, 2 * hm_w + gap_hm, cb_h),
                                     vlim, label=r"mean triplet $\Omega_{\mathrm{KSG}}$ (nats)")
        mid = (col_x[0][0] + col_x[-1][1] + hm_w) / 2
        fig.text(mid / fw, 1 - (y + title_h * 0.45) / fh, TITLES[name], ha="center", va="center",
                 fontsize=24, fontweight="bold")

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)
    print("saved", FIG / fname)


def main() -> None:
    block_figure("synthetic_drive", ["three_node", "ring", "random"],
                 [("global", "Global normalisation"), ("row", "Row normalisation")],
                 x_key="coupling", xlabel="Coupling $K$", ylabel="Drive $P$",
                 fname="combined_simulation_3panels")
    block_figure("synthetic_gain", ["three_node", "ring", "random"], [("global", "Wilson–Cowan")],
                 x_key="gain", xlabel="Gain $g$", ylabel="Coupling $K$",
                 fname="supplementary_wc_aK_combined")


if __name__ == "__main__":
    main()
