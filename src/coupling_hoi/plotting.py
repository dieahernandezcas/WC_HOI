"""Shared figure style for parameter-sweep maps.

Panels show a quantity over a two-parameter grid with a diverging colour map
centred at zero, a dashed contour marking the region in which the noise-free
network oscillates, and the network drawn on the left of each block.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import gaussian_filter

CMAP = "RdBu_r"
SMOOTH_SIGMA = 1.0          # display smoothing, in grid points
PERCENTILE_LIMIT = 95       # colour limits from this percentile of |values|
OSC_LEVEL = 0.5
C_GROUP = ("#1a3a6e", "#7a1010")
C_ADD = "#1e5225"
C_DIF = "#5a3a10"
C_SPINE = "#9aabb8"
NODE_COLOUR = "#1a3a6e"
EDGE_COLOUR = "#6a7a8a"

STYLE = {
    "font.family": "sans-serif", "font.size": 20.0, "axes.labelsize": 20.0,
    "xtick.labelsize": 18.0, "ytick.labelsize": 18.0, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
}


def smooth(a: np.ndarray) -> np.ndarray:
    return gaussian_filter(np.asarray(a, dtype=float), sigma=SMOOTH_SIGMA)


def symmetric_limit(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrays])
    return float(np.percentile(values, PERCENTILE_LIMIT)) or 1e-12


def fmt_tick(v: float) -> str:
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def heat_panel(ax, data, osc, x, y, vlim, xlabel="", ylabel="", label=None, show_x=True):
    """One map; ``data`` and ``osc`` have shape (len(y), len(x))."""
    im = ax.imshow(data, origin="lower", aspect="auto", extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=CMAP, vmin=-vlim, vmax=vlim, interpolation="bilinear")
    if osc is not None and np.any(osc > OSC_LEVEL) and np.any(osc < OSC_LEVEL):
        ax.contour(np.linspace(x[0], x[-1], osc.shape[1]), np.linspace(y[0], y[-1], osc.shape[0]),
                   osc, levels=[OSC_LEVEL], colors="k", linewidths=1.5, linestyles="--")
    ax.set_xlabel(xlabel, fontsize=20.0, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20.0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
    ax.tick_params(labelsize=18.0, length=4, width=1.0, pad=3)
    if not show_x:
        ax.tick_params(bottom=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_color(C_SPINE)
        s.set_linewidth(1.0)
    if label:
        ax.text(0.04, 0.96, f"({label})", transform=ax.transAxes, fontsize=17.0,
                fontweight="bold", va="top", ha="left", color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    return im


def horizontal_colourbar(fig, im, rect, vlim, label=r"$\Omega_{\mathrm{KSG}}$ (nats)"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = [-vlim, 0.0, vlim]
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt_tick(t) for t in ticks], fontsize=15.0)
    cb.ax.tick_params(length=4.0, pad=3, bottom=True, top=False, labelbottom=True, labeltop=False)
    cb.set_label(label, fontsize=17.0, labelpad=4)
    cb.ax.xaxis.set_label_position("bottom")
    cb.outline.set_edgecolor(C_SPINE)
    cb.outline.set_linewidth(0.8)
    return cb


def draw_network(ax, adjacency: np.ndarray) -> None:
    """Nodes on a circle, links as straight lines."""
    n = adjacency.shape[0]
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    radius = 1.0 if n <= 3 else 1.25
    node_r = 0.3 if n <= 3 else 0.23
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pts = np.c_[np.cos(ang), np.sin(ang)] * radius
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] or adjacency[j, i]:
                ax.plot(*pts[[i, j]].T, color=EDGE_COLOUR, lw=2.2, zorder=1)
    for k, p in enumerate(pts):
        ax.add_patch(plt.Circle(p, node_r, color=NODE_COLOUR, zorder=2, lw=0))
        ax.add_patch(plt.Circle(p, node_r, color="none", ec="white", lw=1.2, zorder=3))
        ax.text(*p, str(k), ha="center", va="center", fontsize=24 if n <= 3 else 15,
                color="white", fontweight="bold", zorder=4)
