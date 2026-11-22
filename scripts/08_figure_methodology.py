"""Overview figure of the study (three stacked panels).

(A) The question: within the brain network, a region involved in imagining
    and a region involved in speaking exchange information; their signals are
    correlated. Does the O-information reveal how such regions interact?
(B) Modelling the zebrafish brain: the structural connectome, the
    Wilson-Cowan node and the four coupling functions.
(C) Synthetic networks: the same model and couplings on small networks with
    controlled topology, used to understand the zebrafish results.

Every panel is drawn in physical units (inches) with equal aspect, so that
circles and curves keep their shape.

Usage:
    python scripts/08_figure_methodology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402
from scipy.sparse.csgraph import connected_components, shortest_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coupling_hoi.config import RAW_ZEBRAFISH_DIR, RESULTS_DIR  # noqa: E402
from coupling_hoi.data import load_zebrafish  # noqa: E402
from coupling_hoi.networks import synthetic_networks  # noqa: E402

FIG = RESULTS_DIR / "figures"

# palette (additive and diffusive colours match the result figures)
INK = "#1f2a37"
NAVY = "#1d3557"
MUTED = "#6b7a8c"
CARD_EDGE = "#d7dee7"
SHADOW = "#e9edf2"
C_ADD, C_ADD_BG = "#1e5225", "#eef6f0"
C_DIF, C_DIF_BG = "#5a3a10", "#f7f1e9"
C_TRI1, C_TRI2 = "#e76f51", "#2a6f97"
C_E, C_I = "#c0392b", "#2c6fbb"
QUESTION_BG, QUESTION_EDGE, QUESTION_INK = "#fff6e5", "#f0c27b", "#6b4a12"

FIG_W = 7.4
PANEL_H = (3.45, 4.15, 4.08)
GAP = 0.12

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "mathtext.fontset": "dejavusans"})


# --------------------------------------------------------------------------- #
# Drawing helpers (all coordinates in inches within a panel)
# --------------------------------------------------------------------------- #

def card(ax, w, h, letter, title):
    ax.add_patch(FancyBboxPatch((0.07, 0.03), w - 0.1, h - 0.08,
                                boxstyle="round,pad=0,rounding_size=0.16", fc=SHADOW, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((0.04, 0.06), w - 0.1, h - 0.08,
                                boxstyle="round,pad=0,rounding_size=0.16", fc="white", ec=CARD_EDGE,
                                lw=0.9, zorder=0.5))
    ax.text(0.2, h - 0.2, letter, fontsize=15, fontweight="bold", color=INK, va="top")
    ax.text(0.5, h - 0.215, title, fontsize=11, fontweight="bold", color=NAVY, va="top")


def subtitle(ax, x, y, tag, text, fs=8.4):
    """Bold italic sub-panel title, e.g. 'A1  Brain network'."""
    ax.text(x, y, f"{tag}  {text}", ha="center", va="center", fontsize=fs, fontweight="bold",
            style="italic", color=NAVY)


def form_label(ax, x, y, text, colour, fs=8.4):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=colour)


SUB_DY = 0.66  # distance of sub-panel titles below the top of a panel

# Every panel is split into three columns of equal width; arrows sit in the gaps.
COLS = ((0.16, 2.34), (2.62, 4.8), (5.08, 7.26))
CX = tuple((a + b) / 2 for a, b in COLS)
GAP_X = ((COLS[0][1] + COLS[1][0]) / 2, (COLS[1][1] + COLS[2][0]) / 2)


def column_tiles(ax, h, bottom=0.52):
    """Light background tiles marking the three sub-panels."""
    for a, b in COLS:
        ax.add_patch(FancyBboxPatch((a, bottom), b - a, h - 0.44 - bottom,
                                    boxstyle="round,pad=0,rounding_size=0.1", fc="#f8fafc",
                                    ec="#e4e9f0", lw=0.7, zorder=0.6))


def gap_arrow(ax, k, y):
    g = GAP_X[k]
    arrow(ax, (g - 0.11, y), (g + 0.13, y), lw=1.6, z=6)


def caption(ax, x, y, text, **kw):
    kw.setdefault("fontsize", 8)
    ax.text(x, y, text, ha="center", va="center", color=MUTED, style="italic", **kw)


def arrow(ax, p, q, colour=MUTED, rad=0.0, lw=1.4, style="-|>", ms=10, z=3, ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                                 mutation_scale=ms, lw=lw, color=colour, zorder=z, linestyle=ls,
                                 alpha=alpha, capstyle="round", joinstyle="round"))


def link(ax, p, q, colour, rad=0.0, lw=1.2, alpha=1.0, z=2):
    return ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                 lw=lw, color=colour, alpha=alpha, zorder=z, capstyle="round"))


def node(ax, xy, r, colour, label=None, fs=8, z=5, halo=True):
    if halo:
        ax.add_patch(mp.Circle(xy, r * 1.75, fc=colour, ec="none", alpha=0.14, zorder=z - 1))
    ax.add_patch(mp.Circle(xy, r, fc=colour, ec="white", lw=1.3, zorder=z))
    if label:
        ax.text(*xy, label, ha="center", va="center", color="white", fontsize=fs,
                fontweight="bold", zorder=z + 1)


def pill(ax, x, y, text, colour, bg, fs=8.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=colour,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.5", fc=bg, ec=colour, lw=0.8), zorder=6)


def smooth_trace(ax, x0, y0, w, h, phase, colour, freq=2.5, seed=0, lw=1.6, z=4):
    t = np.linspace(0, 1, 400)
    rng = np.random.default_rng(seed)
    wobble = gaussian_filter(rng.standard_normal(400), 18) * 6
    y = np.sin(2 * np.pi * freq * t + phase) + 0.35 * np.sin(2 * np.pi * 2.3 * freq * t + 2 * phase) + wobble
    y = (y - y.min()) / (y.max() - y.min())
    ax.plot(x0 + w * t, y0 + h * y, color=colour, lw=lw, zorder=z, solid_capstyle="round")


def rounded_polygon(pts, frac=0.3, n=14):
    """Vertices of a polygon whose corners are replaced by smooth quadratic arcs;
    `frac` is the fraction of each edge used by the rounding at either end."""
    pts = np.asarray(pts, dtype=float)
    out = []
    m = len(pts)
    t = np.linspace(0, 1, n)[:, None]
    for k in range(m):
        prev, cur, nxt = pts[k - 1], pts[k], pts[(k + 1) % m]
        a = cur + (prev - cur) * frac
        b = cur + (nxt - cur) * frac
        out.append((1 - t) ** 2 * a + 2 * (1 - t) * t * cur + t ** 2 * b)
    return np.vstack(out)


def bowed_triangle(ax, pts, colour):
    """Triangle with slightly bowed edges and a translucent fill."""
    centre = pts.mean(axis=0)
    ax.add_patch(mp.Polygon(pts, closed=True, fc=colour, ec="none", alpha=0.13, zorder=1.5))
    for i in range(3):
        a, b = pts[i], pts[(i + 1) % 3]
        cross = (b - a)[0] * (centre - a)[1] - (b - a)[1] * (centre - a)[0]
        rad = 0.18 if cross > 0 else -0.18
        link(ax, a, b, colour, rad=-rad, lw=1.8, z=2)


# --------------------------------------------------------------------------- #
# Panel A: the question
# --------------------------------------------------------------------------- #

def _brain_outline(origin, scale):
    """Smooth closed outline of a stylised human cerebrum (lateral view, frontal pole left)."""
    from scipy.interpolate import splev, splprep
    pts = np.array([(0.15, 0.95), (0.10, 1.32), (0.30, 1.76), (0.75, 2.06), (1.35, 2.22), (2.00, 2.20),
                    (2.60, 2.02), (3.05, 1.66), (3.30, 1.20), (3.24, 0.82), (2.95, 0.62), (2.50, 0.66),
                    (2.10, 0.52), (1.60, 0.44), (1.10, 0.52), (0.82, 0.72), (0.55, 0.74), (0.30, 0.80)])
    pts = np.vstack([pts, pts[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, 400)
    x, y = splev(u, tck)
    return np.c_[x, y] * scale + origin


def _smooth_curve(points, n=100):
    from scipy.interpolate import splev, splprep
    p = np.asarray(points)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0, k=min(3, len(p) - 1))
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.c_[x, y]


def draw_brain(ax, origin, scale, sulci=True):
    """Cerebrum, cerebellum and brainstem with sulci; returns the outline path, the
    coordinate transform and the cerebrum patch (used for clipping)."""
    from matplotlib.path import Path as MPath
    o = np.asarray(origin)
    tf = lambda p: np.asarray(p) * scale + o  # noqa: E731
    # brainstem and cerebellum behind the cerebrum
    stem = _smooth_curve([(2.15, 0.62), (2.22, 0.30), (2.30, 0.02), (2.45, 0.02), (2.48, 0.32), (2.55, 0.60)])
    ax.add_patch(mp.Polygon(tf(stem), closed=True, fc="#e9d7cf", ec="#c9aea3", lw=0.9, zorder=1))
    ax.add_patch(mp.Ellipse(tf((2.78, 0.48)), 0.86 * scale, 0.5 * scale, angle=-8, fc="#ecdcd4",
                            ec="#c9aea3", lw=0.9, zorder=1.1))
    for k in range(4):
        cb = _smooth_curve([(2.42, 0.42 + 0.07 * k), (2.75, 0.36 + 0.08 * k), (3.12, 0.44 + 0.06 * k)], 40)
        ax.plot(*tf(cb).T, color="#cfb3a8", lw=0.7, zorder=1.2)
    outline = _brain_outline(o, scale)
    cerebrum = mp.Polygon(outline, closed=True, fc="#f6ece8", ec="#b89488", lw=1.3, zorder=1.5)
    ax.add_patch(cerebrum)
    if sulci:
        # sulci: gently wavy lines clipped to the cerebrum
        rng = np.random.default_rng(3)
        paths = [[(0.4, 1.9), (0.7, 1.6), (0.65, 1.3), (0.9, 1.1)],
                 [(1.25, 2.15), (1.2, 1.8), (1.45, 1.5), (1.35, 1.2)],          # central sulcus
                 [(1.75, 2.15), (1.8, 1.85), (1.65, 1.6)],
                 [(2.3, 2.05), (2.2, 1.8), (2.45, 1.55), (2.4, 1.3)],
                 [(2.85, 1.75), (2.7, 1.5), (2.95, 1.2)],
                 [(0.25, 1.35), (0.5, 1.45), (0.75, 1.75)],
                 [(1.95, 1.4), (2.15, 1.2), (2.6, 1.1), (3.05, 1.0)],
                 [(1.25, 0.72), (1.6, 0.68), (2.0, 0.72), (2.4, 0.82)]]
        for path in paths:
            c = _smooth_curve(path, 60)
            line, = ax.plot(*tf(c).T, color="#d4b5aa", lw=1.0, zorder=1.6, solid_capstyle="round")
            line.set_clip_path(cerebrum)
        sylvian = _smooth_curve([(0.82, 0.86), (1.3, 0.98), (1.8, 1.08), (2.3, 1.2)], 80)
        line, = ax.plot(*tf(sylvian).T, color="#c39b8f", lw=1.6, zorder=1.65, solid_capstyle="round")
        line.set_clip_path(cerebrum)
    return MPath(outline), tf, cerebrum


def panel_question(ax, w, h):
    scale = 0.62
    origin = np.array([CX[0] - 1.7 * scale, 1.0])
    brain_path, tf, cerebrum = draw_brain(ax, origin, scale, sulci=False)
    outline = brain_path.vertices

    # a connected complex network strictly inside the cerebrum
    rng = np.random.default_rng(11)
    cand = np.c_[rng.uniform(0.2, 3.3, 1500), rng.uniform(0.45, 2.2, 1500)] * scale + origin
    clearance = np.min(np.linalg.norm(cand[:, None] - outline[None], axis=2), axis=1)
    cand = cand[brain_path.contains_points(cand) & (clearance > 0.1)]
    pts = [cand[0]]
    for q in cand[1:]:  # keep nodes evenly spread
        if np.min(np.linalg.norm(np.array(pts) - q, axis=1)) > 0.145:
            pts.append(q)
    pts = np.array(pts)
    n = len(pts)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    adj = np.zeros((n, n), bool)
    for i in range(n):
        adj[i, np.argsort(dist[i])[1:4]] = True
    adj |= adj.T
    _, comp = connected_components(adj, directed=False)
    while comp.max() > 0:  # join components through their closest pair of nodes
        d = np.where(comp[:, None] != comp[None, :], dist, np.inf)
        i, j = np.unravel_index(np.argmin(np.where(comp[:, None] == 0, d, np.inf)), d.shape)
        adj[i, j] = adj[j, i] = True
        _, comp = connected_components(adj, directed=False)
    edge_rad = 0.12
    for i, j in zip(*np.nonzero(np.triu(adj))):
        link(ax, pts[i], pts[j], "#9aa5b4", rad=edge_rad, lw=0.6, alpha=0.6, z=2).set_clip_path(cerebrum)

    # two triplets of regions (not necessarily linked to each other directly)
    def triplet_near(centre):
        # three nearby nodes, spread enough to read as a group of distinct regions
        order = np.argsort(np.linalg.norm(pts - centre, axis=1))[:9]
        best, area = None, -1.0
        for a_ in range(len(order)):
            for b_ in range(a_ + 1, len(order)):
                for c_ in range(b_ + 1, len(order)):
                    p, q, r = pts[order[[a_, b_, c_]]]
                    ar = abs((q - p)[0] * (r - p)[1] - (q - p)[1] * (r - p)[0]) / 2
                    if ar > area and max(np.linalg.norm(q - p), np.linalg.norm(r - p), np.linalg.norm(r - q)) < 0.42:
                        best, area = order[[a_, b_, c_]], ar
        return best
    tri_img, tri_spk = triplet_near(tf((2.35, 1.62))), triplet_near(tf((1.0, 1.12)))
    highlighted = set(tri_img) | set(tri_spk)
    others = [i for i in range(n) if i not in highlighted]
    ax.scatter(pts[others, 0], pts[others, 1], s=9, c="#8794a6", edgecolors="white", linewidths=0.4, zorder=2.5)

    # shortest route through the network between the two triplets
    lengths = np.where(adj, dist, 0.0)
    best = min(((i, j) for i in tri_spk for j in tri_img), key=lambda ij: dist[ij])
    _, pred = shortest_path(lengths, directed=False, indices=best[0], return_predecessors=True)
    route = [best[1]]
    while route[-1] != best[0]:
        route.append(pred[route[-1]])
    route = route[::-1]  # from the speaking to the imagining triplet
    grad = LinearSegmentedColormap.from_list("route", [C_TRI2, C_TRI1])
    for s_idx, (i, j) in enumerate(zip(route[:-1], route[1:])):
        f = (s_idx + 0.5) / (len(route) - 1)
        link(ax, pts[i], pts[j], grad(f), rad=edge_rad, lw=2.0, z=3.2)
        dvec = pts[j] - pts[i]
        mid = (pts[i] + pts[j]) / 2 + 0.5 * edge_rad * np.array([dvec[1], -dvec[0]])
        ax.add_patch(mp.Circle(mid, 0.03, fc=grad(f), ec="white", lw=0.7, zorder=4.6))
    for i in route[1:-1]:
        ax.add_patch(mp.Circle(pts[i], 0.035, fc="white", ec="#7d8796", lw=0.8, zorder=3.4))

    for tri, col in ((tri_img, C_TRI1), (tri_spk, C_TRI2)):
        p = pts[tri]
        c = p.mean(axis=0)
        hull = rounded_polygon(c + 1.75 * (p - c))  # soft group outline with rounded corners
        ax.add_patch(mp.Polygon(hull, closed=True, fc=col, ec=col, alpha=0.16, lw=4, zorder=2.7,
                                joinstyle="round"))
        ax.add_patch(mp.Polygon(hull, closed=True, fc="none", ec=col, lw=1.1, ls=(0, (3, 2)), zorder=2.8,
                                joinstyle="round"))
        for q in p:
            node(ax, q, 0.05, col, halo=False, z=4)

    # legend below the brain
    centre_x = tf((1.7, 0))[0]
    start = centre_x - 0.95  # legend width: two markers, two labels and a gap (about 1.9 in)
    for x0, col, lab in ((start, C_TRI1, "imagine"), (start + 1.15, C_TRI2, "speak")):
        ax.plot([x0, x0 + 0.3], [0.78, 0.78], color=col, lw=2.4, solid_capstyle="round")
        ax.add_patch(mp.Circle((x0 + 0.15, 0.78), 0.045, fc=col, ec="white", lw=0.8, zorder=4))
        ax.text(x0 + 0.4, 0.78, lab, va="center", fontsize=8, fontweight="bold", color=col)

    # signals recorded from the two regions: strongly correlated
    gap_arrow(ax, 0, 1.72)
    base_rng = np.random.default_rng(5)
    t = np.linspace(0, 1, 400)
    common = np.sin(2 * np.pi * 2.6 * t) + 0.45 * np.sin(2 * np.pi * 6.1 * t + 1.0)
    for k, (y0, col, lab) in enumerate(((2.1, C_TRI1, "imagine"), (1.57, C_TRI1, "imagine"),
                                         (1.04, C_TRI2, "speak"))):
        own = gaussian_filter(base_rng.standard_normal(400), 14) * 5
        y = common + 0.45 * own + 0.15 * k
        y = (y - y.min()) / (y.max() - y.min())
        ax.plot(CX[1] - 0.62 + 1.42 * t, y0 + 0.36 * y, color=col, lw=1.5, zorder=4, solid_capstyle="round")
        ax.text(CX[1] - 0.7, y0 + 0.18, f"$x_{k + 1}$", ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(CX[1], 0.74, "one triplet of signals", ha="center", fontsize=7.3, color=MUTED)

    # higher-order measure: what the O-information of a triplet reports
    gap_arrow(ax, 1, 1.72)
    gx, gy, gw, gh = CX[2] + 0.45, 1.0, 0.15, 1.45
    cmap = plt.get_cmap("RdBu_r")
    for i in range(120):
        ax.add_patch(mp.Rectangle((gx, gy + gh * i / 120), gw, gh / 120 + 1e-3, fc=cmap(i / 119),
                                  ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="none", ec="#4a5563", lw=0.8, zorder=4))
    ax.text(gx + gw / 2, gy + gh + 0.1, r"$\Omega>0$", ha="center", va="bottom", fontsize=7.5, color="#b2182b")
    ax.text(gx + gw / 2, gy - 0.1, r"$\Omega<0$", ha="center", va="top", fontsize=7.5, color="#2166ac")
    ix = CX[2] - 0.35
    # redundancy: the three signals carry the same information (overlapping sets)
    cy = gy + gh * 0.8
    for ang, col in zip((90, 210, 330), (C_TRI1, C_TRI1, C_TRI2)):
        off = 0.065 * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        ax.add_patch(mp.Circle((ix, cy + 0.03) + off, 0.135, fc=col, ec=col, alpha=0.28, lw=1.0, zorder=4))
    ax.text(ix, cy - 0.27, "redundancy", ha="center", fontsize=7.3, fontweight="bold", color="#b2182b")
    ax.text(ix, cy - 0.4, "shared information", ha="center", fontsize=6.3, color=MUTED)
    # synergy: information present only in the triplet as a whole
    cy = gy + gh * 0.24
    tri = np.array([(ix, cy + 0.2), (ix - 0.17, cy - 0.08), (ix + 0.17, cy - 0.08)])
    ax.add_patch(mp.Polygon(tri, closed=True, fc="#2166ac", ec="#2166ac", alpha=0.22, lw=3,
                            joinstyle="round", zorder=3.5))
    for q, col in zip(tri, (C_TRI1, C_TRI1, C_TRI2)):
        node(ax, q, 0.055, col, halo=False, z=4.5)
    ax.text(ix, cy - 0.3, "synergy", ha="center", fontsize=7.3, fontweight="bold", color="#2166ac")
    ax.text(ix, cy - 0.43, "joint information", ha="center", fontsize=6.3, color=MUTED)

    top = h - SUB_DY
    subtitle(ax, CX[0], top, "A1", "Brain network")
    subtitle(ax, CX[1], top, "A2", "Correlated signals")
    subtitle(ax, CX[2], top, "A3", "O-information")

    ax.text(w / 2, 0.27, "Does the O-information reveal how brain regions interact?",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=QUESTION_INK,
            bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.25", fc=QUESTION_BG, ec=QUESTION_EDGE, lw=0.9))


# --------------------------------------------------------------------------- #
# Panel B: modelling the zebrafish brain
# --------------------------------------------------------------------------- #

def brain_inset(fig, rect_fig):
    img = np.load(RAW_ZEBRAFISH_DIR / "brain_top.npy")
    coords = np.load(RAW_ZEBRAFISH_DIR / "node_coordinates_65.npy")
    ds = load_zebrafish()
    xy = coords[ds.region_index][:, :2]
    a = ds.sc_undirected
    r0, r1, c0, c1 = 70, 890, 55, 545
    crop = img[r0:r1, c0:c1]
    crop = (crop - np.percentile(crop, 1)) / (np.percentile(crop, 99.7) - np.percentile(crop, 1))
    crop = np.clip(crop, 0, 1) ** 0.85

    ax = fig.add_axes(rect_fig)
    im = ax.imshow(crop, cmap="bone", vmin=0, vmax=1.05, extent=[c0, c1, r1, r0])
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    frame = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                           transform=ax.transAxes, fc="none", ec="none")
    ax.add_patch(frame)
    im.set_clip_path(frame)

    # strongest connections as gently curved, weight-graded lines
    iu = np.triu_indices_from(a, 1)
    order = np.argsort(a[iu])[::-1][:220]
    weights = a[iu][order]
    wn = (weights - weights.min()) / (weights.max() - weights.min() + 1e-12)
    segs, cols, lws = [], [], []
    cmap = plt.get_cmap("YlOrBr")
    for m, wgt in zip(order, wn):
        p, q = xy[iu[0][m]], xy[iu[1][m]]
        mid = (p + q) / 2
        normal = np.array([-(q - p)[1], (q - p)[0]])
        ctrl = mid + 0.15 * normal
        t = np.linspace(0, 1, 20)[:, None]
        curve = (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t ** 2 * q
        segs.append(curve)
        cols.append((*cmap(0.35 + 0.5 * wgt)[:3], 0.25 + 0.45 * wgt))
        lws.append(0.35 + 0.9 * wgt)
    lc = LineCollection(segs, colors=cols, linewidths=lws, zorder=2, capstyle="round")
    lc.set_clip_path(frame)
    ax.add_collection(lc)
    d = a.sum(axis=1)
    dn = (d - d.min()) / (d.max() - d.min())
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=4 + 14 * dn, c=dn, cmap="YlOrBr", vmin=-0.1, vmax=0.9,
                    edgecolors="#4a2a05", linewidths=0.3, zorder=3)
    sc.set_clip_path(frame)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                transform=ax.transAxes, fc="none", ec="#2b2f36", lw=1.0, zorder=5))


def wilson_cowan(ax, origin):
    ox, oy = origin
    e, i = np.array([ox + 0.0, oy]), np.array([ox + 0.78, oy])
    node(ax, e, 0.19, C_E, "E", fs=10)
    node(ax, i, 0.19, C_I, "I", fs=10)
    arrow(ax, e + [0.16, 0.12], i + [-0.16, 0.12], colour=C_E, rad=-0.45, lw=1.4, ms=9)
    ax.add_patch(FancyArrowPatch(i + [-0.16, -0.12], e + [0.16, -0.12], connectionstyle="arc3,rad=-0.45",
                                 arrowstyle="-", lw=1.4, color=C_I, zorder=3))
    ax.add_patch(mp.Circle(e + [0.17, -0.155], 0.035, fc=C_I, ec="none", zorder=4))
    ax.add_patch(mp.Arc(e + [-0.27, 0.0], 0.2, 0.26, theta1=40, theta2=320, color=C_E, lw=1.3, zorder=3))
    ax.add_patch(mp.Arc(i + [0.27, 0.0], 0.2, 0.26, theta1=-140, theta2=140, color=C_I, lw=1.3, zorder=3))
    arrow(ax, e + [0, -0.5], e + [0, -0.24], lw=1.2, ms=8)
    ax.text(e[0], e[1] - 0.6, "drive $P$", ha="center", fontsize=8, color=INK)
    arrow(ax, e + [0, 0.5], e + [0, 0.25], colour=NAVY, lw=1.3, ms=8)
    ax.text(e[0], e[1] + 0.58, r"coupling $K\,G_i$", ha="center", fontsize=7.8, color=NAVY, va="bottom")



def coupling_card(ax, x, y, w, h, colour, bg, order, formula, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=bg, ec=colour, lw=0.9, zorder=2))
    cy = y + h * 0.7
    r_src, r_tgt = 0.058, 0.062
    if order == 2:
        pj, pi = np.array([x + w * 0.3, cy]), np.array([x + w * 0.7, cy])
        arrow(ax, pj + [r_src + 0.03, 0.01], pi - [r_tgt + 0.03, -0.01], colour=colour, rad=-0.3,
              lw=0.9, ms=6, z=4)
        node(ax, pj, r_src, MUTED, "j", fs=5.8, halo=False)
        node(ax, pi, r_tgt, colour, "i", fs=5.8, halo=False)
    else:
        pj, pk = np.array([x + w * 0.3, cy + 0.12]), np.array([x + w * 0.3, cy - 0.12])
        pi = np.array([x + w * 0.7, cy])
        arrow(ax, pj + [r_src + 0.02, -0.01], pi + [-r_tgt - 0.03, 0.03], colour=colour, rad=-0.2,
              lw=0.9, ms=6, z=4)
        arrow(ax, pk + [r_src + 0.02, 0.01], pi + [-r_tgt - 0.03, -0.03], colour=colour, rad=0.2,
              lw=0.9, ms=6, z=4)
        node(ax, pj, r_src, MUTED, "j", fs=5.8, halo=False)
        node(ax, pk, r_src, MUTED, "k", fs=5.8, halo=False)
        node(ax, pi, r_tgt, colour, "i", fs=5.8, halo=False)
    ax.text(x + w / 2, y + h * 0.27, formula, ha="center", va="center", fontsize=fs, color=INK,
            linespacing=1.5, zorder=4)


def panel_zebrafish(ax, fig, panel_rect, w, h):
    px, py, pw, ph = panel_rect
    bw, bh = 1.48, 2.42
    bx, by = CX[0] - bw / 2, 0.88
    brain_inset(fig, [px + bx / w * pw, py + by / h * ph, bw / w * pw, bh / h * ph])
    ax.text(CX[0], 0.66, "111 regions  ·  36 larvae", ha="center", fontsize=7.8, color=INK)

    yc = 1.88  # vertical middle of the three columns
    gap_arrow(ax, 0, yc)
    gap_arrow(ax, 1, yc)
    wilson_cowan(ax, (CX[1] - 0.39, 2.5))
    # excitatory activity of three coupled nodes
    for k, (lab, y0, ph) in enumerate((("Node $i$", 1.42, 0.0), ("Node $j$", 1.07, 1.3), ("Node $k$", 0.72, 2.4))):
        smooth_trace(ax, CX[1] - 0.42, y0, 1.15, 0.24, ph, C_E, freq=3.0, seed=8 + k, lw=1.2)
        ax.text(CX[1] - 0.5, y0 + 0.12, lab, ha="right", va="center", fontsize=7.4, color=INK)
    ax.text(CX[1] + 0.16, 0.55, "excitatory activity $E(t)$", ha="center", va="center", fontsize=7.2,
            color=MUTED)

    top = h - SUB_DY
    subtitle(ax, CX[0], top, "B1", "Zebrafish connectome")
    subtitle(ax, CX[1], top, "B2", "Wilson–Cowan node")
    subtitle(ax, CX[2], top, "B3", "Coupling function $\\mathbf{G_i}$")
    cw, chh = 0.97, 1.1
    xs = (COLS[2][0] + 0.2, COLS[2][0] + 0.2 + cw + 0.05)
    ys = (1.9, 0.72)
    form_label(ax, xs[0] + cw / 2, 3.15, "additive", C_ADD, fs=8)
    form_label(ax, xs[1] + cw / 2, 3.15, "diffusive", C_DIF, fs=8)
    lx = COLS[2][0] + 0.08
    ax.text(lx, ys[0] + chh / 2, "2nd order", rotation=90, ha="center", va="center", fontsize=7.5, color=INK)
    ax.text(lx, ys[1] + chh / 2, "3rd order", rotation=90, ha="center", va="center", fontsize=7.5, color=INK)
    coupling_card(ax, xs[0], ys[0], cw, chh, C_ADD, C_ADD_BG, 2, r"$\sum_j A_{ij}\,x_j$", fs=7.8)
    coupling_card(ax, xs[1], ys[0], cw, chh, C_DIF, C_DIF_BG, 2, r"$\sum_j A_{ij}\,\Delta x_j$", fs=7.8)
    coupling_card(ax, xs[0], ys[1], cw, chh, C_ADD, C_ADD_BG, 3, r"$\sum_{jk} A_{ij}A_{ik}\,x_j x_k$", fs=7.2)
    coupling_card(ax, xs[1], ys[1], cw, chh, C_DIF, C_DIF_BG, 3,
                  r"$\sum_{jk} A_{ij}A_{ik}\,\Delta x_j \Delta x_k$", fs=7.2)
    ax.text(CX[2] + 0.08, 0.58, r"$\Delta x_j = x_j - x_i,\ \ j \neq k$", ha="center", fontsize=6.8, color=MUTED)
    ax.text(w / 2, 0.27, "Which coupling function reproduces the activity of the zebrafish brain?",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=QUESTION_INK,
            bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.25", fc=QUESTION_BG, ec=QUESTION_EDGE, lw=0.9))


# --------------------------------------------------------------------------- #
# Panel C: synthetic networks
# --------------------------------------------------------------------------- #

def draw_topology(ax, centre, radius, adjacency):
    n = adjacency.shape[0]
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    pts = centre + radius * np.c_[np.cos(ang), np.sin(ang)]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                gap = min(abs(i - j), n - abs(i - j))
                rad = 0.0 if n <= 3 else (0.12 if gap <= 2 else 0.05)
                link(ax, pts[i], pts[j], "#8a97a8", rad=rad, lw=1.1, alpha=0.9, z=2)
    for p in pts:
        node(ax, p, 0.072, "#3d6fa3", halo=False, z=4)


def _irregular_region(n, centre, radii, seed):
    """Boolean mask of a smooth, irregular closed region on an n x n unit grid."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, 400)
    r = 1 + sum(rng.uniform(0.04, 0.1) * np.cos(m * theta + rng.uniform(0, 2 * np.pi)) for m in (2, 3, 5))
    x = centre[0] + radii[0] * r * np.cos(theta)
    y = centre[1] + radii[1] * r * np.sin(theta)
    from matplotlib.path import Path as MPath
    g = (np.arange(n) + 0.5) / n
    kk, pp = np.meshgrid(g, g)
    inside = MPath(np.c_[x, y]).contains_points(np.c_[kk.ravel(), pp.ravel()]).reshape(n, n)
    return inside, np.c_[x, y]


# Idealised expectations (conceptual, not simulation output).
# Oscillatory region per coupling form: centre and radii in the unit (K, P) square.
REGION = {"additive": ((0.33, 0.44), (0.27, 0.24)), "diffusive": ((0.56, 0.5), (0.38, 0.3))}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _bump(k, centre, width):
    return np.exp(-((k - centre) / width) ** 2)


def _inside_trend(form, topology, k):
    """Idealised Omega inside the oscillatory regime as a function of coupling K."""
    if form == "diffusive":
        if topology == "organised":   # steady build-up of redundancy with a mild overshoot
            return 0.08 + 0.78 * _sigmoid((k - 0.36) / 0.08) + 0.14 * _bump(k, 0.56, 0.08)
        # random: brief synergy before a weaker, slowly fluctuating redundancy
        return (0.03 + 0.48 * _sigmoid((k - 0.46) / 0.08) - 0.3 * _bump(k, 0.26, 0.07)
                + 0.05 * np.sin(11 * k) * _sigmoid((k - 0.5) / 0.05))
    if topology == "organised":       # transient redundancy that fades as activity saturates
        return 0.04 + 0.6 * _bump(k, 0.24, 0.12)
    # random: weak redundancy at low coupling turning into synergy
    return 0.14 * _bump(k, 0.1, 0.06) - 0.55 * _bump(k, 0.33, 0.1)


def _background(form, topology, k, p):
    """Weak, smoothly varying Omega outside the oscillatory regime."""
    phase = 0.4 if topology == "organised" else 1.9
    sign = 1 if topology == "organised" or form == "additive" else -1
    return sign * (0.1 + 0.08 * k) + 0.07 * np.sin(2.4 * np.pi * k + phase) * np.cos(1.6 * np.pi * p)


def expected_map(form, topology, n=300):
    """Idealised O-information over the (coupling K, drive P) plane: a weak
    background that varies smoothly, and a stronger, K-dependent signal inside an
    irregular oscillatory region. Returns the field, the region boundary and
    the region mask."""
    centre, radii = REGION[form]
    inside, boundary = _irregular_region(n, centre, radii, seed=3 if form == "additive" else 8)
    g = (np.arange(n) + 0.5) / n
    kk, pp = np.meshgrid(g, g)
    depth = distance_transform_edt(inside) / n
    core = 0.6 + 0.4 * np.tanh(depth / 0.07)
    soft = gaussian_filter(inside.astype(float), 9)
    # gently rolling relief so that the weak background reads as a smooth landscape
    rng = np.random.default_rng({"additive": 1, "diffusive": 2}[form] * 10 + (topology == "random"))
    relief = gaussian_filter(rng.standard_normal((n, n)), 34)
    relief = 0.12 * relief / relief.std()
    background = _background(form, topology, kk, pp) + relief
    field = (1 - soft) * background + soft * (_inside_trend(form, topology, kk) * core + 0.35 * relief)
    return gaussian_filter(field, 2.0), boundary, inside


# contour-map palette for the expected Omega: the same red-white-blue scale as panel A3
OMEGA_CMAP = plt.get_cmap("RdBu_r")
OMEGA_LEVELS = np.linspace(-1, 1, 23)


def expected_panel(fig, rect_fig, field, boundary, colour, xlabel, ylabel):
    ax = fig.add_axes(rect_fig)
    g = (np.arange(field.shape[0]) + 0.5) / field.shape[0]
    ax.contourf(g, g, np.clip(field, -0.999, 0.999), levels=OMEGA_LEVELS, cmap=OMEGA_CMAP, zorder=1)
    ax.contour(g, g, field, levels=OMEGA_LEVELS, colors="white", linewidths=0.45, linestyles="solid",
               alpha=0.9, zorder=2)
    ax.plot(*boundary.T, color="white", lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot(*boundary.T, color=INK, lw=1.0, solid_capstyle="round", zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(colour)
        sp.set_linewidth(1.4)
    if xlabel:
        ax.set_xlabel(r"coupling $K$ $\rightarrow$", fontsize=7.5, labelpad=3, color=INK)
    if ylabel:
        ax.set_ylabel(r"drive $P$ $\rightarrow$", fontsize=7.5, labelpad=3, color=INK)


TOPO_COLOUR = {"organised": "#2f7d4a", "random": "#9a6324"}
FORM_STYLE = {"additive": "-", "diffusive": (0, (3.5, 1.8))}


def expected_trend(fig, rect_fig, rows, forms):
    """Idealised Omega in the oscillatory regime as coupling grows, using the same
    rule as the maps: colour = topology, line style = coupling form."""
    ax = fig.add_axes(rect_fig)
    ax.axhspan(0, 1.2, color="#fbe9e4", zorder=0)
    ax.axhspan(-1.2, 0, color="#e6eff7", zorder=0)
    ax.axhline(0, color="#8a96a6", lw=0.7, ls="--", zorder=1)
    k_stop = None
    for _, topo in rows:
        for form, _, _ in forms:
            _, _, inside = expected_map(form, topo)
            g = (np.arange(inside.shape[1]) + 0.5) / inside.shape[1]
            exists = inside.mean(axis=0) > 0.04       # coupling values with an oscillatory regime
            curve = np.where(exists, _inside_trend(form, topo, g), np.nan)
            col = TOPO_COLOUR[topo]
            ax.plot(g, curve, color=col, lw=1.35, ls=FORM_STYLE[form], solid_capstyle="round",
                    dash_capstyle="round", zorder=3)
            idx = np.flatnonzero(exists)
            for j in (idx[0], idx[-1]):
                ax.plot(g[j], curve[j], "o", ms=3.2, mfc="white", mec=col, mew=1.0, zorder=4)
            if form == "additive":
                k_stop = g[idx[-1]]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.95, 1.08)
    ax.set_xticks([])
    ax.set_yticks([0])
    ax.set_yticklabels(["0"], fontsize=7)
    ax.tick_params(length=2, width=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color(INK)
    ax.set_xlabel(r"coupling $K$ $\rightarrow$", fontsize=7.5, labelpad=3, color=INK)
    ax.set_ylabel(r"$\Omega$ in the oscillatory regime", fontsize=7.2, labelpad=2, color=INK)
    ax.text(0.04, 0.97, "redundancy", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#b2182b")
    ax.text(0.97, 0.03, "synergy", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#2166ac")
    return ax


def panel_synthetic(ax, fig, panel_rect, w, h):
    px, py, pw, ph = panel_rect
    to_fig = lambda x, y, ww, hh: [px + x / w * pw, py + y / h * ph, ww / w * pw, hh / h * ph]  # noqa: E731
    nets = synthetic_networks()
    rows = (("ring", "organised"), ("random", "random"))
    forms = (("additive", C_ADD, C_ADD_BG), ("diffusive", C_DIF, C_DIF_BG))

    top = h - SUB_DY
    y_rows = (2.6, 1.56)
    mw, mh = 0.86, 0.86
    mx = (COLS[1][0] + 0.2, COLS[1][0] + 0.2 + mw + 0.08)
    subtitle(ax, CX[0], top, "C1", "Topologies")
    subtitle(ax, CX[1], top, "C2", "Expected $\\mathbf{\\Omega}$")
    subtitle(ax, CX[2], top, "C3", "Additive vs diffusive")
    for (form, col, _), x0 in zip(forms, mx):
        form_label(ax, x0 + mw / 2, y_rows[0] + mh / 2 + 0.15, form, col)
    for r, ((net, label), yc) in enumerate(zip(rows, y_rows)):
        draw_topology(ax, np.array([CX[0] + 0.28, yc]), 0.4, nets[net].adjacency)
        ax.text(CX[0] - 0.55, yc, label, ha="center", va="center", fontsize=8.3, color=INK)
        for k, ((form, col, _), x0) in enumerate(zip(forms, mx)):
            field, boundary, _ = expected_map(form, label)
            expected_panel(fig, to_fig(x0, yc - mh / 2, mw, mh), field, boundary, col,
                           xlabel=(r == 1), ylabel=(k == 0))

    # colour scale to the right of the maps
    cmap = OMEGA_CMAP
    bx, by0, by1, bw = mx[1] + mw + 0.1, y_rows[1] - mh / 2, y_rows[0] + mh / 2, 0.1
    for i in range(100):
        ax.add_patch(mp.Rectangle((bx, by0 + (by1 - by0) * i / 100), bw, (by1 - by0) / 100 + 1e-3,
                                  fc=cmap(i / 99), ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((bx, by0), bw, by1 - by0, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="none", ec="#4a5563", lw=0.7, zorder=4))
    ax.text(bx + bw / 2, by1 + 0.07, r"$\Omega>0$", ha="center", va="bottom", fontsize=7.3, color="#b2182b")
    ax.text(bx + bw / 2, by0 - 0.07, r"$\Omega<0$", ha="center", va="top", fontsize=7.3, color="#2166ac")

    # legend below the maps
    lx = CX[1] - 0.6
    ax.plot([lx, lx + 0.25], [0.8, 0.8], color=INK, lw=1.0)
    ax.text(lx + 0.31, 0.8, "oscillatory regime", fontsize=7, color=MUTED, va="center")

    # flow between the three sub-panels
    ymid = (y_rows[0] + y_rows[1]) / 2
    gap_arrow(ax, 0, ymid)
    gap_arrow(ax, 1, ymid)
    expected_trend(fig, to_fig(COLS[2][0] + 0.48, by0, 1.58, by1 - by0), rows, forms)
    # legend of C3: colour = topology, line style = coupling form
    ly = by0 - 0.38
    for j, (_, topo) in enumerate(rows):
        x0 = COLS[2][0] + 0.32 + j * 0.92
        ax.plot([x0, x0 + 0.2], [ly, ly], color=TOPO_COLOUR[topo], lw=1.6, solid_capstyle="round")
        ax.text(x0 + 0.25, ly, topo, fontsize=6.8, color=TOPO_COLOUR[topo], fontweight="bold", va="center")
    for j, (form, _, _) in enumerate(forms):
        x0 = COLS[2][0] + 0.32 + j * 0.92
        ax.plot([x0, x0 + 0.2], [ly - 0.17, ly - 0.17], color=INK, lw=1.2, ls=FORM_STYLE[form])
        ax.text(x0 + 0.25, ly - 0.17, form, fontsize=6.8, color=INK, va="center")


    ax.text(w / 2, 0.27, "When and why does the coupling form change the O-information?",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=QUESTION_INK,
            bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.25", fc=QUESTION_BG, ec=QUESTION_EDGE, lw=0.9))


# --------------------------------------------------------------------------- #

def main() -> None:
    fig_h = sum(PANEL_H) + GAP * (len(PANEL_H) + 1)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    titles = ("The question", "Modelling the zebrafish brain", "Synthetic networks to understand the connectome")
    letters = "ABC"
    top = fig_h - GAP
    for k, ph_in in enumerate(PANEL_H):
        rect = [0.0, (top - ph_in) / fig_h, 1.0, ph_in / fig_h]
        ax = fig.add_axes(rect)
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, ph_in)
        ax.set_aspect("equal")
        ax.axis("off")
        card(ax, FIG_W, ph_in, letters[k], titles[k])
        if k == 0:
            panel_question(ax, FIG_W, ph_in)
        elif k == 1:
            panel_zebrafish(ax, fig, rect, FIG_W, ph_in)
        else:
            panel_synthetic(ax, fig, rect, FIG_W, ph_in)
        top -= ph_in + GAP
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"methodology.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved", FIG / "methodology")


if __name__ == "__main__":
    main()
