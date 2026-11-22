"""Animated overview: a stimulus (a singing bird, seen and heard) triggers brain activity, the network becomes
active, two groups of regions exchange information and their signals become
correlated.

The brain, its network and the colour scheme are those of Figure 1
(scripts/08_figure_methodology.py). The animation is illustrative: signals are
synthetic and do not come from the recordings or the models.

Output: results/figures/brain_interaction.gif

Usage:
    python scripts/09_animation_brain_interaction.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.interpolate import splev, splprep  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.sparse.csgraph import connected_components, shortest_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("fig08", ROOT / "scripts" / "08_figure_methodology.py")
fig08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig08)

OUT = ROOT / "results" / "figures" / "brain_interaction.gif"
FPS = 12
PHASES = (0, 26, 58, 90, 150)          # frame boundaries of the four scenes
N_FRAMES = PHASES[-1]
W, H = 10.0, 5.6

INK, MUTED = fig08.INK, fig08.MUTED
C_A, C_B = fig08.C_TRI1, fig08.C_TRI2   # coral and blue groups of regions
ACTIVE = "#f4a261"
CAPTIONS = ("1 · A stimulus triggers brain activity: seeing and hearing a bird",
            "2 · Activity spreads through the brain network",
            "3 · Two groups of regions exchange information",
            "4 · Their signals become correlated: shared information, Ω > 0")

BRAIN_ORIGIN, BRAIN_SCALE = np.array([2.63, 2.32]), 0.9
_tf = lambda p: np.asarray(p, dtype=float) * BRAIN_SCALE + BRAIN_ORIGIN  # noqa: E731
PERCH = np.array([0.82, 2.72])          # where the bird lands, on a branch at the left
BIRD_SCALE = 1.35
SIGHT, SOUND = "#fb8500", "#7b5ea7"    # colours of the visual and auditory inputs
HAPPY_FRAMES = (14, 34)                 # the face turns from neutral to smiling in this window
SKIN, SHADE, LINE = "#f9dccb", "#e6b49c", "#8a6a3a"
HAIR, HAIR_LIGHT, BROW = "#6a4834", "#86634b", "#6b3d66"


# --------------------------------------------------------------------------- #
# Static geometry
# --------------------------------------------------------------------------- #

def _px(points_px):
    """Map points traced on a 430 x 600 px profile drawing (facing right) to canvas coordinates
    of the head facing left; the brain fits inside the skull under the same mapping."""
    p = np.asarray(points_px, dtype=float)
    return _tf(np.c_[3.7575 - 0.0105 * p[:, 0], 2.68 - 0.0105 * p[:, 1]])


def _closed(points_px, n=300):
    q = _px(list(points_px) + [points_px[0]])
    tck, _ = splprep([q[:, 0], q[:, 1]], s=0, per=True)
    return np.column_stack(splev(np.linspace(0, 1, n), tck))


def _open(points_px, n=120):
    return fig08._smooth_curve(_px(points_px), n)


EYE = _px([(358, 232)])[0]
EAR = _px([(158, 258)])[0]

HEAD_PX = [(20, 650), (45, 520), (70, 430), (88, 365), (55, 290), (38, 200), (50, 110), (100, 55), (180, 30),
           (270, 30), (340, 60), (372, 110), (380, 140), (383, 175), (382, 200), (372, 222), (385, 245),
           (405, 270), (421, 290), (424, 302), (413, 311), (392, 316), (393, 333), (403, 347), (390, 357),
           (399, 368), (384, 386), (392, 414), (386, 437), (352, 446), (322, 442), (298, 458), (279, 495),
           (268, 545), (268, 650)]
HAIR_PX = [(88, 368), (40, 290), (14, 200), (22, 110), (65, 55), (130, 33), (200, 17), (275, 21), (335, 37),
           (378, 62), (390, 92), (372, 118), (332, 117), (296, 122), (268, 138), (252, 175), (232, 215),
           (205, 245), (170, 262), (140, 292), (115, 330)]
EAR_PX = [(172, 210), (152, 203), (130, 222), (124, 252), (132, 282), (148, 302), (170, 314), (188, 302),
          (190, 272), (186, 240)]


def build_network():
    """Connected network inside the cerebrum, two triplets and the route between them."""
    from matplotlib.path import Path as MPath
    outline = fig08._brain_outline(BRAIN_ORIGIN, BRAIN_SCALE)
    path = MPath(outline)
    tf = lambda p: np.asarray(p) * BRAIN_SCALE + BRAIN_ORIGIN  # noqa: E731
    rng = np.random.default_rng(11)
    cand = np.c_[rng.uniform(0.2, 3.3, 1500), rng.uniform(0.45, 2.2, 1500)] * BRAIN_SCALE + BRAIN_ORIGIN
    clearance = np.min(np.linalg.norm(cand[:, None] - outline[None], axis=2), axis=1)
    cand = cand[path.contains_points(cand) & (clearance > 0.1)]
    pts = [cand[0]]
    for q in cand[1:]:
        if np.min(np.linalg.norm(np.array(pts) - q, axis=1)) > 0.2:
            pts.append(q)
    pts = np.array(pts)
    n = len(pts)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    adj = np.zeros((n, n), bool)
    for i in range(n):
        adj[i, np.argsort(dist[i])[1:4]] = True
    adj |= adj.T
    _, comp = connected_components(adj, directed=False)
    while comp.max() > 0:
        d = np.where(comp[:, None] != comp[None, :], dist, np.inf)
        i, j = np.unravel_index(np.argmin(np.where(comp[:, None] == 0, d, np.inf)), d.shape)
        adj[i, j] = adj[j, i] = True
        _, comp = connected_components(adj, directed=False)

    def triplet_near(centre):
        order = np.argsort(np.linalg.norm(pts - centre, axis=1))[:8]
        best, area = order[:3], -1.0
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                for c in range(b + 1, len(order)):
                    p, q, r = pts[order[[a, b, c]]]
                    ar = abs((q - p)[0] * (r - p)[1] - (q - p)[1] * (r - p)[0]) / 2
                    if ar > area and max(np.linalg.norm(q - p), np.linalg.norm(r - p),
                                         np.linalg.norm(r - q)) < 0.55:
                        best, area = order[[a, b, c]], ar
        return best

    tri_a, tri_b = triplet_near(tf((2.35, 1.62))), triplet_near(tf((1.0, 1.12)))
    lengths = np.where(adj, dist, 0.0)
    s, t = min(((i, j) for i in tri_b for j in tri_a), key=lambda ij: dist[ij])
    _, pred = shortest_path(lengths, directed=False, indices=s, return_predecessors=True)
    route = [t]
    while route[-1] != s:
        route.append(pred[route[-1]])
    route = route[::-1]
    return pts, adj, tri_a, tri_b, route, tf


PTS, ADJ, TRI_A, TRI_B, ROUTE, TF = build_network()
VISUAL_AREA = TF((3.05, 1.25))          # posterior (occipital) pole, where visual input arrives
AUDITORY_AREA = TF((1.55, 0.72))        # temporal lobe, where auditory input arrives
DIST_FROM_INPUT = np.minimum(np.linalg.norm(PTS - VISUAL_AREA, axis=1), np.linalg.norm(PTS - AUDITORY_AREA, axis=1))
OPTIC = fig08._smooth_curve([EYE, TF((0.5, 0.78)), TF((1.7, 0.95)), VISUAL_AREA], 60)
AUDITORY = fig08._smooth_curve([EAR + [-0.12, 0.2], TF((1.75, 0.3)), AUDITORY_AREA], 40)

# synthetic signals: a common rhythm shared progressively by the three traces
T_SIG = np.linspace(0, 1, 300)
_rng = np.random.default_rng(5)
COMMON = np.sin(2 * np.pi * 3.0 * T_SIG) + 0.45 * np.sin(2 * np.pi * 7.1 * T_SIG + 1.0)
OWN = [gaussian_filter(_rng.standard_normal(300), 6) * 4 for _ in range(3)]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def phase_progress(frame, k):
    return ease((frame - PHASES[k]) / (PHASES[k + 1] - PHASES[k]))


def draw_static(ax, happy=0.0):
    """Head, face and brain. `happy` in [0, 1] moves the face from neutral to smiling."""
    head = np.vstack([_open(HEAD_PX, 500)])
    head_patch = mp.Polygon(head, closed=True, fc=SKIN, ec=LINE, lw=1.4, zorder=0.5, joinstyle="round")
    ax.add_patch(head_patch)
    # soft shading under the jaw and along the back of the neck
    for pts, a in (([(215, 372), (262, 410), (320, 436), (352, 447), (300, 458), (282, 492), (262, 470),
                     (232, 420)], 0.55),
                   ([(88, 370), (110, 420), (95, 520), (60, 650), (35, 650), (60, 500), (75, 420)], 0.35)):
        sh = mp.Polygon(_closed(pts), closed=True, fc=SHADE, ec="none", alpha=a, zorder=0.55)
        ax.add_patch(sh)
        sh.set_clip_path(head_patch)
    jaw = _open([(205, 355), (238, 397), (290, 426), (345, 440)])
    ax.plot(*jaw.T, color=LINE, lw=1.2, zorder=0.6, solid_capstyle="round")
    # hair, with a lighter sheen on the crown
    hair = mp.Polygon(_closed(HAIR_PX, 400), closed=True, fc=HAIR, ec=LINE, lw=1.4, zorder=0.7,
                      joinstyle="round")
    ax.add_patch(hair)
    sheen = mp.Polygon(_closed([(120, 60), (220, 30), (320, 50), (300, 110), (200, 130), (110, 150)]),
                       closed=True, fc=HAIR_LIGHT, ec="none", alpha=0.55, zorder=0.71)
    ax.add_patch(sheen)
    sheen.set_clip_path(hair)

    # eye in profile: two lids meeting at the outer corner and a dark cornea at the front;
    # the lower lid rises with the smile
    e = 1.1
    ax.add_patch(mp.Ellipse(EYE + [0.03, -0.1], 0.3, 0.12, fc="#f0b9a3", ec="none", alpha=0.6, zorder=2.95))
    apex = EYE + np.array([0.2, 0.0]) * e
    up_front = EYE + np.array([-0.12, 0.15]) * e
    lo_front = EYE + np.array([-0.12, -0.14 + 0.05 * happy]) * e
    upper = fig08._smooth_curve([up_front, EYE + np.array([0.04, 0.095]) * e, apex], 40)
    lower = fig08._smooth_curve([lo_front, EYE + np.array([0.04, -0.085 + 0.03 * happy]) * e, apex], 40)
    white = np.vstack([upper, lower[::-1]])
    clip = mp.Polygon(white, closed=True, transform=ax.transData)
    ax.add_patch(mp.Polygon(white, closed=True, fc="white", ec="none", zorder=3))
    cornea = mp.Ellipse(EYE + np.array([-0.075, 0.0]) * e, 0.1 * e, 0.27 * e, fc="#3d4a56", ec="none", zorder=3.1)
    ax.add_patch(cornea)
    cornea.set_clip_path(clip)
    shine = mp.Ellipse(EYE + np.array([-0.07, 0.05]) * e, 0.03, 0.07, fc="white", ec="none", zorder=3.2)
    ax.add_patch(shine)
    shine.set_clip_path(clip)
    ax.plot(*upper.T, color="#5a4030", lw=2.0, zorder=3.4, solid_capstyle="round")
    ax.plot(*lower.T, color=LINE, lw=1.1, zorder=3.4, solid_capstyle="round")
    lift = 0.05 * happy
    brow = fig08._smooth_curve(_px([(318, 216), (348, 200), (382, 196)]) + [0, lift], 30)
    ax.plot(*brow.T, color=BROW, lw=3.4, zorder=3.4, solid_capstyle="round")

    # ear, drawn over the hair
    ax.add_patch(mp.Polygon(_closed(EAR_PX), closed=True, fc=SKIN, ec=LINE, lw=1.4, zorder=3.0, joinstyle="round"))
    ax.plot(*_open([(176, 300), (166, 288), (160, 262), (150, 236), (156, 222), (172, 226)]).T, color=LINE,
            lw=1.1, zorder=3.05)
    ax.plot(*_open([(150, 255), (160, 270), (172, 268)]).T, color=LINE, lw=1.0, zorder=3.05)

    # nose: wing of the nostril
    ax.plot(*_open([(388, 292), (384, 304), (393, 311), (406, 309)]).T, color=LINE, lw=1.3, zorder=3)

    # mouth: the line between the lips curves upwards as the face smiles
    front = _px([(390, 357)])[0]
    corner = _px([(370, 360)])[0] + (_px([(358, 343)])[0] - _px([(370, 360)])[0]) * happy
    mid = _px([(380, 358.5)])[0] + (_px([(378, 366)])[0] - _px([(380, 358.5)])[0]) * happy
    ax.plot(*fig08._smooth_curve([front, mid, corner], 30).T, color="#9c5a4a", lw=1.6, zorder=3,
            solid_capstyle="round")
    if happy > 0:  # smile line on the cheek
        crease = fig08._smooth_curve([corner + [0.04, 0.13], corner + [0.08, 0.05], corner + [0.06, -0.04]], 20)
        ax.plot(*crease.T, color=LINE, lw=1.0, alpha=happy, zorder=3)
    # blush appears with the smile
    for r, a in ((0.2, 0.16), (0.13, 0.24)):
        ax.add_patch(mp.Ellipse(_px([(330, 318)])[0], 2 * r, 1.3 * r, fc="#f28482", ec="none",
                                alpha=a * happy, zorder=2.9))
    _, _, cerebrum = fig08.draw_brain(ax, BRAIN_ORIGIN, BRAIN_SCALE, sulci=False)
    return cerebrum


def bird_position(frame):
    """The bird flies in from the left along a gentle arc and lands on the branch."""
    u = ease(frame / 12.0)
    start = np.array([-0.6, 3.6])
    pos = start + (PERCH - start) * u + np.array([0.0, 0.35 * np.sin(np.pi * u)])
    hop = 0.03 * abs(np.sin(frame * 0.5)) if frame > 12 else 0.0
    return pos + [0.0, hop], frame < 12


def branch(ax):
    twig = fig08._smooth_curve([(-0.1, 2.33), (0.5, 2.4), (1.1, 2.38), (1.45, 2.46)], 40)
    ax.plot(*twig.T, color="#7a5232", lw=5, zorder=3.5, solid_capstyle="round")
    ax.plot(*fig08._smooth_curve([(0.3, 2.39), (0.38, 2.17), (0.52, 2.08)], 20).T, color="#7a5232", lw=3,
            zorder=3.5, solid_capstyle="round")
    for c, ang in (((0.54, 2.06), -30), ((1.3, 2.54), 35), ((0.1, 2.45), 20)):
        ax.add_patch(mp.Ellipse(c, 0.22, 0.1, angle=ang, fc="#6aa84f", ec="#3f7d2c", lw=0.8, zorder=3.55))


def bird(ax, pos, frame, flying, singing):
    """A small round songbird facing right (towards the person); returns the tip of the beak."""
    from matplotlib.transforms import Affine2D
    tr = Affine2D().scale(BIRD_SCALE).translate(*pos) + ax.transData
    blue, dark, belly, beak = "#5aa9e6", "#2f6f9f", "#fde4a5", "#f4a261"
    z = 4.2

    def patch(p):
        p.set_transform(tr)
        ax.add_patch(p)

    def line(x, y, **kw):
        ax.plot(x, y, transform=tr, solid_capstyle="round", **kw)

    patch(mp.Polygon([(-0.17, 0.04), (-0.38, 0.17), (-0.4, 0.06), (-0.36, -0.02), (-0.17, -0.05)], closed=True,
                     fc=dark, ec=dark, lw=1.0, zorder=z, joinstyle="round"))
    if not flying:
        for dx in (-0.03, 0.06):
            line([dx, dx], [-0.16, -0.25], color="#c77d3a", lw=1.8, zorder=z)
    patch(mp.Ellipse((0, 0), 0.46, 0.38, fc=blue, ec=dark, lw=1.4, zorder=z + 0.1))
    patch(mp.Ellipse((0.06, -0.07), 0.28, 0.2, fc=belly, ec="none", zorder=z + 0.15))
    head = np.array([0.17, 0.14])
    patch(mp.Circle(head, 0.14, fc=blue, ec=dark, lw=1.4, zorder=z + 0.2))
    patch(mp.Ellipse((0.07, 0.03), 0.24, 0.24, fc=blue, ec="none", zorder=z + 0.21))
    for k in range(3):   # small crest
        base = head + [-0.03 + 0.03 * k, 0.13]
        line([base[0], base[0] - 0.03 + 0.02 * k], [base[1], base[1] + 0.07], color=dark, lw=1.5, zorder=z + 0.19)
    ang = 35 * np.sin(frame * 1.6) if flying else -15 + 4 * np.sin(frame * 0.3)
    patch(mp.Ellipse((-0.05, 0.04), 0.28, 0.15, angle=ang, fc=dark, ec=dark, lw=0.8, zorder=z + 0.3))
    eye = head + [0.04, 0.03]
    patch(mp.Circle(eye, 0.042, fc="white", zorder=z + 0.4))
    patch(mp.Circle(eye + [0.01, 0.0], 0.026, fc="#1f2a36", zorder=z + 0.41))
    patch(mp.Circle(eye + [0.018, 0.012], 0.009, fc="white", zorder=z + 0.42))
    patch(mp.Ellipse(head + [0.03, -0.06], 0.07, 0.04, fc="#f28482", alpha=0.6, ec="none", zorder=z + 0.4))
    o = 0.045 * (0.5 + 0.5 * np.sin(frame * 1.3)) if singing else 0.0
    root = head + [0.12, -0.01]
    for part in ([root + [0, 0.035], root + [0.13, 0.01 + o], root + [0, -0.005]],
                 [root + [0, -0.005], root + [0.11, -0.02 - o], root + [0, -0.035]]):
        patch(mp.Polygon(part, closed=True, fc=beak, ec="#c77d3a", lw=0.8, zorder=z + 0.35, joinstyle="round"))
    return pos + BIRD_SCALE * (root + [0.12, 0.0])


def render(frame, fig):
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    happy = ease((frame - HAPPY_FRAMES[0]) / (HAPPY_FRAMES[1] - HAPPY_FRAMES[0]))
    cerebrum = draw_static(ax, happy)
    k = max(i for i in range(4) if frame >= PHASES[i])
    ax.text(W / 2, H - 0.35, CAPTIONS[k], ha="center", va="center", fontsize=15, fontweight="bold",
            color=fig08.NAVY)

    # scene 1: a bird arrives and sings; the person sees it and hears it
    branch(ax)
    pos, flying = bird_position(frame)
    singing = 12 <= frame < PHASES[2]
    beak = bird(ax, pos, frame, flying, singing)
    a_in = np.clip((frame - 8) / 8, 0, 1) * (1 - np.clip((frame - PHASES[2] + 4) / 10, 0, 1))
    if a_in > 0:
        # sight: light from the bird to the eye
        for dy in (-0.08, 0.0, 0.08):
            a, b = beak + [0.08, dy], EYE + [-0.15, dy * 0.4]
            ax.plot(*np.c_[a, b], color=SIGHT, lw=1.0, ls=(0, (3, 3)), alpha=0.6 * a_in, zorder=3)
            for s in np.arange(0, 1, 0.25):
                q = a + (b - a) * ((s + frame * 0.06) % 1.0)
                ax.add_patch(mp.Circle(q, 0.028, fc=SIGHT, alpha=0.9 * a_in, zorder=4))
        # sound: notes travel from the beak towards the ear, and waves reach the ear
        for k in range(4):
            u = ((frame - 12) * 0.03 + k / 4) % 1.0
            if frame < 12:
                continue
            q = beak + (EAR + [-0.3, 0.05] - beak) * u + np.array([0.0, 0.45 * np.sin(np.pi * u)])
            ax.text(*q, "\u266a" if k % 2 else "\u266b", ha="center", va="center", fontsize=19, color=SOUND,
                    alpha=a_in * np.sin(np.pi * u), zorder=4.5)
        for k in range(3):
            r = 0.18 + 0.12 * ((k + frame * 0.15) % 3)
            th = np.deg2rad(np.linspace(120, 240, 30))
            alpha = a_in * (1 - (r - 0.18) / 0.36) * 0.8
            ax.plot(EAR[0] + r * np.cos(th), EAR[1] + 0.05 + r * np.sin(th), color=SOUND, lw=1.6, alpha=alpha,
                    zorder=3.6, solid_capstyle="round")
        ax.text(*(beak + [-0.05, 0.5]), "sound", ha="center", fontsize=10, style="italic", color=SOUND, alpha=a_in)

    # sensory pathways to the visual (posterior) and auditory (temporal) areas
    p_path = ease((frame - (PHASES[1] - 6)) / 14)
    if p_path > 0:
        m = max(2, int(p_path * len(OPTIC)))
        ln, = ax.plot(*OPTIC[:m].T, color=SIGHT, lw=2.2, alpha=0.75, zorder=3.3)
        ln.set_clip_path(cerebrum)
        m = max(2, int(p_path * len(AUDITORY)))
        ax.plot(*AUDITORY[:m].T, color=SOUND, lw=2.2, alpha=0.75, zorder=3.3, solid_capstyle="round")

    # network edges
    for i, j in zip(*np.nonzero(np.triu(ADJ))):
        fig08.link(ax, PTS[i], PTS[j], "#9aa5b4", rad=0.12, lw=0.6, alpha=0.55, z=2).set_clip_path(cerebrum)

    # scene 2: a wave of activity spreads from the visual and auditory areas
    wave_front = (frame - PHASES[1]) * 0.12
    act = np.exp(-((DIST_FROM_INPUT - wave_front) / 0.45) ** 2) if frame >= PHASES[1] else np.zeros(len(PTS))
    flicker = 0.25 + 0.25 * np.sin(frame * 0.7 + np.arange(len(PTS)) * 1.3) if frame >= PHASES[2] else 0
    level = np.clip(act + flicker, 0, 1)
    cmap = LinearSegmentedColormap.from_list("act", ["#8794a6", ACTIVE])
    ax.scatter(PTS[:, 0], PTS[:, 1], s=14 + 40 * level, c=cmap(level), edgecolors="white", linewidths=0.5,
               zorder=2.5)

    # scene 3: two groups of regions and the exchange of information along the network
    p3 = phase_progress(frame, 2)
    if frame >= PHASES[2]:
        for tri, col in ((TRI_A, C_A), (TRI_B, C_B)):
            p = PTS[tri]
            c = p.mean(axis=0)
            hull = fig08.rounded_polygon(c + 1.75 * (p - c))
            ax.add_patch(mp.Polygon(hull, closed=True, fc=col, ec=col, alpha=0.18 * p3, lw=4, zorder=2.7,
                                    joinstyle="round"))
            ax.add_patch(mp.Polygon(hull, closed=True, fc="none", ec=col, lw=1.2, ls=(0, (3, 2)),
                                    alpha=p3, zorder=2.8))
            for q in p:
                ax.add_patch(mp.Circle(q, 0.06, fc=col, ec="white", lw=1.2, alpha=max(0.3, p3), zorder=4))
        grad = LinearSegmentedColormap.from_list("route", [C_B, C_A])
        seg = list(zip(ROUTE[:-1], ROUTE[1:]))
        for s_idx, (i, j) in enumerate(seg):
            f = (s_idx + 0.5) / len(seg)
            fig08.link(ax, PTS[i], PTS[j], grad(f), rad=0.12, lw=2.4 * p3, alpha=p3, z=3.2)
        # packets travelling in both directions
        poly = np.array([PTS[i] for i in ROUTE])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        for direction, col in ((1, C_A), (-1, C_B)):
            for off in (0.0, 0.33, 0.66):
                u = (off + direction * frame * 0.035) % 1.0
                d = u * cum[-1]
                idx = min(np.searchsorted(cum, d) - 1, len(poly) - 2)
                idx = max(idx, 0)
                w = (d - cum[idx]) / max(cum[idx + 1] - cum[idx], 1e-9)
                q = poly[idx] + (poly[idx + 1] - poly[idx]) * w
                ax.add_patch(mp.Circle(q, 0.05, fc=col, ec="white", lw=0.8, alpha=p3, zorder=4.6))

    # the three recorded regions (one triplet): two from the coral group, one from the blue group
    if frame >= PHASES[2] + 10:
        a_lab = min(1.0, (frame - PHASES[2] - 10) / 8)
        for idx, lab, col, off in ((TRI_A[0], "$x_1$", C_A, (0.16, 0.14)), (TRI_A[1], "$x_2$", C_A, (0.18, -0.12)),
                                   (TRI_B[0], "$x_3$", C_B, (-0.2, 0.14))):
            q = PTS[idx]
            ax.add_patch(mp.Circle(q, 0.1, fc="none", ec=col, lw=2.0, alpha=a_lab, zorder=4.7))
            ax.text(q[0] + off[0], q[1] + off[1], lab, ha="center", va="center", fontsize=12, color=col,
                    fontweight="bold", alpha=a_lab, zorder=4.8,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85 * a_lab))

    # scene 4: correlated signals
    p4 = phase_progress(frame, 3)
    x0, wd = 6.75, 2.95
    ax.text(x0 + wd / 2, 4.35, "signals of one triplet of regions", ha="center", fontsize=11, style="italic", color=MUTED)
    corr = 0.15 + 0.8 * p4
    shown = max(2, int(len(T_SIG) * min(1.0, 0.25 + (frame - PHASES[2]) / (PHASES[4] - PHASES[2]))))
    for k_sig, (y0, col, lab) in ([] if frame < PHASES[1] else enumerate(((3.55, C_A, "$x_1$"), (2.65, C_A, "$x_2$"), (1.75, C_B, "$x_3$")))):
        y = corr * COMMON + (1 - corr) * OWN[k_sig]
        y = (y - y.min()) / (y.max() - y.min())
        ax.text(x0 - 0.12, y0 + 0.25, lab, ha="right", va="center", fontsize=13, color=INK)
        ax.plot(x0 + wd * T_SIG[:shown], y0 + 0.5 * y[:shown], color=col, lw=2.0,
                alpha=0.35 if frame < PHASES[2] else 1.0, zorder=4, solid_capstyle="round")
    if frame >= PHASES[3] + 18:
        a = min(1.0, (frame - PHASES[3] - 18) / 10)
        ax.text(x0 + wd / 2, 1.0, "Ω > 0 · redundancy", ha="center", fontsize=14, fontweight="bold",
                color="#b2182b", alpha=a,
                bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.3", fc="#fbe9e4", ec="#e8a59a", alpha=a))


def main() -> None:
    fig = plt.figure(figsize=(W, H), dpi=90)
    anim = FuncAnimation(fig, lambda f: render(f, fig), frames=N_FRAMES, interval=1000 / FPS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
