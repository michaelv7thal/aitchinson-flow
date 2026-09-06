"""fig:image-text — the image-versus-text geometry behind the EqM collapse.
Left: continuous image data, marginal mean inside the typical set, a learnable
basin. Right: the 2-simplex, data at the one-hot vertices, the sampler starting
at the random source on the ridge where every vertex basin meets.

Two corrections to earlier versions of this panel.

1. The right panel used to draw concentric bowl contours at the centroid with
   the field pointing inwards, captioned "a single tilted bowl".  The
   measurement in fig:energy-landscape falsifies that: the trained field points
   AT the nearest vertex from anywhere with signal, and every vertex is a basin.

2. The centre marker used to be mu_1, captioned "equidistant from all".  Both
   are wrong.  What sits at the centre is the random source x_0 the sampler
   starts from (radius 0.51 against 12.27 for the data), and it is the source,
   not mu_1, that has no vertex nearer than any other.  mu_1 is the unigram
   tilt of prop:collapse, a fifth of the way from the centroid out to a vertex
   (2.43 against 12.27), and it is not where the descent ends: at gamma=0 the
   descent leaves the source and sharpens to a vertex out at the data radius
   (app:sampling).  Measured on runs/compu_mse_det at n=256, the returned
   iterate sits at radius 12.273, carries 0.9999 mean probability on its own
   argmax, keeps the character the source already pointed at in 45% of
   positions against 3.7% chance, and has its argmax fixed after two descent
   steps.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import _style
from _style import (INK, INK2, MUTED, ROLE_ATTRACTOR, ROLE_DATA, ROLE_FIELD,
                    ROLE_SHELL)


def r_blob(theta, scale):
    return scale * (1 + 0.16 * np.sin(2 * theta + 0.7) + 0.07 * np.cos(3 * theta))


def field_arrow(ax, p0, p1, lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                 lw=lw, color=ROLE_FIELD, zorder=5,
                                 shrinkA=0, shrinkB=0))


def mu_marker(ax, x, y):
    ax.plot([x], [y], marker="o", ms=7.5, color=ROLE_ATTRACTOR,
            mec="white", mew=1.2, zorder=6)


def main():
    _style.apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(_style.TEXTWIDTH_IN, 2.85),
                                   gridspec_kw={"wspace": 0.18})
    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.55, 1.55)
        ax.axis("off")

    theta = np.linspace(0, 2 * np.pi, 400)

    # ---- Left: continuous images -------------------------------------
    for scale in (0.42, 0.70, 0.98):
        axL.plot(r_blob(theta, scale) * np.cos(theta),
                 r_blob(theta, scale) * np.sin(theta),
                 color=MUTED, lw=0.7, zorder=2)

    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 9:
        th = rng.uniform(0, 2 * np.pi)
        rr = np.sqrt(rng.random()) * 0.78 * r_blob(th, 0.98)
        q = np.array([rr * np.cos(th), rr * np.sin(th)])
        if np.hypot(q[0] - 0.05, q[1] - 0.02) < 0.34:  # keep mu_1 clear
            continue
        if any(np.linalg.norm(q - w) < 0.40 for w in pts):
            continue
        pts.append(q)
    pts = np.array(pts)

    # every image is a point of zero gradient: draw the well around each one
    for q in pts:
        axL.plot(q[0] + 0.13 * np.cos(theta), q[1] + 0.13 * np.sin(theta),
                 color=MUTED, lw=0.55, zorder=3)
    axL.plot(pts[:, 0], pts[:, 1], ls="", marker="o", ms=3.6,
             color=ROLE_DATA, zorder=4)

    mu_marker(axL, 0.05, 0.02)
    axL.annotate(r"$\mu_1$", xy=(0.11, 0.05), xytext=(0.34, 0.16), fontsize=9,
                 color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=2, shrinkB=2))

    # first stage: the coarse basin, the field pointing in towards mu_1
    for j in range(7):
        th = 2 * np.pi * j / 7 + 0.2
        r0 = 1.30 * r_blob(th, 0.98)
        p0 = np.array([r0 * np.cos(th), r0 * np.sin(th)])
        p1 = p0 - 0.24 * p0 / np.linalg.norm(p0)
        field_arrow(axL, p0, p1)

    # second stage: inside the basin the field points into the nearest image
    top = pts[np.argmax(pts[:, 1])]
    for k in range(3):
        th = 2 * np.pi * k / 3 + 0.6
        u = np.array([np.cos(th), np.sin(th)])
        field_arrow(axL, top + 0.30 * u, top + 0.16 * u, lw=0.8)

    axL.annotate("a learnable basin", xy=(-0.50, 0.82), xytext=(-1.55, 1.30),
                 fontsize=8.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=2, shrinkB=2))
    axL.annotate(r"$\nabla E=0$ at each image",
                 xy=tuple(top + np.array([0.10, 0.16])), xytext=(0.10, 1.30),
                 fontsize=8.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=2, shrinkB=2))
    axL.text(0, -1.40, "the field converges onto the data",
             fontsize=9, color=INK, ha="center")

    # ---- Right: the 2-simplex -----------------------------------------
    A, B, C = np.array([-1.0, -0.85]), np.array([1.0, -0.85]), np.array([0.0, 0.882])
    M = (A + B + C) / 3
    tri = np.array([A, B, C, A])
    axR.plot(tri[:, 0], tri[:, 1], color=INK, lw=0.8, zorder=2)
    for v in (A, B, C):
        axR.plot([v[0]], [v[1]], marker="o", ms=6, color=ROLE_DATA,
                 mec="white", mew=0.8, zorder=4)
    axR.text(-1.13, -0.99, r"$e_1$", fontsize=8, color=INK2, ha="center")
    axR.text(1.13, -0.99, r"$e_2$", fontsize=8, color=INK2, ha="center")
    axR.text(0.16, 0.97, r"$e_3$", fontsize=8, color=INK2, ha="left")

    # watershed: the medians divide the simplex into one basin per vertex
    for v in (A, B, C):
        opp = [w for w in (A, B, C) if not np.allclose(w, v)]
        mid = (opp[0] + opp[1]) / 2
        axR.plot([M[0], mid[0]], [M[1], mid[1]], color=MUTED, lw=0.7,
                 ls=(0, (3, 2.5)), zorder=2)

    # the source: where the sampler starts, on the ridge between the basins
    axR.plot([M[0]], [M[1]], marker="o", ms=7.5, color=ROLE_SHELL,
             mec="white", mew=1.2, zorder=7)
    axR.text(-0.10, -0.36, r"$x_0$", fontsize=9, color=INK, ha="right", va="top")

    # the unigram tilt: a fifth of the way from the centroid out to a vertex
    u_mu = (B - M) / np.linalg.norm(B - M)
    axR.add_patch(FancyArrowPatch(M, M + 0.30 * u_mu, arrowstyle="-|>",
                                  mutation_scale=8, lw=1.2,
                                  color=ROLE_ATTRACTOR, zorder=6,
                                  shrinkA=0, shrinkB=0))
    axR.annotate(r"$\mu_1$", xy=tuple(M + 0.30 * u_mu), xytext=(0.62, -0.14),
                 fontsize=9, color=ROLE_ATTRACTOR, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=ROLE_ATTRACTOR, lw=0.6,
                                 shrinkA=2, shrinkB=2))

    # the descent: it leaves the source and sharpens to one of the vertices
    axR.add_patch(FancyArrowPatch(M + 0.34 * u_mu, B - 0.10 * u_mu,
                                  arrowstyle="-|>", mutation_scale=9, lw=1.1,
                                  color=ROLE_FIELD, zorder=5,
                                  connectionstyle="arc3,rad=-0.22",
                                  shrinkA=0, shrinkB=0))

    # the same field elsewhere: it points at whichever vertex is nearest
    u_a = (A - M) / np.linalg.norm(A - M)
    for r0 in (0.52, 0.82):
        p0 = M + r0 * u_a
        axR.add_patch(FancyArrowPatch(p0, p0 + 0.22 * u_a, arrowstyle="-|>",
                                      mutation_scale=8, lw=1.0,
                                      color=ROLE_FIELD, zorder=5,
                                      shrinkA=0, shrinkB=0))

    axR.text(-1.58, 1.30, "one basin per vertex, and the\nsource on the ridge between them",
             fontsize=8.5, color=INK, ha="left", va="center", linespacing=1.35)
    axR.text(0, -1.40, "the descent sharpens the source into a vertex",
             fontsize=9, color=INK, ha="center")

    _style.save(fig, "image_text_geometry")


if __name__ == "__main__":
    main()
