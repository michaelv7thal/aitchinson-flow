"""fig:image-text — the image-versus-text geometry behind the exclusivity
result. Left: continuous image data, marginal mean inside the typical set,
a learnable basin. Right: the 2-simplex, data at the one-hot vertices,
mu_1 at the centroid, which is the watershed between the vertex basins.

The right panel used to draw concentric bowl contours at the centroid with the
field pointing inwards, captioned "a single tilted bowl".  The measurement in
fig:energy-landscape falsifies that: the trained field points AT the nearest
vertex from anywhere with signal, and every vertex is a basin.  What is special
about mu_1 is not that it is the only attractor but that it is equidistant from
all of them -- Theta(sqrt(L)) from each -- so it is the one place where the
field has no nearest vertex to name, and that is exactly where the sampler
starts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import _style
from _style import INK, INK2, MUTED, ROLE_ATTRACTOR, ROLE_DATA, ROLE_FIELD


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
    while len(pts) < 14:
        th = rng.uniform(0, 2 * np.pi)
        rr = np.sqrt(rng.random()) * 0.8 * r_blob(th, 0.98)
        p = (rr * np.cos(th), rr * np.sin(th))
        if np.hypot(p[0] - 0.05, p[1] - 0.02) > 0.22:  # keep mu_1 visible
            pts.append(p)
    pts = np.array(pts)
    axL.plot(pts[:, 0], pts[:, 1], ls="", marker="o", ms=3.6,
             color=ROLE_DATA, zorder=3)

    mu_marker(axL, 0.05, 0.02)
    axL.annotate(r"$\mu_1$", xy=(0.11, 0.05), xytext=(0.34, 0.16), fontsize=9,
                 color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=2, shrinkB=2))

    for j in range(7):
        th = 2 * np.pi * j / 7 + 0.2
        r0 = 1.30 * r_blob(th, 0.98)
        p0 = np.array([r0 * np.cos(th), r0 * np.sin(th)])
        p1 = p0 - 0.24 * p0 / np.linalg.norm(p0)
        field_arrow(axL, p0, p1)

    axL.annotate("a learnable basin", xy=(-0.50, 0.82), xytext=(-1.52, 1.28),
                 fontsize=9, color=INK, ha="left", va="center",
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

    mu_marker(axR, *M)
    axR.text(0.10, -0.40, r"$\mu_1$", fontsize=9, color=INK, ha="left", va="top")

    # Field arrows towards e1 and e2 only; the e3 direction is left free for the
    # distance measure, which runs up the same median.
    for v in (A, B):
        u = (v - M) / np.linalg.norm(v - M)
        for r0 in (0.52, 0.82):
            p0 = M + r0 * u
            axR.add_patch(FancyArrowPatch(p0, p0 + 0.22 * u, arrowstyle="-|>",
                                          mutation_scale=8, lw=1.0,
                                          color=ROLE_FIELD, zorder=5,
                                          shrinkA=0, shrinkB=0))

    axR.add_patch(FancyArrowPatch((0.0, M[1] + 0.11), (0.0, 0.80),
                                  arrowstyle="<->", mutation_scale=7,
                                  lw=0.8, color=INK, zorder=4,
                                  shrinkA=0, shrinkB=0))
    axR.text(0.13, 0.46, r"$\Theta(\sqrt{L})$", fontsize=9, color=INK, ha="left")

    axR.text(-1.58, 1.30, "one basin per vertex,\nand $\mu_1$ equidistant from all",
             fontsize=8.5, color=INK, ha="left", va="center", linespacing=1.35)
    axR.text(0, -1.40, "the sampler starts where no vertex is nearest",
             fontsize=9, color=INK, ha="center")

    _style.save(fig, "image_text_geometry")


if __name__ == "__main__":
    main()
