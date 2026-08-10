"""fig:image-text — the image-versus-text geometry behind the exclusivity
result. Left: continuous image data, marginal mean inside the typical set,
a learnable basin. Right: the 2-simplex, data at the one-hot vertices,
mu_1 at the centroid, field pointing away from all data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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

    for radius in (0.15, 0.26, 0.37):
        axR.add_patch(Circle(M, radius, fill=False, ec=MUTED, lw=0.7, zorder=2))

    mu_marker(axR, *M)
    axR.annotate(r"$\mu_1$", xy=(M[0] - 0.06, M[1]), xytext=(-0.55, -0.33),
                 fontsize=9, color=INK, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=2, shrinkB=2))

    # field arrows: everywhere toward the centroid, away from every vertex
    dirs = [(A, (0.90, 0.60)), (B, (0.90, 0.60)),
            ((A + B) / 2, (0.56,)), ((B + C) / 2, (0.56,)), ((C + A) / 2, (0.56,))]
    for target, radii in dirs:
        u = (target - M) / np.linalg.norm(target - M)
        for r0 in radii:
            length = min(0.24, r0 - 0.42)
            p0 = M + r0 * u
            p1 = M + (r0 - length) * u
            field_arrow(axR, p0, p1)

    # Hilbert distance from mu_1 to the nearest vertex, offset off the median
    axR.add_patch(FancyArrowPatch((0.06, M[1] + 0.08), (0.06, 0.80),
                                  arrowstyle="<->", mutation_scale=7,
                                  lw=0.8, color=INK, zorder=4,
                                  shrinkA=0, shrinkB=0))
    axR.text(0.28, 0.66, r"$\Theta(\sqrt{L})$", fontsize=9, color=INK, ha="left")

    axR.annotate("a single\ntilted bowl", xy=(-0.335, -0.117), xytext=(-1.22, 0.48),
                 fontsize=9, color=INK, ha="center", va="center", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=8, shrinkB=2))
    axR.text(0, -1.40, "the field points away from all data",
             fontsize=9, color=INK, ha="center")

    _style.save(fig, "image_text_geometry")


if __name__ == "__main__":
    main()
