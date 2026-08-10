"""fig:energy-landscape — schematic: why EqM fails generation and recovery
identically. 1D energy cross-section along the noise-to-data path with an
informative interior shell, plus a strip showing the mid-alpha recovery
bump sampling exactly that shell."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import _style
from _style import BASELINE, INK, INK2, MUTED, ROLE_ATTRACTOR, ROLE_FIELD, ROLE_SHELL

SHELL = (0.40, 0.72)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def energy(s):
    s = np.asarray(s, dtype=float)
    e = 1.0 - np.exp(-((s - 0.08) ** 2) / 0.045)
    e -= 0.45 * sigmoid((s - 0.56) / 0.045)
    for center, depth in [(0.86, 0.62), (0.93, 0.68), (1.00, 0.75)]:
        e -= depth * np.exp(-((s - center) ** 2) / 1.2e-5)
    return e


def field_arrow(ax, p0, p1, rad=0.0, scale=9, lw=1.1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=scale,
                                 lw=lw, color=ROLE_FIELD, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main():
    _style.apply_style()
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(_style.TEXTWIDTH_IN, 2.7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.14})

    s = np.linspace(-0.02, 1.03, 4000)

    # ---- Panel A: energy cross-section -------------------------------
    for ax in (axA, axB):
        ax.axvspan(*SHELL, color=ROLE_SHELL, alpha=0.10, lw=0, zorder=0)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(False)
    for x0 in SHELL:
        axA.axvline(x0, color=BASELINE, lw=0.6, zorder=1)

    axA.plot(s, energy(s), color=INK, lw=1.8, zorder=3)
    axA.set_xlim(-0.02, 1.03)
    axA.set_ylim(-0.32, 1.25)
    axA.spines["bottom"].set_visible(False)
    axA.text(-0.035, 1.13, r"$E(x)$", fontsize=9, color=INK, ha="right")

    # region labels
    axA.text(0.20, 1.14, "unigram tilt (wide, shallow)", fontsize=8, color=INK2, ha="center")
    axA.text(0.56, 1.14, "usable gradient: the shell", fontsize=8, color=INK2, ha="center")
    axA.text(0.875, 1.14, r"flat: $c(\gamma)\to 0$", fontsize=8, color=INK2, ha="center")

    # mu_1 attractor at the bowl minimum
    axA.plot([0.08], [0.0], marker="o", ms=7, color=ROLE_ATTRACTOR,
             mec="white", mew=1.2, zorder=6)
    axA.text(0.08, -0.20, r"$\mu_1$ (unigram)", fontsize=9, color=INK, ha="center")

    # generation: init in noise, long descent into the bowl
    axA.plot([0.24], [energy(0.24)], marker="o", ms=5.5, color=INK,
             mec="white", mew=1.0, zorder=6)
    field_arrow(axA, (0.235, 0.47), (0.105, 0.05), rad=-0.22)
    axA.text(0.27, 0.56, "generation: init in noise", fontsize=9, color=INK, ha="left")

    # recovery: init on a flat shoulder, almost no force
    axA.plot([0.965], [energy(0.965)], marker="o", ms=5.5, color=INK,
             mec="white", mew=1.0, zorder=6)
    field_arrow(axA, (0.955, 0.585), (0.915, 0.585))
    axA.text(0.99, 0.74, "recovery: init on a shoulder", fontsize=9, color=INK, ha="right")

    # the only data-directed force: tangent arrows on the shell's downhill
    for a, b in [(0.585, 0.635), (0.655, 0.705)]:
        field_arrow(axA, (a, float(energy(a)) + 0.05), (b, float(energy(b)) + 0.05),
                    scale=7, lw=1.0)

    axA.text(0.42, 0.10, "generation and recovery descend\nthe same shell-supported field",
             fontsize=9, color=INK, ha="center", linespacing=1.35)
    axA.annotate("per-token minima: sub-resolution\nspikes with flat shoulders",
                 xy=(0.925, -0.05), xytext=(0.76, -0.30), fontsize=8, color=INK2,
                 ha="right", va="bottom", linespacing=1.35,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=3, shrinkB=1))

    # ---- Panel B: recovery-gain strip --------------------------------
    axB.axhline(0.0, color=BASELINE, lw=0.6, zorder=1)
    axB.plot(s, np.exp(-((s - 0.56) ** 2) / 0.02), color=ROLE_SHELL, lw=1.6, zorder=3)
    axB.set_ylim(-0.15, 1.30)
    axB.text(0.01, 0.80, r"recovery gain $\Delta$", fontsize=8, color=INK2,
             ha="left", transform=axB.transAxes)
    axB.annotate(r"mid-$\alpha$ recovery bump", xy=(0.585, 0.97), xytext=(0.68, 0.80),
                 fontsize=8, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=2, shrinkB=2))
    axB.spines["left"].set_visible(False)
    axB.spines["bottom"].set_color(BASELINE)
    axB.set_xlabel(r"position along the noise $\to$ data path", fontsize=9, labelpad=13)
    axB.text(0.0, -0.32, "noise", fontsize=8, color=MUTED, ha="left",
             transform=axB.get_xaxis_transform())
    axB.text(1.0, -0.32, "data", fontsize=8, color=MUTED, ha="right",
             transform=axB.get_xaxis_transform())

    _style.save(fig, "energy_landscape")


if __name__ == "__main__":
    main()
