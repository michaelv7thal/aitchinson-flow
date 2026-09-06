"""fig:energy-landscape — the field sharpens, it does not transport.

Everything here is measured on the deterministic-CLR fixed-interpolant cell
(runs/compu_mse_det), read along EqM's own path x_gamma = (1-g)*x0 + g*x1.

The claim the section makes is that the one-hot structure re-emerges almost
immediately along the path, so the target is a deterministic function of its
own input over nearly all of it and carries nothing at all at the source.  The
field trained on it learns one rule, sharpen whichever token already dominates,
and that rule moves no probability mass between positions: whatever text the
input happens to contain is taken as valid and made confident.

  strip   the three regimes at true scale: everything that is not plain
          amplification is the first 3% of the path.

  panel A the content.  Fraction of positions whose argmax is the true
          character, before and after the descent.  The two curves sit on top
          of each other everywhere: from gamma >= 0.05 the text is already in
          the input and comes back untouched, and below gamma = 0.03 the
          descent adds no correct characters -- at the source its entire gain
          is 0.038 -> 0.082, which is uniform chance 1/K = 0.037 moving to
          unigram chance sum_k p_k^2 = 0.075.  No context is used anywhere.

  panel B the confidence, and the only thing the descent changes.  Mean mass
          on the true character, before and after, on a log scale.  At
          gamma = 0.05 an input as flat as 0.067 comes back at 0.993; at the
          source the same field returns an equally sharp one-hot on the WRONG
          character, at 1.1e-5, five orders of magnitude below what it
          returns everywhere else.

The x-axis is symlog so that gamma = 0 is on it and the band where the
neighbours would be needed is legible; the strip keeps the proportions honest.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for band_geometry
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import _style
from _style import BASELINE, BLUE, INK, INK2, MUTED, ROLE_FIELD
from band_geometry import A_K, geometry, t_of

RUN = Path(__file__).resolve().parents[2] / "runs" / "compu_mse_det"
FILES = ("recovery_native_path.json", "recovery_native_path_fine.json",
         "native_radius.json", "recovery_native_path_fill.json")

K, EPS, SIGMA = 27, 1e-4, 0.10
G_FLAT = 0.005        # gamma_lo, the trained band's lower edge (A_K = 0.11)
G_DECIDED = 0.03      # gamma*, its equilibrium (A_K = 0.95); paper sec:res-eqm-band
UNIFORM = 1.0 / 27.0
LINTHRESH, LINSCALE = 0.005, 0.62
XLO, XHI = -6e-4, 1.15

BAND = "#f6e3c3"      # the one band where the neighbours would be needed
DEAD = "#ececeb"      # no information left


def load():
    """Merge the three native-path sweeps, keyed by gamma."""
    rows = {}
    for name in FILES:
        f = RUN / name
        if not f.exists():
            raise SystemExit(f"missing {f}; run scripts/recovery_native_path.py")
        for r in json.load(open(f))["rows"]:
            rows.setdefault(round(float(r["gamma"]), 4), {}).update(r)
    return dict(sorted(rows.items()))


def series(rows, key, fn=lambda v: v):
    g = [k for k, r in rows.items() if key in r]
    return np.array(g), np.array([fn(rows[k][key]) for k in g])


def regimes(ax, y0=0.0, y1=1.0):
    ax.axvspan(XLO, G_FLAT, y0, y1, color=DEAD, lw=0, zorder=0)
    ax.axvspan(G_FLAT, G_DECIDED, y0, y1, color=BAND, lw=0, zorder=0)


def main():
    _style.apply_style()
    fig, (axS, axA, axB) = plt.subplots(
        3, 1, figsize=(_style.TEXTWIDTH_IN, 4.3),
        gridspec_kw={"height_ratios": [0.15, 1.1, 1.0], "hspace": 0.30})

    rows = load()
    g_acc, acc_pre = series(rows, "token_acc_perturbed")
    _, acc_post = series(rows, "token_acc")
    g_in, p_in = series(rows, "logp_true_pre", np.exp)
    g_out, p_out = series(rows, "logp_true_post", np.exp)

    # ---- strip: the same three regimes, at true scale ------------------
    axS.set_xlim(0.0, 1.0)
    axS.set_ylim(0, 1)
    axS.axvspan(0.0, G_FLAT, color=DEAD, lw=0)
    axS.axvspan(G_FLAT, G_DECIDED, color=BAND, lw=0)
    for sp in axS.spines.values():
        sp.set_visible(True)
        sp.set_color(BASELINE)
    axS.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axS.grid(False)
    axS.text(0.54, 0.5, r"$97\%$ of the path", fontsize=7.5, color=INK2,
             ha="center", va="center")
    axS.text(-0.004, 2.1, "the path at true scale", fontsize=7.5, color=INK2,
             ha="left", va="center", transform=axS.transAxes)

    # ---- panel A: the content, which the descent does not change --------
    axA.set_xscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
    axA.set_xlim(XLO, XHI)
    axA.set_ylim(-0.05, 1.22)
    regimes(axA, 0.0, 1.0)
    axA.fill_between(g_acc, acc_pre, acc_post, color=ROLE_FIELD, alpha=0.6,
                     lw=0, zorder=3, label="_")
    # the closed-form decode probability of the band derivation
    # (paper app:band-edges): the measured input curve should sit on it
    geo = geometry(K, EPS, SIGMA)
    g_th = np.concatenate([[0.0], np.geomspace(2e-4, 0.95, 90)])
    ak = np.array([A_K(t_of(g, geo["Delta"], SIGMA), K) for g in g_th])
    axA.plot(g_th, ak, color=INK, lw=0.9, zorder=6,
             label=r"closed form $A_K(t(\gamma))$")
    axA.plot(g_acc, acc_pre, color=MUTED, lw=1.5, ls=(0, (4, 2)), zorder=4,
             label=r"the input $x_\gamma$")
    axA.plot(g_acc, acc_post, color=BLUE, lw=1.5, marker="o", ms=2.7, zorder=5,
             label="after the descent")
    axA.set_yticks([0, 0.5, 1.0])
    axA.set_yticklabels(["0", "0.5", "1"])
    axA.tick_params(axis="y", left=True, labelleft=True, labelsize=7)
    axA.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axA.grid(False)
    axA.spines["bottom"].set_visible(False)
    axA.set_ylabel("characters correct", fontsize=8, color=INK2, labelpad=3)

    axA.text(0.0016, 1.09, "no\ninformation", fontsize=7, color=INK2, ha="center",
             va="center", linespacing=1.25)
    axA.text(0.0122, 1.09, "the band $[\\gamma_{\\mathrm{lo}},\\,\\gamma^{\\ast}]$:\nthe neighbours are needed", fontsize=7,
             color=INK2, ha="center", va="center", linespacing=1.25)
    axA.text(0.2, 1.09, "the input already names the token", fontsize=7,
             color=INK2, ha="center", va="center")

    axA.annotate("the descent adds no\ncorrect characters",
                 xy=(0.0138, 0.50), xytext=(0.055, 0.62), fontsize=7.5,
                 color=INK, ha="left", va="center", linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=4, shrinkB=3))
    axA.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02), fontsize=7.5,
               labelcolor=INK2, borderpad=0.2)

    # ---- panel B: the confidence, all the descent changes ---------------
    axB.set_xscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
    axB.set_xlim(XLO, XHI)
    axB.set_yscale("log")
    axB.set_ylim(2.5e-6, 3.0)
    regimes(axB, 0.0, 1.0)
    axB.axhline(UNIFORM, color=BASELINE, lw=0.7, zorder=1)
    axB.plot(g_in, p_in, color=MUTED, lw=1.5, ls=(0, (4, 2)), zorder=3,
             label=r"the input $x_\gamma$")
    axB.plot(g_out, p_out, color=BLUE, lw=1.6, marker="o", ms=2.9, zorder=4,
             label="after the descent")
    axB.minorticks_off()
    axB.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    axB.set_yticklabels([r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$",
                         r"$10^{-2}$", r"$10^{-1}$", "1"])
    # landmark ticks, every one a measured gamma; the paper caption lists the
    # full 18-point grid the markers sit on
    axB.set_xticks([0.0, 0.005, 0.03, 0.05, 0.1, 0.3, 0.9])
    axB.set_xticklabels(["0",
                         "$\\gamma_{\\mathrm{lo}}$\n0.005",
                         "$\\gamma^{\\ast}$\n0.03",
                         "0.05", "0.1", "0.3", "0.9"])
    axB.tick_params(axis="y", left=True, labelleft=True, labelsize=7)
    axB.tick_params(axis="x", which="major", bottom=True, labelbottom=True,
                    labelsize=7, length=2.6, pad=2)
    axB.grid(False)
    axB.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7.2,
               labelcolor=INK2, borderpad=0.2)
    axB.set_ylabel("mean mass on the\ntrue character", fontsize=8, color=INK2,
                   labelpad=3)
    axB.set_xlabel(r"$\gamma$: position along the training path "
                   r"$x_\gamma=(1-\gamma)x_0+\gamma x_1$", fontsize=8.5,
                   labelpad=6)

    axB.text(1.14, UNIFORM * 1.45, r"uniform chance $1/K$", fontsize=7,
             color=MUTED, ha="right", va="bottom")
    axB.annotate("the output is just as sharp,\nbut on the wrong character:\n"
                 r"$1.1\times10^{-5}$ left on the true one",
                 xy=(2.6e-4, 1.35e-5), xytext=(0.016, 2.6e-5), fontsize=7.5,
                 color=INK, ha="left", va="center", linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                 shrinkA=3, shrinkB=3))

    # ---- zoom connectors: strip sliver -> panel A ----------------------
    for xs, xa in ((0.0, XLO), (G_DECIDED, G_DECIDED)):
        fig.add_artist(ConnectionPatch(
            xyA=(xs, 0.0), coordsA=axS.transData,
            xyB=(xa, 1.22), coordsB=axA.transData,
            color=BASELINE, lw=0.6, ls=(0, (2.5, 2))))

    _style.save(fig, "energy_landscape")


if __name__ == "__main__":
    main()
