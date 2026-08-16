"""fig:heal-compare — localize-then-inpaint repair of replace (geometric)
versus false-information (contextual) corruption, three localizers
(paper tab:heal-synth). Operating point: least-damaging swept FPR
(= max net_per_corrupt, which is FPR 0.02 for every arm)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt

import _style
from _style import BASELINE, BENCH_HEAL, BLUE, INK, ORANGE

BENCH = BENCH_HEAL
LOCALIZERS = [("NLL", "nll"), ("BLR", "blr"), ("BGMM", "bgmm")]
SCHEMES = [("replace (geometric)", "replace", BLUE),
           ("false information (contextual)", "falseinfo", ORANGE)]
PANELS = [("net recovery / corrupt token", "net_per_corrupt"),
          (r"localization $F_1$", "loc_f1"),
          ("localization precision", "loc_precision")]


def best_row(loc, scheme):
    data = json.loads((BENCH / f"{loc}_{scheme}.json").read_text())
    return max(data["sweep"], key=lambda s: s["net_per_corrupt"])


def main():
    _style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(_style.TEXTWIDTH_IN, 2.35),
                             layout="constrained")
    x = np.arange(len(LOCALIZERS))
    width, off = 0.32, 0.17

    rows = {(loc, sch): best_row(loc, sch)
            for _, loc in LOCALIZERS for _, sch, _c in SCHEMES}
    for k, r in rows.items():
        print(k, {m: round(r[m], 3) for _, m in PANELS}, "fpr", r["target_fpr"])

    handles = []
    for ax, (title, metric) in zip(axes, PANELS):
        for j, (sch_label, sch, color) in enumerate(SCHEMES):
            vals = [rows[(loc, sch)][metric] for _, loc in LOCALIZERS]
            bars = ax.bar(x + (j * 2 - 1) * off, vals, width, color=color, lw=0)
            if metric == "net_per_corrupt":
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2,
                            v + (0.014 if v >= 0 else -0.016),
                            f"${v:+.2f}$",
                            fontsize=7, color=INK, ha="center",
                            va="bottom" if v >= 0 else "top")
        if ax is axes[0]:
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SCHEMES]
        ax.set_title(title, color=INK, pad=4)
        ax.set_xticks(x, [lab for lab, _ in LOCALIZERS])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y")
        if metric == "net_per_corrupt":
            ax.axhline(0.0, color=BASELINE, lw=0.8, zorder=1)
            ax.set_ylim(-0.21, 0.50)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_ylim(0.0, 0.88)
    fig.legend(handles, [lab for lab, _, _ in SCHEMES], ncol=2,
               loc="outside upper center")

    _style.save(fig, "heal_compare")


if __name__ == "__main__":
    main()
