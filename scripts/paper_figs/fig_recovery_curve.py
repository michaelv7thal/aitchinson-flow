"""fig:recovery-curve — recovery gain vs perturbation for the three
Dirichlet FM training budgets (paper tab:recovery)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt

import _style
from _style import BASELINE, BLUES_ORDINAL, INK, MUTED, REPO

FULL = "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge"
EP30 = "runs/sflm_bench_a100_20g_L256/DirichletFM_ep30_d30k"

# All three arms now carry the 0.1-increment ladder from
# _driver/drive_recovery_fine.sh (one JSON per α); load() falls back to the
# published 5-point recovery.json when that directory is absent or empty.
RUNS = [
    ("10 ep (benchmark)", "runs/sflm_bench_a100_20g_L256/DirichletFM/recovery_fine", BLUES_ORDINAL[0]),
    ("30 ep / 30k windows", f"{EP30}/recovery_fine", BLUES_ORDINAL[1]),
    ("full corpus", f"{FULL}/recovery_fine", BLUES_ORDINAL[2]),
]


def load(relpath):
    """Read (α, Δ) pairs from either a single recovery.json or a directory of
    per-α recovery JSONs written by the fine-ladder driver."""
    path = REPO / relpath
    if path.is_dir():
        files = sorted(path.glob("alpha_*.json"))
        if not files:
            path = path.parent / "recovery.json"
            files = [path]
    else:
        files = [path]
    rows = [r for f in files for r in json.loads(f.read_text())["rows"]]
    pts = sorted((r["alpha"], r["delta"]) for r in rows if r.get("mode") == "recovery")
    return [p[0] for p in pts], [p[1] for p in pts]



def main():
    _style.apply_style()
    fig, ax = plt.subplots(figsize=(_style.TEXTWIDTH_IN, 2.55))
    ax.grid(axis="y")
    ax.axhline(0.0, color=BASELINE, lw=0.7, zorder=1)

    for label, relpath, color in RUNS:
        alphas, deltas = load(relpath)
        print(label, [round(a, 2) for a in alphas], [round(d, 3) for d in deltas])
        ax.plot(alphas, deltas, color=color, lw=1.8, marker="o", ms=4.2,
                mec="white", mew=0.8, zorder=3)

    # direct labels above each line (identity by proximity). Both fine-ladder
    # arms peak at α=0.6 — a maximum the old 5-point grid straddled — so their
    # labels sit above that peak rather than on the apparent 0.5–0.7 plateau.
    ax.text(0.60, 0.350, "full corpus", fontsize=8.5, color=INK, ha="center")
    ax.text(0.60, 0.240, "30 ep / 30k windows", fontsize=8.5, color=INK, ha="center")
    ax.text(0.60, 0.118, "10 ep (benchmark)", fontsize=8.5, color=INK, ha="center")

    ax.set_xlabel(r"perturbation $\alpha$")
    ax.set_ylabel(r"recovery gain $\Delta_\alpha$")
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.022, 0.385)

    _style.save(fig, "recovery_curve")


if __name__ == "__main__":
    main()
