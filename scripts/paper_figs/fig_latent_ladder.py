"""fig:latent-ladder — linear separability of the frozen backbone's feature
space across the whole corruption ladder (paper app:latent-ladder).

The main text reads the geometry at one corruption rate (0.3, tab:latent). This
figure shows every rate, so that operating point can be seen for what it is: a
display choice inside a monotone trend, not a threshold.

Reads the per-rate metric dumps written by ``scripts/plot_latent_split.py``:
  bench_ood/latent_ladder/rate_<r>/            replace, shuffle, falseinfo
  bench_ood/latent_ladder_plausible/rate_<r>/  plausible (the affordable subset:
                                               each swapped slot costs 48
                                               model-scored candidate words)
Missing rates are simply absent from a curve. With --tex the script also prints
the two appendix table bodies, so the paper's numbers are transcribed by copy
rather than by hand.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _style
from _style import BENCH_OOD, BLUE, ORANGE, AQUA, VIOLET, MUTED, BASELINE

LADDERS = [BENCH_OOD / "latent_ladder",
           BENCH_OOD / "latent_ladder_plausible"]
MAIN_RATE = 0.3                      # the operating point of tab:latent

SCHEMES = [("replace", "replace", BLUE),
           ("shuffle", "shuffle", AQUA),
           ("falseinfo", "false information", ORANGE),
           ("plausible", "plausible", VIOLET)]
# (row label, metrics block, supervised-axis key)
GRANS = [("per sequence", "per_sequence", "lda_axis_auroc"),
         ("per token", "per_token", "energy_axis_auroc")]
READOUTS = [("pca_pc12_probe_auroc", "PC1-2 probe"),   # cmr10 has no en dash
            (None, "supervised axis"),          # granularity-dependent key
            ("full_dim_linear_auroc", "full-dimension probe")]


def _load():
    """{scheme: {granularity: {rate: {readout: auroc}}}}, pooled over ladders."""
    out = {}
    for ladder in LADDERS:
        for run in sorted(ladder.glob("rate_*")):
            f = run / "latent_split.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            rate = float(m["rate"])
            for key, _, _ in SCHEMES:
                for _, block, axis_key in GRANS:
                    got = m.get(block, {}).get(key)
                    if not got:
                        continue
                    row = dict(got)
                    row["axis"] = row.get(axis_key)
                    out.setdefault(key, {}).setdefault(block, {})[rate] = row
    return out


def _series(data, key, block, readout, axis_key):
    rates = sorted(data.get(key, {}).get(block, {}))
    field = axis_key if readout is None else readout
    pts = [(r, data[key][block][r].get("axis" if readout is None else field))
           for r in rates]
    pts = [(r, v) for r, v in pts if v is not None]
    return [r for r, _ in pts], [v for _, v in pts]


def fig_ladder(data):
    fig, axes = plt.subplots(2, 3, figsize=(_style.TEXTWIDTH_IN, 4.1),
                             layout="constrained", sharex=True, sharey=True)
    for row, (glabel, block, axis_key) in enumerate(GRANS):
        for col, (readout, rlabel) in enumerate(READOUTS):
            ax = axes[row, col]
            ax.axhline(0.5, color=BASELINE, lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.axvline(MAIN_RATE, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=1)
            for key, _, color in SCHEMES:
                xs, ys = _series(data, key, block, readout, axis_key)
                if not xs:
                    continue
                ax.plot(xs, ys, color=color, lw=1.3, marker="o", ms=2.6,
                        zorder=3)
            if row == 0:
                ax.set_title(rlabel, pad=4)
            if col == 0:
                ax.set_ylabel(f"{glabel}\nAUROC")
            if row == 1:
                ax.set_xlabel("corruption rate")
            ax.set_ylim(0.45, 1.03)
            ax.set_xlim(0, 1.03)
            ax.locator_params(nbins=5)
    shown = [(lab, c) for key, lab, c in SCHEMES if key in data]  # no empty entries
    handles = [Line2D([], [], color=c, lw=1.4, marker="o", ms=3) for _, c in shown]
    fig.legend(handles, [lab for lab, _ in shown], ncol=len(shown),
               loc="outside upper center")
    _style.save(fig, "latent_ladder")


def print_tex(data):
    """The two appendix table bodies, ready to paste into app:latent-ladder."""
    for glabel, block, axis_key in GRANS:
        rates = sorted({r for key, _, _ in SCHEMES
                        for r in data.get(key, {}).get(block, {})})
        print(f"% ---- {glabel} ----")
        for r in rates:
            cells = []
            for key, _, _ in SCHEMES:
                got = data.get(key, {}).get(block, {}).get(r)
                for field in ("pca_pc12_probe_auroc", "axis",
                              "full_dim_linear_auroc"):
                    v = got.get(field) if got else None
                    cells.append("--" if v is None else f"{v:.3f}")
            print(f"\t\t{r:<4} & " + " & ".join(cells) + r" \\")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", action="store_true",
                    help="also print the appendix table bodies")
    args = ap.parse_args()
    data = _load()
    if not data:
        raise SystemExit("no ladder runs found — run drive_latent_all.sh first")
    _style.apply_style()
    fig_ladder(data)
    if args.tex:
        print_tex(data)


if __name__ == "__main__":
    main()
