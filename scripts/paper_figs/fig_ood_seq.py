"""fig:ood-seq — sequence-level AUROC versus corruption rate for the three
synthetic schemes (paper tab:ood-seq)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt

import _style
from _style import BENCH_OOD, DETECTORS, INK, MUTED

BENCH = BENCH_OOD
SCHEMES = [("replace", "replace"), ("shuffle", "shuffle"), ("falseinfo", "false information")]


def load_seq(relpath, sfx):
    rows = json.loads((BENCH / relpath).read_text())["rows"]
    key = f"auroc_seq_{sfx}"
    out = {}
    for r in rows:
        if r.get("rate", 0) > 0 and r.get("scheme") in {s for s, _ in SCHEMES} and key in r:
            out.setdefault(r["scheme"], []).append((r["rate"], r[key]))
    return {k: sorted(v) for k, v in out.items()}


def main():
    _style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(_style.TEXTWIDTH_IN, 2.45),
                             sharey=True, layout="constrained")
    handles, labels = [], []
    for det_label, relpath, sfx, color in DETECTORS:
        data = load_seq(relpath, sfx)
        hero = det_label in ("NLL", "GPT-2 NLL")
        for ax, (scheme, _tag) in zip(axes, SCHEMES):
            rates, vals = zip(*data[scheme])
            (ln,) = ax.plot(rates, vals, color=color,
                            ls="--" if det_label == "Var" else "-",
                            lw=2.0 if hero else 1.3,
                            marker="o", ms=4.0 if hero else 3.2,
                            mec="white", mew=0.7,
                            zorder=4 if hero else 3)
        handles.append(ln)
        labels.append(det_label)

    for ax, (_scheme, tag) in zip(axes, SCHEMES):
        ax.set_title(tag, color=INK, pad=4)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 1.0])
        ax.set_ylim(0.44, 1.03)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_ylabel("sequence AUROC")
    axes[0].text(0.98, 0.508, "chance", fontsize=7, color=MUTED,
                 ha="right", va="bottom", transform=axes[0].get_yaxis_transform())
    fig.supxlabel("corruption rate", fontsize=9)
    fig.legend(handles, labels, ncol=8, loc="outside upper center")

    _style.save(fig, "ood_seq_auroc")


if __name__ == "__main__":
    main()
