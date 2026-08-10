"""fig:ood-word-auroc — word-level localization AUROC (max-pool) versus
corruption rate for the three synthetic schemes (paper tab:ood-word)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt

import _style
from _style import DETECTORS, INK, MUTED, REPO

BENCH = REPO / "bench_ood"
SCHEMES = [("replace", "replace"), ("shuffle", "shuffle"), ("falseinfo", "false information")]


def load_word_max(relpath):
    rows = json.loads((BENCH / relpath).read_text())["rows"]
    out = {}
    for r in rows:
        if r.get("rate", 0) > 0 and r.get("scheme") in {s for s, _ in SCHEMES} and "auroc_word_max" in r:
            out.setdefault(r["scheme"], []).append((r["rate"], r["auroc_word_max"]))
    return {k: sorted(v) for k, v in out.items()}


def main():
    _style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(_style.TEXTWIDTH_IN, 2.45),
                             sharey=True, layout="constrained")
    handles, labels = [], []
    for det_label, relpath, _sfx, color in DETECTORS:
        data = load_word_max(relpath)
        hero = det_label == "NLL"
        for ax, (scheme, _tag) in zip(axes, SCHEMES):
            rates, vals = zip(*data[scheme])
            (ln,) = ax.plot(rates, vals, color=color,
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
    axes[0].set_ylabel("word-level AUROC (max-pool)")
    axes[0].text(0.98, 0.508, "chance", fontsize=7, color=MUTED,
                 ha="right", va="bottom", transform=axes[0].get_yaxis_transform())
    fig.supxlabel("corruption rate", fontsize=9)
    fig.legend(handles, labels, ncol=7, loc="outside upper center")

    _style.save(fig, "ood_word_auroc_max")


if __name__ == "__main__":
    main()
