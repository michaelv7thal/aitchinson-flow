"""fig:ood-falseinfo — false information: sequence-level AUROC rises with
the swap rate while per-token AUROC stays flat (paper tab:ood-falseinfo)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt

import _style
from _style import BENCH_OOD, DETECTORS, INK, MUTED

BENCH = BENCH_OOD


def load_falseinfo(relpath, sfx):
    rows = json.loads((BENCH / relpath).read_text())["rows"]
    seq, tok = [], []
    for r in rows:
        if r.get("scheme") != "falseinfo" or r.get("rate", 0) <= 0:
            continue
        seq.append((r["rate"], r[f"auroc_seq_{sfx}"]))
        # native unit: BPE for the GPT-2 rows, never the char-attributed number
        tok_key = f"auroc_token_{sfx}_bpe"
        if tok_key not in r:
            tok_key = f"auroc_token_{sfx}"
        tok.append((r["rate"], r[tok_key]))
    return sorted(seq), sorted(tok)


def main():
    _style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(_style.TEXTWIDTH_IN, 2.55),
                             sharey=True, layout="constrained")
    handles, labels = [], []
    for det_label, relpath, sfx, color in DETECTORS:
        seq, tok = load_falseinfo(relpath, sfx)
        hero = det_label in ("NLL", "GPT-2 NLL")
        for ax, series in zip(axes, (seq, tok)):
            rates, vals = zip(*series)
            (ln,) = ax.plot(rates, vals, color=color,
                            lw=2.0 if hero else 1.3,
                            marker="o", ms=4.0 if hero else 3.2,
                            mec="white", mew=0.7,
                            zorder=4 if hero else 3)
        handles.append(ln)
        labels.append(det_label)

    for ax, tag in zip(axes, ("sequence AUROC", "per-token AUROC (native unit)")):
        ax.set_title(tag, color=INK, pad=4)
        ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 1.0])
        ax.set_ylim(0.38, 1.03)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_ylabel("AUROC")
    axes[0].text(0.98, 0.508, "chance", fontsize=7, color=MUTED,
                 ha="right", va="bottom", transform=axes[0].get_yaxis_transform())
    fig.supxlabel("false-information swap rate", fontsize=9)
    fig.legend(handles, labels, ncol=7, loc="outside upper center")

    _style.save(fig, "ood_falseinfo_seq_tok")


if __name__ == "__main__":
    main()
