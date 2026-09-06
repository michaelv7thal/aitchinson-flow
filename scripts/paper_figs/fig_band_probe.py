"""fig:band-probe — the in-training bigram-divergence probe of the six band
runs at the benchmark budget (paper tab:eqm-band-bench, sweeps/band_L256.yaml).

One line per run, read from runs/<cell>/history.jsonl (one JSON row per epoch,
key `bigram_kl`). The probe is the trainer's own diagnostic: 16 samples, 100
descent steps, after every epoch (training.sample_eval_n / sample_eval_steps in
each run's config.json), so its LEVEL is not comparable to the n=256 / 400-step
eval.json numbers the table prints; only the shape is read. The two numbers the
paper quotes from it (1.51 at epoch 6, 0.52 at epoch 7 of the seed-42 cell) are
rows 6 and 7 of runs/band_L256_ep10_d10k/history.jsonl; check_claims.py does not
read JSONL, so they are not in claims.tsv.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt

import _style
from _style import AQUA, BLUE, GREEN, INK, MUTED, ORANGE, REPO, VIOLET

RUNS = [
    # (label, run dir, color, linestyle, linewidth)
    (r"$\gamma\sim\mathcal{U}[0.005,\,0.030]$, seed 42", "runs/band_L256_ep10_d10k", ORANGE, "-", 2.0),
    (r"$\gamma\sim\mathcal{U}[0.005,\,0.030]$, seed 43", "runs/band_L256_ep10_d10k_s43", ORANGE, "--", 1.4),
    (r"$\gamma\sim\mathcal{U}[0.012,\,0.030]$", "runs/band_L256_ep10_d10k_g012_gs030", BLUE, "-", 1.2),
    (r"$\gamma\sim\mathcal{U}[0,\,0.030]$", "runs/band_L256_ep10_d10k_g0_gs030", GREEN, "-", 1.2),
    (r"$\gamma\sim\mathcal{U}[0.005,\,0.036]$", "runs/band_L256_ep10_d10k_g005_gs99", AQUA, "-", 1.2),
    (r"$\gamma\sim\mathcal{U}[0,\,0.036]$", "runs/band_L256_ep10_d10k_g0", VIOLET, "-", 1.2),
]


def load(relpath):
    rows = [json.loads(l) for l in (REPO / relpath / "history.jsonl").read_text().splitlines() if l.strip()]
    ks = [r["bigram_kl"] for r in rows]
    return list(range(1, len(ks) + 1)), ks


def main():
    _style.apply_style()
    fig, ax = plt.subplots(figsize=(_style.TEXTWIDTH_IN, 2.5))
    ax.grid(axis="y")

    for label, relpath, color, ls, lw in RUNS:
        ep, ks = load(relpath)
        print(f"{relpath:<40} " + " ".join(f"{k:.3f}" for k in ks))
        ax.plot(ep, ks, color=color, ls=ls, lw=lw, marker="o", ms=3.4,
                mec="white", mew=0.6, label=label, zorder=3 if lw > 1.5 else 2)

    ax.axvline(7, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.text(7.08, 2.62, "epoch 7", fontsize=7.5, color=MUTED, va="top")

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"bigram divergence $\mathrm{KL}_{\mathrm{bi}}$")
    ax.set_xlim(0.7, 10.3)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.0, 2.8)
    ax.legend(loc="lower left", ncol=2, fontsize=7.2, handlelength=2.2, labelcolor=INK)

    _style.save(fig, "band_probe")


if __name__ == "__main__":
    main()
