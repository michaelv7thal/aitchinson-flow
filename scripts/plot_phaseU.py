"""Phase U plot — grouped bar chart of the four-AUROC table per method.

Locality-clean methods (SE, top-K entropy) → flat bars across columns.
Cascade-contaminated methods (linear probe, BLR-Laplace, SVGP, EqM
auditor, SAPLMA) → AUROC₂ ≈ AUROC₁ (cascade fires across positions) and
AUROC₃, AUROC₄ near 0.5 (shuffle destroys the cascade).

Usage:
    python scripts/plot_phaseU.py runs/capstone/U/phaseU_cascade_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COL_ORDER = (
    "AUROC1_corrupted_aligned",
    "AUROC2_uncorrupted_aligned",
    "AUROC3_corrupted_shuffled",
    "AUROC4_uncorrupted_shuffled",
)
COL_LABELS = ("corrupted\naligned", "uncorrupted\naligned", "corrupted\nshuffled", "uncorrupted\nshuffled")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    src = Path(args.input)
    data = json.loads(src.read_text())
    methods = list(data.keys())
    n_meth = len(methods)
    n_cols = len(COL_ORDER)

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * n_meth), 5))
    width = 0.85 / n_cols
    x = np.arange(n_meth)
    for j, key in enumerate(COL_ORDER):
        vals = [data[m].get(key, float("nan")) for m in methods]
        ax.bar(x + (j - (n_cols - 1) / 2) * width, vals, width=width, label=COL_LABELS[j])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Phase U — cascade-audit four-AUROC table")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(args.out) if args.out else src.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
