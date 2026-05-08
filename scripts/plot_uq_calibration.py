"""Calibration + AUROC bar chart for a Phase K / L UQ eval JSON.

Reads ``runs/<name>/uq_eval.json`` (produced by ``scripts/eval_uq.py``)
and emits ``runs/<name>/uq_calibration.png`` with two panels:

  Panel A  — AUROC bar chart per method (predictive mean as score).
  Panel B  — ECE bar chart per method that exposes calibrated probs
             (linear probe, BLR-Laplace, ensemble, SVGP).

Usage:
    python scripts/plot_uq_calibration.py \\
        --json runs/hal_gpt2_qa_meanpool/uq_eval.json \\
        --out runs/hal_gpt2_qa_meanpool/uq_calibration.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_method_metrics(summary: dict) -> tuple[list[str], list[float], list[float]]:
    """Return (names, aucs, eces). aucs/eces aligned 1:1; ECE is NaN where
    the method has no calibrated probability.
    """
    rows: list[tuple[str, float, float]] = []
    if "linear_probe" in summary:
        rows.append((
            "Linear",
            summary["linear_probe"].get("auc", float("nan")),
            summary["linear_probe"].get("ece", float("nan")),
        ))
    if "mahalanobis" in summary:
        rows.append((
            "Mahalanobis",
            summary["mahalanobis"].get("auc", float("nan")),
            float("nan"),
        ))
    if "blr_laplace" in summary:
        rows.append((
            "BLR-Laplace",
            summary["blr_laplace"].get("auc_prob", float("nan")),
            summary["blr_laplace"].get("ece", float("nan")),
        ))
    if "ensemble_5" in summary:
        rows.append((
            "Ensemble×5",
            summary["ensemble_5"].get("auc_mean", float("nan")),
            summary["ensemble_5"].get("ece", float("nan")),
        ))
    if "svgp" in summary:
        rows.append((
            "SVGP",
            summary["svgp"].get("auc_prob", float("nan")),
            summary["svgp"].get("ece", float("nan")),
        ))
    if "spilled_energy_seq" in summary:
        rows.append((
            "SE (zero-train)",
            summary["spilled_energy_seq"].get("auc", float("nan")),
            float("nan"),
        ))
    names = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    eces = [r[2] for r in rows]
    return names, aucs, eces


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="path to uq_eval.json")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument(
        "--auroc-target",
        type=float,
        default=0.85,
        help="protocol's K-pass AUROC threshold (drawn as a horizontal line)",
    )
    p.add_argument(
        "--ece-target",
        type=float,
        default=0.10,
        help="protocol's K-pass ECE threshold (drawn as a horizontal line)",
    )
    args = p.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; cannot render", file=sys.stderr)
        sys.exit(1)

    summary = json.loads(Path(args.json).read_text())
    names, aucs, eces = _extract_method_metrics(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    xs = list(range(len(names)))
    axes[0].bar(xs, aucs, color="#3a7ca5")
    axes[0].axhline(args.auroc_target, color="red", linestyle="--", linewidth=1,
                    label=f"target ≥ {args.auroc_target:.2f}")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("AUROC (predictive mean)")
    axes[0].set_title("Detection AUROC by method")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(aucs):
        if v == v:  # not NaN
            axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ECE panel — only methods with a calibrated prob
    ece_xs = [i for i, e in enumerate(eces) if e == e]
    ece_vals = [eces[i] for i in ece_xs]
    ece_names = [names[i] for i in ece_xs]
    axes[1].bar(range(len(ece_xs)), ece_vals, color="#a55a3a")
    axes[1].axhline(args.ece_target, color="red", linestyle="--", linewidth=1,
                    label=f"target ≤ {args.ece_target:.2f}")
    axes[1].set_xticks(range(len(ece_xs)))
    axes[1].set_xticklabels(ece_names, rotation=20, ha="right")
    axes[1].set_ylabel("ECE (lower = better)")
    axes[1].set_title("Calibration by method")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, axis="y", which="both")
    for i, v in enumerate(ece_vals):
        axes[1].text(i, v * 1.15, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"UQ on {summary.get('lm', '?')} — pool={summary.get('pool', '?')}, "
        f"n_train_pos={summary.get('n_train_pos', '?')} / n_val_pos={summary.get('n_val_pos', '?')}"
    )
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
