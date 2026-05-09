"""Aggregate AUROC numbers from the DFM auditor three-way ablation runs.

Reads each run's ``summary.json`` and prints a markdown table comparing
context modes side-by-side. Used to produce the headline table for the
results writeup.

Usage::

    python scripts/compile_dfm_auditor_results.py runs/dfm_auditor_d512_off \
        runs/dfm_auditor_d512_hidden runs/dfm_auditor_d512_both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def best_epoch(history: list[dict]) -> dict:
    return max(history, key=lambda h: h["auc_E_seq_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="run directories with summary.json")
    parser.add_argument("--out", default=None, help="optional markdown output path")
    args = parser.parse_args()

    rows: list[dict] = []
    for d in args.run_dirs:
        path = Path(d) / "summary.json"
        if not path.exists():
            print(f"!! missing {path}")
            continue
        with open(path) as f:
            s = json.load(f)
        be = best_epoch(s["history"])
        last = s["history"][-1]
        rows.append({
            "run": Path(d).name,
            "mode": s["cfg"].get("context_features", "?"),
            "params_M": s["cfg"].get("n_params_million", float("nan")),
            "epochs": s["cfg"].get("epochs", float("nan")),
            "best_auc_E_mean": be["auc_E_seq_mean"],
            "best_epoch": be["epoch"],
            "last_auc_E_mean": last["auc_E_seq_mean"],
            "last_train_ce": last["train_ce"],
            "auc_DeltaE": last["auc_DeltaE_seq"],
            "ece": last["ece_E_seq_mean"],
        })

    md = ["## DFM auditor — three-way mode ablation\n"]
    md.append(
        "| mode | params (M) | epochs | best AUROC | (epoch) | last train CE | "
        "ΔE AUROC | ECE |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| `{r['mode']}` | {r['params_M']:.1f} | {r['epochs']} | "
            f"**{r['best_auc_E_mean']:.4f}** | {r['best_epoch']} | "
            f"{r['last_train_ce']:.3f} | {r['auc_DeltaE']:.3f} | {r['ece']:.3f} |"
        )
    out = "\n".join(md)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
