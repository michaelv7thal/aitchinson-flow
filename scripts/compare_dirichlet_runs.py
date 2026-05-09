"""Cross-run side-by-side for the Dirichlet experiments.

Reads a list of ``runs/<name>/`` directories, pulls the final-epoch row from
each ``history.jsonl`` plus the post-training ``eval.json``, and prints a
table on the metrics that matter for the capstone (flow_loss, γ-bucket
losses, KL_uni/bi/tri, H_ratio).

Usage:
    python scripts/compare_dirichlet_runs.py runs/baseline_5ep \\
        runs/dphase3_mse_dirichlet runs/dphase3_mse_deterministic_v2

The first row is treated as the reference; subsequent rows print Δ vs ref.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "flow_loss",
    "ce",
    "g<.33",
    "g<.66",
    "g<1",
    "loss",
    "unigram_kl",
    "bigram_kl",
    "trigram_kl",
    "H_gen",
    "H_gt",
    "H_ratio",
)


def _last_history_row(run_dir: Path) -> dict[str, float]:
    p = run_dir / "history.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(line) for line in p.read_text().strip().splitlines() if line.strip()]
    return rows[-1] if rows else {}


def _eval_dict(run_dir: Path) -> dict[str, float]:
    p = run_dir / "eval.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _row_for(run_dir: Path) -> dict[str, float]:
    h = _last_history_row(run_dir)
    e = _eval_dict(run_dir)
    out: dict[str, float] = {}
    for k in METRICS:
        if k in e:
            out[k] = float(e[k])
        elif k in h:
            out[k] = float(h[k])
    if "H_gen" in h and "H_gt" in h and "H_ratio" not in out:
        out["H_ratio"] = float(h["H_gen"]) / float(h["H_gt"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories to compare")
    ap.add_argument("--metrics", type=str, default=None,
                    help="comma-separated subset of metrics to display")
    args = ap.parse_args()

    keys = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else list(METRICS)
    )

    rows = [(Path(r).name, _row_for(Path(r))) for r in args.runs]
    if not rows:
        return 1

    name_w = max(len(name) for name, _ in rows)
    header = "run".ljust(name_w) + " | " + " | ".join(k.rjust(11) for k in keys)
    print(header)
    print("-" * len(header))

    ref_name, ref = rows[0]
    print(ref_name.ljust(name_w) + " | "
          + " | ".join(f"{ref.get(k, float('nan')):>11.5g}" for k in keys))
    for name, r in rows[1:]:
        line_v = name.ljust(name_w) + " | " + " | ".join(
            f"{r.get(k, float('nan')):>11.5g}" for k in keys
        )
        print(line_v)
        deltas = []
        for k in keys:
            if k in r and k in ref:
                d = r[k] - ref[k]
                deltas.append(f"{d:>+11.4g}")
            else:
                deltas.append(" " * 11)
        print(("Δ vs " + ref_name).ljust(name_w) + " | " + " | ".join(deltas))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
