"""Phase R figure — KL_bi vs noise scale α for each of EqM, FMonCLR, LogitKLFlow.

Reads the per-cell eval.json files under ``runs/capstone/R/<name>/`` and
emits a single line plot. The horizontal dashed line is DFM's parity-compute
KL_bi (typically 0.148; configurable via ``--dfm-kl``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/capstone/R")
    p.add_argument("--out", default="runs/capstone/R/phaseR_sde.png")
    p.add_argument("--summary", default="runs/capstone/R/phaseR_sde.json")
    p.add_argument("--dfm-kl", type=float, default=0.148)
    args = p.parse_args(argv)

    root = Path(args.root)
    rows = []
    for evald in sorted(root.glob("*/eval.json")):
        try:
            d = json.loads(evald.read_text())
        except Exception:
            continue
        sk = d.get("sample_kwargs") or {}
        if sk.get("method") != "sde":
            continue
        name = evald.parent.name
        # Family: parse the run name prefix (eqm/fmclr/lkflow).
        family = name.split("_", 1)[0]
        rows.append({
            "name": name,
            "family": family,
            "alpha": float(sk.get("alpha", 0.0)),
            "use_grad": bool(sk.get("use_grad", False)),
            "kl_bi": float(d.get("bigram_kl", float("nan"))),
            "kl_uni": float(d.get("unigram_kl", float("nan"))),
            "n_steps": int(d.get("sample_steps", 0)),
        })

    if not rows:
        print(f"no rows under {root}/*/eval.json")
        return

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        if r["n_steps"] != 128:
            continue  # NFE robustness check is plotted separately if needed
        by_family.setdefault(r["family"], []).append(r)

    for family, fr in by_family.items():
        fr_sorted = sorted(fr, key=lambda x: x["alpha"])
        xs = [x["alpha"] for x in fr_sorted]
        ys = [x["kl_bi"] for x in fr_sorted]
        ax.plot(xs, ys, marker="o", label=family)

    ax.axhline(args.dfm_kl, color="grey", linestyle="--", linewidth=1, label=f"DFM = {args.dfm_kl}")
    ax.set_xlabel("Langevin α (noise scale)")
    ax.set_ylabel("KL_bi")
    ax.set_title("Phase R — KL_bi vs SDE diffusion coefficient")
    ax.legend(frameon=False)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
