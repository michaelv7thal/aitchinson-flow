"""Phase H visualization — auditor-driven generation NLL distributions.

Three panels:

1. **Per-position NLL distributions** (violin) for clean, EqM-generated,
   and random-slot baseline. Shows the gap to natural text.
2. **NLL across hyperparameters**: bar chart of mean NLL at varying σ
   and NFE, with the protocol's F3 success/partial/fail thresholds.
3. **Sample text grid**: side-by-side clean vs generated for the first
   four chunks. (Rendered as text on the figure.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx-json",   default="runs/phaseH_audited.json")
    p.add_argument("--logit-json", default="runs/phaseH_audited_logit.json")
    p.add_argument("--out",        default="runs/phaseH_audited.png")
    args = p.parse_args(argv)

    ctx = json.loads(Path(args.ctx_json).read_text())
    logit = json.loads(Path(args.logit_json).read_text())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel 1: NLL bar chart with thresholds.
    ax = axes[0]
    rows = [
        ("Clean WikiText",      ctx["nll_clean_mean"], "tab:green"),
        ("EqM ctx (gen)",       ctx["nll_gen_mean"],   "tab:blue"),
        ("EqM logit-only (gen)", logit["nll_gen_mean"], "tab:cyan"),
        ("Random slot",         ctx["nll_random_mean"],"tab:gray"),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(5.5, color="green", linestyle="--", linewidth=0.8,
               label="F3 success (NLL ≤ 5.5)")
    ax.axhline(7.0, color="orange", linestyle=":", linewidth=0.8,
               label="F3 partial (NLL ≤ 7.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("Mean per-token NLL under GPT-2 (lower = more natural)")
    ax.set_title("Phase H — LM validity of auditor-driven samples")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(values) * 1.18)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: text comparisons (top 4 samples).
    ax = axes[1]
    ax.axis("off")
    samples = ctx.get("samples", [])[:4]
    text_lines = []
    text_lines.append("Side-by-side text comparison (EqM ctx auditor)\n")
    text_lines.append("─" * 80)
    for s in samples:
        text_lines.append(f"\n[chunk {s['idx']}]")
        clean_short = (s["clean"][:120] + "…") if len(s["clean"]) > 120 else s["clean"]
        gen_short = (s["generated"][:120] + "…") if len(s["generated"]) > 120 else s["generated"]
        text_lines.append(f"  CLEAN:     {clean_short}")
        text_lines.append(f"  GENERATED: {gen_short}")
    ax.text(0.0, 1.0, "\n".join(text_lines), fontsize=8, family="monospace",
            verticalalignment="top", transform=ax.transAxes)
    ax.set_title("Decoded samples (first 4 chunks)")

    fig.suptitle(
        "Phase H — auditor-driven generation: F3 verdict",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
