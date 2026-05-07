"""Phase B + Phase E summary plots.

Two panels:

1. **Phase B / parity-compute KL_bi bar chart.** All four methods at the
   ``data_50k_ep5`` platform, plus the LogitKLFlow NFE scan, with the
   protocol's "acceptable" / "fail" thresholds drawn.
2. **Phase E BPC overlay.** Per-method bar chart with the SFM/SEDD
   published numbers for context. The continuous-model surrogates are
   marked as "not directly comparable" via a hatched fill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CKPTS_KL = [
    ("eqm_data50k_ep5_v2",   "EqM (NAG)"),
    ("dfm_data50k_ep5_v2",   "DFM"),
    ("fmclr_data50k_ep5_v2", "FMonCLR"),
    ("lkflow_data50k_ep5",   "LogitKLFlow"),
]

NFE_RUNS = [
    ("lkflow_data50k_ep5/eval_nfe32.json",  "LogitKLFlow NFE=32"),
    ("lkflow_data50k_ep5/eval.json",        "LogitKLFlow NFE=64"),
    ("lkflow_data50k_ep5/eval_nfe128.json", "LogitKLFlow NFE=128"),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/phaseB_E_summary.png")
    args = p.parse_args(argv)

    runs_root = Path("runs")

    # -- Phase B / cross-method KL_bi --
    main_methods: list[tuple[str, float]] = []
    for ck, label in CKPTS_KL:
        p_ev = runs_root / ck / "eval.json"
        if not p_ev.exists():
            continue
        d = json.loads(p_ev.read_text())
        main_methods.append((label, float(d.get("bigram_kl", float("nan")))))

    # NFE scan rows
    nfe_rows: list[tuple[str, float]] = []
    for sub, label in NFE_RUNS:
        p_ev = runs_root / sub
        if not p_ev.exists():
            continue
        d = json.loads(p_ev.read_text())
        nfe_rows.append((label, float(d.get("bigram_kl", float("nan")))))

    # -- Phase E / BPC table --
    bpc_rows: list[tuple[str, float, str]] = []  # label, bpc, method
    for ck, label in CKPTS_KL:
        p_bpc = runs_root / ck / "bpc.json"
        if not p_bpc.exists():
            continue
        d = json.loads(p_bpc.read_text())
        bpc_rows.append((label, float(d["bpc"]), d.get("method", "?")))

    # Published-baseline references for the BPC panel.
    published = [
        ("SFM (published, 50ep)",  1.39, "discrete_elbo"),
        ("SEDD (published, 50ep)", 1.32, "discrete_elbo"),
        ("Optimal (text8)",        1.40, "reference"),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Panel 1: KL_bi bar chart
    ax = axes[0]
    labels = [l for l, _ in main_methods] + [l for l, _ in nfe_rows]
    values = [v for _, v in main_methods] + [v for _, v in nfe_rows]
    colors = (
        ["tab:blue"] * len([l for l, _ in main_methods if "DFM" not in l])
        + ["tab:green"] * 1
        + []
    )
    # Color: DFM green, others blue, NFE scan gray
    bar_colors: list[str] = []
    for label, _ in main_methods:
        if "DFM" in label:
            bar_colors.append("tab:green")
        elif "EqM" in label:
            bar_colors.append("tab:blue")
        elif "FMonCLR" in label:
            bar_colors.append("tab:orange")
        elif "LogitKLFlow" in label:
            bar_colors.append("tab:red")
        else:
            bar_colors.append("tab:gray")
    bar_colors += ["lightgray"] * len(nfe_rows)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("KL_bi (bigram KL vs train)")
    ax.set_title("Phase B — KL_bi at parity compute (data_50k_ep5)")
    # Threshold lines
    ax.axhline(0.50, color="green", linestyle="--", linewidth=0.8, label="protocol target (≤0.50)")
    ax.axhline(0.30, color="darkgreen", linestyle=":", linewidth=0.8, label="\"acceptable\" (≤0.30)")
    ax.axhline(1.00, color="red", linestyle=":", linewidth=0.8, label="protocol fail (>1.00)")
    # Annotate bars
    for rect, val in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.04,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) * 1.18)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: BPC bar chart with published references
    ax = axes[1]
    all_bpc = bpc_rows + published
    labels = [l for l, _, _ in all_bpc]
    values = [v for _, v, _ in all_bpc]
    methods = [m for _, _, m in all_bpc]

    colors_bpc = []
    hatches = []
    for label, val, method in all_bpc:
        if method == "discrete_elbo":
            colors_bpc.append("tab:green")
            hatches.append("")
        elif method == "reference":
            colors_bpc.append("tab:gray")
            hatches.append("")
        elif method == "implied_x1_ce_surrogate":
            colors_bpc.append("tab:blue")
            hatches.append("//")
        elif method == "clean_logit_ce":
            colors_bpc.append("tab:red")
            hatches.append("//")
        else:
            colors_bpc.append("lightgray")
            hatches.append("")

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors_bpc, edgecolor="black", linewidth=0.6)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("BPC (bits per character)")
    ax.set_title("Phase E — BPC overlay (text8 test split)\nhatched = continuous-model surrogate, not comparable")
    for rect, val in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.06,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Phase B + E — KL_bi at parity compute and BPC overlay (text8)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot_phaseB_E] wrote {out_path}")


if __name__ == "__main__":
    main()
