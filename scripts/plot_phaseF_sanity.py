"""Plot the Phase F sanity bars — EqM auditor AUROC vs zero-train baselines.

Reads ``runs/phaseF_sanity.json`` (and ``..._logit.json``) and renders a
two-panel figure that shows the *narrowness* of the trained auditor's
gap over trivial baselines, plus the cascade-contamination signal at
uncorrupted positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ctx-json",   default="runs/phaseF_sanity.json")
    p.add_argument("--logit-json", default="runs/phaseF_sanity_logit.json")
    p.add_argument("--out",        default="runs/phaseF_sanity.png")
    args = p.parse_args(argv)

    ctx = json.loads(Path(args.ctx_json).read_text())
    logit = json.loads(Path(args.logit_json).read_text())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Panel 1: AUROC vs trivial baselines on val split ----
    ax = axes[0]
    rows = [
        ("EqM ctx (val)",            ctx["1_train_vs_val"]["val"]["Tok_EqM_Upos"]),
        ("EqM logit-only (val)",     logit["1_train_vs_val"]["val"]["Tok_EqM_Upos"]),
        ("Spilled Energy (val)",     ctx["1_train_vs_val"]["val"]["Tok_SE"]),
        ("Linear probe on h_LLM",    ctx["2_trivial_baselines_val"]["linear_probe_h"]["test_auroc"]),
        ("Top-K entropy",            ctx["2_trivial_baselines_val"]["tok"]["topk_entropy_tok"]),
        ("Top-K peak gap",           ctx["2_trivial_baselines_val"]["tok"]["topk_peak_gap_tok"]),
        ("h_LLM L2 norm",            ctx["2_trivial_baselines_val"]["tok"]["h_norm_tok"]),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["tab:blue", "tab:cyan", "tab:purple", "tab:olive",
              "tab:gray", "tab:gray", "tab:gray"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.95, color="green", linestyle=":", linewidth=0.8,
               label="F1 Tok target (0.95)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.5, label="chance")
    for r, v in zip(bars, values):
        ax.text(r.get_x() + r.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Tok AUROC (held-out, n=950 corrupted positions)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(
        "Phase F sanity — Tok AUROC of trained auditors vs zero-train baselines\n"
        "(GPT-2 cache, 60 held-out chunks)"
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # ---- Panel 2: Cascade contamination — corrupted vs uncorrupted Tok AUROC ----
    ax = axes[1]
    rows2 = [
        ("EqM ctx (val)",
            ctx["1_train_vs_val"]["val"]["Tok_EqM_Upos"],
            ctx["3_uncorrupted_pos_val"]["EqM_U_pos"]),
        ("EqM logit-only (val)",
            logit["1_train_vs_val"]["val"]["Tok_EqM_Upos"],
            logit["3_uncorrupted_pos_val"]["EqM_U_pos"]),
        ("Spilled Energy (val)",
            ctx["1_train_vs_val"]["val"]["Tok_SE"],
            ctx["3_uncorrupted_pos_val"]["SE_pos"]),
    ]
    labels = [r[0] for r in rows2]
    corrupt_aucs = [r[1] for r in rows2]
    uncorrupt_aucs = [r[2] for r in rows2]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, corrupt_aucs, w, label="at CORRUPTED positions",
                   color="tab:red", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + w/2, uncorrupt_aucs, w, label="at UNCORRUPTED positions",
                   color="tab:gray", edgecolor="black", linewidth=0.6)
    for r, v in zip(bars1, corrupt_aucs):
        ax.text(r.get_x() + r.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for r, v in zip(bars2, uncorrupt_aucs):
        ax.text(r.get_x() + r.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.5, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tok AUROC (held-out)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title(
        "Cascade contamination — does the discriminator localise corruption?\n"
        "Ideal: high AUROC at corrupted, ≈0.5 at uncorrupted."
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Phase F sanity checks — caveats on the F1 numbers",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
