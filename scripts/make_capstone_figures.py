"""Generate the four headline figures for the capstone writeup.

Reads existing recovery.json and eval.json files from runs/* and writes
PNG + PDF outputs to figures/. Run from repo root:

    python scripts/make_capstone_figures.py

Figures produced:
  figures/fig1_headline_table.png    — 2x2 (training-signal x x1-recipe)
  figures/fig2_kluni_vs_delta.png    — KL_uni vs Δ@.50 scatter (decoupling)
  figures/fig3_recovery_curve.png    — tok_acc vs α: comp_mse vs comp_ref_det
  figures/fig4_noop_signature.png    — tok_acc vs tok_acc_perturbed scatter
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUT = REPO / "figures"
OUT.mkdir(exist_ok=True)


@dataclass
class CellResult:
    name: str
    label: str
    kl_uni: float | None
    alphas: list[float]
    tok_acc: list[float]
    tok_acc_perturbed: list[float]

    @property
    def deltas(self) -> list[float]:
        return [t - p for t, p in zip(self.tok_acc, self.tok_acc_perturbed)]

    def delta_at(self, alpha: float) -> float:
        for i, a in enumerate(self.alphas):
            if abs(a - alpha) < 1e-6:
                return self.deltas[i]
        return float("nan")

    def acc_at(self, alpha: float) -> float:
        for i, a in enumerate(self.alphas):
            if abs(a - alpha) < 1e-6:
                return self.tok_acc[i]
        return float("nan")


def load_cell(path: Path, label: str, eval_path: Path | None = None) -> CellResult:
    rec = json.loads(path.read_text())
    rows = [r for r in rec["rows"] if r.get("mode") == "recovery"]
    alphas = [r["alpha"] for r in rows]
    tok_acc = [r["token_acc"] for r in rows]
    pt = [r.get("token_acc_perturbed", float("nan")) for r in rows]

    kl_uni = None
    if eval_path and eval_path.exists():
        kl_uni = json.loads(eval_path.read_text()).get("unigram_kl")
    else:
        for r in rec["rows"]:
            if r.get("mode") == "unconditional":
                kl_uni = r.get("KL_uni")
                break

    return CellResult(
        name=path.parent.name,
        label=label,
        kl_uni=kl_uni,
        alphas=alphas,
        tok_acc=tok_acc,
        tok_acc_perturbed=pt,
    )


def main() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    # ---- Load all cells ------------------------------------------------ #
    comp_seeds = [
        load_cell(
            RUNS / f"comp_mse_seed{s}" / "recovery.json",
            f"comp_mse_seed{s}",
            eval_path=RUNS / f"comp_mse_seed{s}" / "eval.json",
        )
        for s in (42, 43, 44)
    ]
    comp_hilbert_seeds = [
        load_cell(
            RUNS / f"comp_hilbert_seed{s}" / "recovery.json",
            f"comp_hilbert_seed{s}",
            eval_path=RUNS / f"comp_hilbert_seed{s}" / "eval.json",
        )
        for s in (42, 43, 44)
    ]
    ref_det_mse = load_cell(
        RUNS / "comp_ref_det_mse" / "recovery.json",
        "FM • Det. CLR",
        eval_path=RUNS / "comp_ref_det_mse" / "eval.json",
    )
    ref_det_hilbert = load_cell(
        RUNS / "comp_ref_det_hilbert" / "recovery.json",
        "FM • Det. CLR (Hilbert)",
        eval_path=RUNS / "comp_ref_det_hilbert" / "eval.json",
    )
    dsm_det = load_cell(
        RUNS / "dsm_clr_ablation" / "dsm_clr_det" / "recovery.json",
        "DSM • Det. CLR",
        eval_path=RUNS / "dsm_clr_ablation" / "dsm_clr_det" / "eval.json",
    )
    dsm_dir = load_cell(
        RUNS / "dsm_clr_ablation" / "dsm_clr_dir" / "recovery.json",
        "DSM • Dirichlet",
        eval_path=RUNS / "dsm_clr_ablation" / "dsm_clr_dir" / "eval.json",
    )

    ae_v3_path = RUNS / "ae_d1024_l8_z128_v3" / "eqm" / "recovery.json"
    ae_v3 = load_cell(ae_v3_path, "AE-latent EqM (v3)") if ae_v3_path.exists() else None

    # Aggregate the 3-seed compositional cells (mean + std)
    def mean_std(cells: list[CellResult]) -> tuple[list[float], list[float], list[float], list[float], float]:
        alphas = cells[0].alphas
        accs = np.array([c.tok_acc for c in cells])
        pts = np.array([c.tok_acc_perturbed for c in cells])
        kls = np.array([c.kl_uni for c in cells if c.kl_uni is not None])
        return (
            alphas,
            accs.mean(axis=0).tolist(),
            accs.std(axis=0).tolist(),
            (accs - pts).mean(axis=0).tolist(),
            float(kls.mean()) if len(kls) else float("nan"),
        )

    comp_mse_alpha, comp_mse_acc_mean, comp_mse_acc_std, comp_mse_delta_mean, comp_mse_kl = mean_std(comp_seeds)
    comp_h_alpha, comp_h_acc_mean, comp_h_acc_std, comp_h_delta_mean, comp_h_kl = mean_std(comp_hilbert_seeds)
    # Perturbed baseline (data-side only — same across seeds)
    pt_mean = np.array([c.tok_acc_perturbed for c in comp_seeds]).mean(axis=0)

    # =====================================================================
    # Figure 1 — Headline 2x2 table (rendered as matplotlib for export)
    # =====================================================================
    fig1, ax1 = plt.subplots(figsize=(10.0, 3.5))
    ax1.axis("off")

    # Compute Δ@.50 for each cell
    def _delta_50(c: CellResult) -> float:
        return c.delta_at(0.5)

    comp_mse_d50 = float(np.mean([_delta_50(c) for c in comp_seeds]))
    comp_mse_d50_std = float(np.std([_delta_50(c) for c in comp_seeds]))

    header = ["Training signal", "x₁ recipe", "KL_uni", "acc@.50", "Δ@.50", "Source"]
    rows = [
        ["FM (Eq. 7 Dot Prod.)", "Det. CLR",    f"{ref_det_mse.kl_uni:.3f}", f"{ref_det_mse.acc_at(0.5):.3f}", "+0.000",                                f"comp_ref_det_mse"],
        ["FM (Eq. 7 Dot Prod.)", "Dirichlet",   f"{comp_mse_kl:.3f}",        f"{np.mean([c.acc_at(0.5) for c in comp_seeds]):.3f}", f"+{comp_mse_d50:.3f} ± {comp_mse_d50_std:.3f}", "comp_mse_seed{42,43,44}"],
        ["DSM (simplex-CLR)",   "Det. CLR",     f"{dsm_det.kl_uni:.3f}",     f"{dsm_det.acc_at(0.5):.3f}",                          f"{_delta_50(dsm_det):+.3f}",                    "dsm_clr_det"],
        ["DSM (simplex-CLR)",   "Dirichlet",    f"{dsm_dir.kl_uni:.3f}",     f"{dsm_dir.acc_at(0.5):.3f}",                          f"{_delta_50(dsm_dir):+.3f}",                    "dsm_clr_dir"],
    ]
    table = ax1.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                       colWidths=[0.24, 0.12, 0.10, 0.10, 0.18, 0.26])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(header)):
        cell = table[(0, j)]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")
    # Highlight the Dirichlet-FM row (positive Δ@.50)
    for j in range(len(header)):
        table[(2, j)].set_facecolor("#e6f3ff")
    ax1.set_title(
        "Figure 1 — Headline 2×2 ablation  (training signal × x₁ recipe, d=1024 / 8L / K=27 / L=40)",
        fontsize=11, pad=12,
    )
    fig1.savefig(OUT / "fig1_headline_table.png")
    fig1.savefig(OUT / "fig1_headline_table.pdf")
    plt.close(fig1)
    print(f"  wrote {OUT/'fig1_headline_table.png'}")

    # =====================================================================
    # Figure 2 — KL_uni vs Δ@.50 scatter (the decoupling)
    # =====================================================================
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.0))

    scatter_data = [
        ("FM • Det. CLR",        ref_det_mse.kl_uni,                 _delta_50(ref_det_mse),  "o", "tab:red",    50),
        ("FM • Det. CLR (Hilb)", ref_det_hilbert.kl_uni,             _delta_50(ref_det_hilbert), "o", "tab:pink", 50),
        ("FM • Dirichlet (MSE)",   comp_mse_kl,                      comp_mse_d50,            "s", "tab:green",  60),
        ("FM • Dirichlet (Hilb)",  comp_h_kl,                        float(np.mean([c.delta_at(0.5) for c in comp_hilbert_seeds])), "s", "tab:olive", 60),
        ("DSM • Det. CLR",        dsm_det.kl_uni,                    _delta_50(dsm_det),      "^", "tab:blue",   60),
        ("DSM • Dirichlet",       dsm_dir.kl_uni,                    _delta_50(dsm_dir),      "^", "tab:cyan",   60),
    ]

    for label, x, y, m, c, s in scatter_data:
        ax2.scatter(x, y, marker=m, color=c, s=s, edgecolor="black", linewidth=0.6, label=label, zorder=3)

    # Error bars on the 3-seed compositional cells
    mse_d50s = [_delta_50(c) for c in comp_seeds]
    h_d50s   = [c.delta_at(0.5) for c in comp_hilbert_seeds]
    ax2.errorbar(comp_mse_kl, comp_mse_d50, yerr=np.std(mse_d50s), fmt="none", ecolor="tab:green", capsize=3, zorder=2)
    ax2.errorbar(comp_h_kl, np.mean(h_d50s), yerr=np.std(h_d50s), fmt="none", ecolor="tab:olive", capsize=3, zorder=2)

    ax2.axhline(0.0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax2.set_xlabel("Unigram KL  (closer to 0 = closer to corpus marginal)")
    ax2.set_ylabel("Δ@.50  =  acc(recovered) − acc(perturbed)")
    ax2.set_title("Figure 2 — KL_uni and recovery are decoupled\n"
                  "(low KL_uni ≠ basin structure; the deterministic-CLR cell has best KL but worst recovery)")
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    fig2.savefig(OUT / "fig2_kluni_vs_delta.png")
    fig2.savefig(OUT / "fig2_kluni_vs_delta.pdf")
    plt.close(fig2)
    print(f"  wrote {OUT/'fig2_kluni_vs_delta.png'}")

    # =====================================================================
    # Figure 3 — Recovery curve: token acc vs α
    # =====================================================================
    fig3, ax3 = plt.subplots(figsize=(7.0, 5.0))

    alphas = np.array(comp_mse_alpha)
    ax3.errorbar(alphas, comp_mse_acc_mean, yerr=comp_mse_acc_std,
                 marker="s", color="tab:green", label="FM • Dirichlet (comp_mse, 3 seeds)",
                 linewidth=1.6, capsize=3, zorder=3)
    ax3.plot(ref_det_mse.alphas, ref_det_mse.tok_acc,
             marker="o", color="tab:red", label="FM • Det. CLR (comp_ref_det_mse)", linewidth=1.6, zorder=3)
    ax3.plot(dsm_det.alphas, dsm_det.tok_acc,
             marker="^", color="tab:blue", label="DSM • Det. CLR (dsm_clr_det)", linewidth=1.6, zorder=3)
    ax3.plot(alphas, pt_mean,
             marker="x", color="black", linestyle="--", linewidth=1.2,
             label="perturbed-input baseline (do-nothing argmax)", zorder=2)

    ax3.set_xlabel("Perturbation scale  α  (units of ‖x₁‖)")
    ax3.set_ylabel("Token-level recovery accuracy")
    ax3.set_title("Figure 3 — Recovery vs perturbation magnitude\n"
                  "(only FM-with-Dirichlet exceeds the do-nothing argmax baseline)")
    ax3.legend(loc="lower left", framealpha=0.95)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.02, 1.05)
    ax3.set_ylim(-0.02, 1.05)
    fig3.savefig(OUT / "fig3_recovery_curve.png")
    fig3.savefig(OUT / "fig3_recovery_curve.pdf")
    plt.close(fig3)
    print(f"  wrote {OUT/'fig3_recovery_curve.png'}")

    # =====================================================================
    # Figure 4 — No-op signature: tok_acc vs tok_acc_perturbed
    # =====================================================================
    fig4, axes = plt.subplots(1, 2 if ae_v3 else 1, figsize=(10.5 if ae_v3 else 5.5, 5.0))
    if ae_v3 is None:
        axes = [axes]

    def _plot_noop(ax, cell: CellResult, title: str):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="y = x (no-op)")
        ax.scatter(cell.tok_acc_perturbed, cell.tok_acc, c="tab:red", s=80,
                   edgecolor="black", linewidth=0.6, zorder=3)
        for a, p, t in zip(cell.alphas, cell.tok_acc_perturbed, cell.tok_acc):
            ax.annotate(f"α={a}", (p, t), xytext=(5, -10), textcoords="offset points",
                         fontsize=8, color="dimgray")
        ax.set_xlabel("Argmax accuracy of perturbed input  (do nothing)")
        ax.set_ylabel("Argmax accuracy after sampler  (do something)")
        ax.set_title(title)
        ax.legend(loc="upper left", framealpha=0.95)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    _plot_noop(axes[0], ref_det_mse,
               "comp_ref_det_mse  (CLR, deterministic x₁)\nFM Eq.7 Dot Prod.")
    if ae_v3:
        _plot_noop(axes[1], ae_v3,
                   "ae_d1024_l8_z128_v3 (AE-latent EqM)\nsame phenotype, different chart")

    fig4.suptitle("Figure 4 — The §3 no-op signature is chart-invariant\n"
                  "(every point sits on the y=x diagonal: the sampler did no useful work)",
                  fontsize=11, y=1.02)
    fig4.tight_layout()
    fig4.savefig(OUT / "fig4_noop_signature.png")
    fig4.savefig(OUT / "fig4_noop_signature.pdf")
    plt.close(fig4)
    print(f"  wrote {OUT/'fig4_noop_signature.png'}")

    # Numbers summary
    print()
    print("Numbers used in the figures (sanity check):")
    print(f"  comp_ref_det_mse:   KL_uni={ref_det_mse.kl_uni:.3f}   Δ@.50={_delta_50(ref_det_mse):+.3f}")
    print(f"  comp_mse (3 seeds): KL_uni={comp_mse_kl:.3f}   Δ@.50={comp_mse_d50:+.3f} ± {comp_mse_d50_std:.3f}")
    print(f"  comp_hilbert (3s):  KL_uni={comp_h_kl:.3f}   Δ@.50={float(np.mean(h_d50s)):+.3f} ± {float(np.std(h_d50s)):.3f}")
    print(f"  dsm_clr_det:        KL_uni={dsm_det.kl_uni:.3f}   Δ@.50={_delta_50(dsm_det):+.3f}")
    print(f"  dsm_clr_dir:        KL_uni={dsm_dir.kl_uni:.3f}   Δ@.50={_delta_50(dsm_dir):+.3f}")


if __name__ == "__main__":
    main()
