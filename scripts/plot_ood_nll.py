"""Plots for the DenoiserNLL OOD sweep (scripts/ood_denoiser_nll.py).

Reads a ``denoiser_nll_sweep.json`` and writes, next to it:

  1. ``<stem>_auroc.png`` — AUROC vs corruption rate, one line per scheme, 2x2
     panels: sequence NLL / sequence variance / per-token NLL / per-token
     variance. The 0.5 chance line is drawn for reference.
  2. ``<stem>_heatmap_<which>_<idx>.png`` — per-TOKEN NLL (and variance, if the
     baseline was computed) over sequence position; corrupted tokens marked red.

Standalone — re-plot without re-running the (GPU-bound) sweep:

    uv run python scripts/plot_ood_nll.py \
        --json ood_out/nll/denoiser_nll_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_sweep(data: dict, out_path: Path) -> Path | None:
    plt = _plt()
    rows = [r for r in data.get("rows", []) if r.get("scheme")]
    if not rows:
        return None
    schemes: list[str] = []
    for r in rows:
        if r["scheme"] not in schemes:
            schemes.append(r["scheme"])
    panels = [
        ("auroc_seq_nll", "Sequence NLL AUROC"),
        ("auroc_seq_var", f"Sequence variance AUROC (t={data.get('t_var')})"),
        ("auroc_token_nll", "Per-token NLL AUROC"),
        ("auroc_token_var", f"Per-token variance AUROC (t={data.get('t_var')})"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        drew = False
        for scheme in schemes:
            pts = [(r["rate"], r.get(key)) for r in rows if r["scheme"] == scheme]
            pts = [(x, y) for x, y in pts if y is not None and y == y]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker="o", label=scheme)
                drew = True
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.set_xlabel("corruption rate")
        ax.set_ylabel("AUROC")
        if not drew:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
    axes[0, 0].legend(title="scheme", fontsize=9)
    fig.suptitle(
        f"{data.get('detector', 'DenoiserNLL')} — OOD AUROC vs corruption ladder "
        f"(t_nll={data.get('t_nll')}, t_var={data.get('t_var')})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_heatmap(ex: dict, out_path: Path, max_pos: int = 120) -> Path:
    plt = _plt()
    from matplotlib.patches import Rectangle

    N = ex["NLL_t"][:max_pos]
    L = len(N)
    text = ex.get("text", "")[:L]
    corrupted = (ex.get("corrupted") or [False] * L)[:L]
    corr_idx = [j for j, c in enumerate(corrupted) if c]

    panels = [(N, r"NLL $-\log p_t$", "viridis")]
    V = ex.get("Var_t")
    if V is not None:
        panels.append((V[:max_pos], r"variance $\mathrm{Var}_t$", "magma"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(max(8.0, L * 0.12), 1.8 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (vals, label, cmap) in zip(axes, panels):
        im = ax.imshow([vals], aspect="auto", cmap=cmap)
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        for j in corr_idx:
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=1.5))
    axes[-1].set_xticks(range(L))
    labels = axes[-1].set_xticklabels(list(text), fontsize=6, family="monospace")
    for j in corr_idx:
        if j < len(labels):
            labels[j].set_color("red")
            labels[j].set_fontweight("bold")
    axes[-1].set_xlabel("sequence position (char labels; red = corrupted)")
    title = f"{ex.get('which', '?')} #{ex.get('idx', 0)}"
    if corr_idx:
        title += f"  —  {len(corr_idx)} corrupted token(s) marked red"
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_from_json(json_path, out_dir=None, max_pos: int = 120) -> list[Path]:
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    out_dir = Path(out_dir) if out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem
    written: list[Path] = []
    sweep = plot_sweep(data, out_dir / f"{stem}_auroc.png")
    if sweep is not None:
        written.append(sweep)
    for ex in data.get("examples", []):
        name = f"{stem}_heatmap_{ex.get('which', 'ex')}_{ex.get('idx', 0)}.png"
        written.append(plot_heatmap(ex, out_dir / name, max_pos=max_pos))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-pos", type=int, default=120)
    args = ap.parse_args()
    for p in plot_from_json(args.json, args.out_dir, args.max_pos):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
