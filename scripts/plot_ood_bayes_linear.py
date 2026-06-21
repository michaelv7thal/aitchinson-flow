"""Plots for the BayesLinHead OOD sweep (scripts/ood_bayes_linear.py).

Reads a ``bayes_linear_sweep.json`` and writes, next to it:

  1. ``<stem>_auroc.png`` — AUROC vs corruption rate, one line per scheme,
     2x2 panels (sequence energy / sequence variance / per-token energy /
     per-token variance). The 0.5 chance line is drawn for reference.
  2. ``<stem>_heatmap_<which>_<idx>.png`` — per-TOKEN energy and variance
     heatmaps over sequence position. Corrupted tokens are MARKED: a red box
     around the column in both heatmaps and a red, bold character tick label.

Standalone — re-plot without re-running the (GPU-bound) sweep:

    uv run python scripts/plot_ood_bayes_linear.py \
        --json ood_out/bayeslin_pca0/bayes_linear_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _plt():
    """Lazy Agg-backend matplotlib import (headless-safe; matches eval_ood.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_sweep(data: dict, out_path: Path) -> Path | None:
    """AUROC-vs-rate sweep, 2x2 panels, one line per corruption scheme."""
    plt = _plt()
    rows = [r for r in data.get("rows", []) if r.get("scheme")]
    if not rows:
        return None
    schemes: list[str] = []
    for r in rows:
        if r["scheme"] not in schemes:
            schemes.append(r["scheme"])
    panels = [
        ("auroc_seq_energy", "Sequence energy AUROC"),
        ("auroc_seq_uncertainty", "Sequence variance AUROC"),
        ("auroc_token_energy", "Per-token energy AUROC"),
        ("auroc_token_uncertainty", "Per-token variance AUROC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for scheme in schemes:
            pts = [(r["rate"], r.get(key)) for r in rows if r["scheme"] == scheme]
            # drop missing / NaN (NaN != NaN)
            pts = [(x, y) for x, y in pts if y is not None and y == y]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker="o", label=scheme)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")  # chance
        ax.set_title(title)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
        ax.set_xlabel("corruption rate")
        ax.set_ylabel("AUROC")
    axes[0, 0].legend(title="scheme", fontsize=9)
    det = data.get("detector", "BayesLinHead")
    fig.suptitle(
        f"{det} — OOD AUROC vs corruption ladder "
        f"(t_eval={data.get('t_eval')}, feat_dim={data.get('feat_dim')})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_heatmap(ex: dict, out_path: Path, max_pos: int = 120) -> Path:
    """Per-token energy & variance heatmaps; corrupted positions marked red."""
    plt = _plt()
    from matplotlib.patches import Rectangle

    E = ex["E_t"][:max_pos]
    V = ex["Var_t"][:max_pos]
    L = len(E)
    text = ex.get("text", "")[:L]
    corrupted = (ex.get("corrupted") or [False] * L)[:L]
    corr_idx = [j for j, c in enumerate(corrupted) if c]

    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, L * 0.12), 3.4), sharex=True)
    for ax, vals, label, cmap in [
        (axes[0], E, r"energy $E_t$", "viridis"),
        (axes[1], V, r"variance $\mathrm{Var}_t$", "magma"),
    ]:
        im = ax.imshow([vals], aspect="auto", cmap=cmap)
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        for j in corr_idx:  # red box around each corrupted column
            ax.add_patch(
                Rectangle((j - 0.5, -0.5), 1, 1, fill=False, edgecolor="red", lw=1.5)
            )

    axes[1].set_xticks(range(L))
    labels = axes[1].set_xticklabels(list(text), fontsize=6, family="monospace")
    for j in corr_idx:  # red, bold char label under corrupted positions
        if j < len(labels):
            labels[j].set_color("red")
            labels[j].set_fontweight("bold")
    axes[1].set_xlabel("sequence position (char labels; red = corrupted)")

    title = f"{ex.get('which', '?')} #{ex.get('idx', 0)}"
    if corr_idx:
        title += f"  —  {len(corr_idx)} corrupted token(s) marked red"
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_from_json(json_path, out_dir=None, max_pos: int = 120) -> list[Path]:
    """Read a bayes_linear_sweep.json and write the sweep + heatmap PNGs."""
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
    ap.add_argument("--json", required=True, help="path to bayes_linear_sweep.json")
    ap.add_argument("--out-dir", default=None, help="default: next to --json")
    ap.add_argument("--max-pos", type=int, default=120,
                    help="positions shown in the per-token heatmaps")
    args = ap.parse_args()
    for p in plot_from_json(args.json, args.out_dir, args.max_pos):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
