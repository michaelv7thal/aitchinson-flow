"""Interpretation plots for the PerPosBGMM OOD sweep (scripts/ood_bgmm_perpos.py).

Reads a ``bgmm_perpos_sweep.json`` and writes, next to it:

  1. ``<stem>_auroc.png`` — AUROC vs corruption rate. Rows = {sequence NLL
     (mean-pooled), sequence NLL (max-pooled), per-token localization};
     columns = the swept ``t_eval`` values. One line per corruption scheme; the
     0.5 chance line is drawn for reference. Lets you read off which t_eval and
     which pooling separates each corruption axis — especially the hard
     ``falseinfo`` axis.
  2. ``<stem>_clusters.png`` — the Dirichlet-process cluster structure: per
     ``t_eval`` a bar chart of the top inferred mixture weights, titled with
     ``n_effective`` (# components the variational posterior kept above the
     active-weight threshold). This is the "how many clusters did the infinite
     mixture actually use" interpretation panel.
  3. ``<stem>_heatmap_<which>_<idx>.png`` — per-TOKEN BGMM NLL over sequence
     position; corrupted tokens boxed/labelled in red. Shows the density
     localizing (or not) individual corrupt tokens.

Standalone — re-plot without re-running the (GPU-bound) sweep:

    uv run python scripts/plot_ood_bgmm.py \
        --json ood_out_best/bgmm/bgmm_perpos_sweep.json
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


def _schemes(rows: list[dict]) -> list[str]:
    out: list[str] = []
    for r in rows:
        s = r.get("scheme")
        if s and s not in out:
            out.append(s)
    return out


def _t_evals(rows: list[dict]) -> list[float]:
    out: list[float] = []
    for r in rows:
        t = r.get("t_eval")
        if t is not None and t not in out:
            out.append(t)
    return sorted(out)


def plot_sweep(data: dict, out_path: Path) -> Path | None:
    plt = _plt()
    rows = [r for r in data.get("rows", []) if r.get("scheme")]
    if not rows:
        return None
    schemes = _schemes(rows)
    t_evals = _t_evals(rows)
    metrics = [
        ("auroc_seq_gmm", "Sequence AUROC (mean-pool)"),
        ("auroc_seq_gmm_max", "Sequence AUROC (max-pool)"),
        ("auroc_token_gmm", "Per-token localization AUROC"),
    ]
    nrow, ncol = len(metrics), max(1, len(t_evals))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ri, (key, ylabel) in enumerate(metrics):
        for ci, t in enumerate(t_evals):
            ax = axes[ri][ci]
            drew = False
            for scheme in schemes:
                pts = [(r["rate"], r.get(key)) for r in rows
                       if r["scheme"] == scheme and r.get("t_eval") == t]
                pts = [(x, y) for x, y in pts if y is not None and y == y]
                if pts:
                    xs, ys = zip(*sorted(pts))
                    ax.plot(xs, ys, marker="o", ms=3, label=scheme)
                    drew = True
            ax.axhline(0.5, ls="--", lw=0.8, color="grey")
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.3)
            if ri == 0:
                ax.set_title(f"t_eval = {t}")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if ri == nrow - 1:
                ax.set_xlabel("corruption rate")
            if not drew:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes)
    axes[0][0].legend(title="scheme", fontsize=8)
    fig.suptitle(f"{data.get('detector', 'PerPosBGMM')} — OOD AUROC vs corruption "
                 f"ladder (Bayesian DP mixture, cov={data.get('covariance_type')}, "
                 f"pca={data.get('pca_dim')})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_clusters(data: dict, out_path: Path) -> Path | None:
    """Bar chart of the inferred DP mixture weights per t_eval (n_effective)."""
    plt = _plt()
    meta = data.get("n_effective_per_t") or {}
    if not meta:
        return None
    items = sorted(meta.items(), key=lambda kv: float(kv[0]))
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), squeeze=False)
    axes = axes[0]
    active = data.get("active_thresh")
    for ax, (t, m) in zip(axes, items):
        w = m.get("weights", [])
        ax.bar(range(len(w)), w, color="steelblue")
        if active is not None:
            ax.axhline(active, ls="--", lw=0.9, color="red",
                       label=f"active thr={active:.3g}")
        conv = m.get("converged")
        conv_tag = "" if conv is None else ("  ✓conv" if conv else "  ✗noconv")
        n_eff = m.get("n_effective")
        # the bars show the stored weights; if fewer were stored than are active,
        # say so rather than implying the mixture only used len(w) clusters.
        trunc = f" (top {len(w)} shown)" if (n_eff and len(w) < n_eff) else ""
        ax.set_title(f"t_eval={t}\nn_effective={n_eff}{conv_tag}{trunc}",
                     fontsize=10)
        ax.set_xlabel("mixture component (sorted)")
        ax.set_ylabel("posterior weight")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{data.get('detector', 'PerPosBGMM')} — inferred Dirichlet-process "
                 f"cluster weights (max_components={data.get('max_components')})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_heatmap(ex: dict, out_path: Path, max_pos: int = 120) -> Path:
    plt = _plt()
    from matplotlib.patches import Rectangle

    vals = (ex.get("GMMNLL_t") or ex.get("NLL_t") or [])[:max_pos]
    L = len(vals)
    text = ex.get("text", "")[:L]
    corrupted = (ex.get("corrupted") or [False] * L)[:L]
    flagged = (ex.get("flagged") or [False] * L)[:L]
    corr_idx = [j for j, c in enumerate(corrupted) if c]
    flag_idx = [j for j, f in enumerate(flagged) if f]

    fig, ax = plt.subplots(1, 1, figsize=(max(8.0, L * 0.12), 2.2))
    im = ax.imshow([vals], aspect="auto", cmap="viridis")
    ax.set_yticks([0])
    ax.set_yticklabels([r"BGMM NLL $-\log p(z)$"])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    # true-corrupt = red box; detector-flagged = cyan '▼' above the cell (heal-style)
    for j in corr_idx:
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 1, fill=False,
                               edgecolor="red", lw=1.5))
    if flag_idx:
        ax.scatter(flag_idx, [-0.62] * len(flag_idx), marker="v", s=22,
                   c="#00b3b3", clip_on=False, zorder=5)
    ax.set_xticks(range(L))
    labels = ax.set_xticklabels(list(text), fontsize=6, family="monospace")
    for j in corr_idx:
        if j < len(labels):
            labels[j].set_color("red")
            labels[j].set_fontweight("bold")
    ax.set_xlabel("sequence position (char labels; red box = corrupted; "
                  "cyan ▼ = detector-flagged)")
    title = f"{ex.get('which', '?')} #{ex.get('idx', 0)}  (t_eval={ex.get('t_eval')})"
    if corr_idx:
        tp = len(set(corr_idx) & set(flag_idx))
        title += (f"  —  {len(corr_idx)} corrupted, {len(flag_idx)} flagged "
                  f"({tp} hit)")
    ax.set_title(title)
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
    for fn, name in [(plot_sweep, f"{stem}_auroc.png"),
                     (plot_clusters, f"{stem}_clusters.png")]:
        p = fn(data, out_dir / name)
        if p is not None:
            written.append(p)
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
