"""Phase C visualization — cross-method × corruption-contrast OOD scorecard.

Two panels:

1. Heatmap of AUROC across (model, statistic) rows × corruption columns.
   Pulled directly from the four ``runs/<ckpt>/ood_eval.json`` files.
2. Per-method ROC overlay for the headline contrast (clean vs valid_perm,
   the only contrast where DFM doesn't trivially saturate AUC=1.0). The
   ROCs are reconstructed from the raw scores stored in each json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


CKPTS = [
    "eqm_data50k_ep5_v2",
    "dfm_data50k_ep5_v2",
    "fmclr_data50k_ep5_v2",
    "lkflow_data50k_ep5",
]

DISPLAY = {
    "eqm_data50k_ep5_v2":   "EqM",
    "dfm_data50k_ep5_v2":   "DFM",
    "fmclr_data50k_ep5_v2": "FMonCLR",
    "lkflow_data50k_ep5":   "LogitKLFlow",
}


def _load_summary(run: str) -> dict | None:
    path = ROOT / "runs" / run / "ood_eval.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _flatten_aucs(summary: dict, contrasts: list[str]) -> dict[tuple[str, str], float]:
    """Flatten {stat: {contrast: {auc, auc_abs}}} → {(stat, contrast): auc_for_table}.

    Uses ``auc_abs`` when ``auc`` < 0.5 (the energy can be flipped sign;
    abs reflects bidirectional separation).
    """
    out = {}
    for stat, by_c in summary.get("auc", {}).items():
        for c in contrasts:
            if c not in by_c:
                continue
            v = by_c[c]
            auc, auc_abs = v.get("auc", float("nan")), v.get("auc_abs", float("nan"))
            chosen = auc if auc >= 0.5 else auc_abs
            out[(stat, c)] = chosen
    return out


def _roc_curve(neg: torch.Tensor, pos: torch.Tensor) -> tuple[list[float], list[float], float]:
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return [], [], float("nan")
    p = pos.flatten().numpy()
    n = neg.flatten().numpy()
    thresholds = np.sort(np.concatenate([p, n]))[::-1]
    tpr, fpr = [], []
    for thr in thresholds:
        tpr.append(float((p >= thr).mean()))
        fpr.append(float((n >= thr).mean()))
    # Compute Mann-Whitney U
    combined = np.concatenate([p, n])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)
    auc = float(
        (ranks[: n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )
    return fpr, tpr, auc


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/phaseC_summary.png")
    args = p.parse_args(argv)

    contrasts = ["subst_0.5", "shuffle_0.5", "valid_perm", "rand"]

    rows = []  # list of (label, ckpt-name, stat, dict-of-aucs)
    raw_per_ckpt: dict[str, dict] = {}
    for ck in CKPTS:
        s = _load_summary(ck)
        if s is None:
            print(f"[plot_phaseC] missing ood_eval.json for {ck}, skipping")
            continue
        raw_per_ckpt[ck] = s
        flat = _flatten_aucs(s, contrasts)
        for (stat, c), auc in flat.items():
            rows.append((DISPLAY[ck], ck, stat, c, auc))

    # Build the table: rows = (model, stat), cols = contrasts
    seen = set()
    row_keys: list[tuple[str, str]] = []
    for (model, _, stat, _, _) in rows:
        key = (model, stat)
        if key not in seen:
            seen.add(key)
            row_keys.append(key)
    matrix = np.full((len(row_keys), len(contrasts)), np.nan)
    for r, (model, stat) in enumerate(row_keys):
        for c, contrast in enumerate(contrasts):
            for (m, _, s, ct, auc) in rows:
                if m == model and s == stat and ct == contrast:
                    matrix[r, c] = auc

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # --- Heatmap ---
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "auc", ["#3b6cb0", "#f7f7f7", "#c43c3c"], N=256
    )
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(contrasts)))
    ax.set_xticklabels(contrasts, rotation=20, ha="right")
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{m}: {s}" for (m, s) in row_keys])
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        color="black" if 0.55 < v < 0.85 else "white", fontsize=8)
    ax.set_title("Phase C — clean-vs-corruption AUROC (per model × statistic × contrast)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="AUROC")

    # --- ROC overlay for valid_perm ---
    ax = axes[1]
    chosen_contrast = "valid_perm"
    for ck in CKPTS:
        if ck not in raw_per_ckpt:
            continue
        s = raw_per_ckpt[ck]
        raw = s.get("raw_scores", {})
        if "clean" not in raw or chosen_contrast not in raw:
            continue
        # Pick the best statistic per ckpt for this contrast — "best" means
        # maximum *separation* from chance, i.e. max |AUC − 0.5|.
        best_stat = None
        best_separation = -1.0
        best_auc = float("nan")
        for stat in raw["clean"]:
            if stat not in raw[chosen_contrast]:
                continue
            clean = torch.tensor(raw["clean"][stat])
            corrupt = torch.tensor(raw[chosen_contrast][stat])
            _, _, auc = _roc_curve(clean, corrupt)
            sep = abs(auc - 0.5) if not np.isnan(auc) else -1.0
            if sep > best_separation:
                best_stat, best_auc, best_separation = stat, auc, sep
        if best_stat is None:
            continue
        clean = torch.tensor(raw["clean"][best_stat])
        corrupt = torch.tensor(raw[chosen_contrast][best_stat])
        # If signed AUC was below 0.5, flip the score sign so the curve makes sense.
        flip = best_auc < 0.5
        if flip:
            clean = -clean
            corrupt = -corrupt
            best_auc = 1.0 - best_auc
        fpr, tpr, _ = _roc_curve(clean, corrupt)
        label = f"{DISPLAY[ck]} ({best_stat}{', |·|' if flip else ''}): AUC={best_auc:.3f}"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC — clean vs valid_perm (best statistic per method)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "Phase C — cross-method OOD scorecard on text8 (n=256 val windows × seed 1234)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot_phaseC] wrote {out_path}")


if __name__ == "__main__":
    main()
