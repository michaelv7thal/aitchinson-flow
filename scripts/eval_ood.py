"""OOD scorecard for EqM / DFM / FMonCLR checkpoints.

Showcase eval for the user-stated EBM differentiator: an energy field on the
Aitchison simplex should give a sequence-level uncertainty signal that
separates clean text8 windows from corrupted (substitution / shuffled /
fully-random) ones. DFM's denoiser doesn't expose an analogous quantity
natively; we compute a fairness proxy ``-log p_{1|t≈1}(x|x)`` from its
denoiser logits so the scorecard has one row per model.

Outputs:
    runs/<name>/ood_eval.json   per-(corruption, rate, statistic) means + AUCs
    runs/<name>/ood_eval.png    violin plot + ROC curves

Usage:
    python scripts/eval_ood.py --ckpt runs/data_50k_ep5/epoch_final.pt \\
        --out runs/data_50k_ep5/ood_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import aitchinson_flow.models  # noqa: E402,F401 — populate registry
from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

from scripts.eval_full import _config_from_payload  # noqa: E402


# Corruption ladder. Each entry yields a (corruption_name, rate) cell.
SUBST_RATES = [0.1, 0.25, 0.5, 0.75]
SHUFFLE_RATES = [0.25, 0.5, 0.75, 1.0]


def _build_corruptions(
    clean_ids: torch.Tensor, *, K: int, seed: int = 1234
) -> dict[str, torch.Tensor]:
    """Return a dict of {label → token_ids tensor} matching clean_ids' shape."""
    out: dict[str, torch.Tensor] = {"clean": clean_ids.clone()}
    for r in SUBST_RATES:
        out[f"subst_{r}"] = corrupt_token_ids(
            clean_ids, vocab_size=K, corrupt_rate=r, seed=seed
        )
    for r in SHUFFLE_RATES:
        out[f"shuffle_{r}"] = partially_shuffle_token_ids(
            clean_ids, shuffle_rate=r, seed=seed
        )
    out["rand"] = torch.randint(
        0, K, clean_ids.shape, device=clean_ids.device,
        generator=torch.Generator(device=clean_ids.device).manual_seed(seed),
    )
    return out


@torch.no_grad()
def _score_eqm_like(
    model: Any, ids: torch.Tensor, *, K: int, label_smoothing: float
) -> dict[str, torch.Tensor]:
    """Score with EqM / FMonCLR: E_seq, U_pos.mean, U_pos.max per sequence."""
    feats = token_ids_to_features(ids, K, label_smoothing=label_smoothing)
    e_seq = model.energy(feats)  # (B,)
    out = {"E_seq": e_seq.cpu()}
    if hasattr(model, "position_uncertainty"):
        u = model.position_uncertainty(feats)  # (B, L)
        out["U_pos_mean"] = u.mean(dim=-1).cpu()
        out["U_pos_max"] = u.max(dim=-1).values.cpu()
    return out


@torch.no_grad()
def _score_dfm(model: Any, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """DFM fairness proxy: ``-log p_{1|t≈1}(x | x)`` per sequence.

    Feeds the candidate sequence as both the corrupted input and the target,
    queries the denoiser at t close to 1 (where the denoiser is supposed to
    return the candidate), and reports the per-sequence cross-entropy. A
    well-trained DFM should assign low NLL to clean text8 (in distribution)
    and higher NLL to corrupted variants.
    """
    B = ids.shape[0]
    device = ids.device
    t = torch.full((B,), 0.99, device=device)
    logits = model.forward(ids, t)  # (B, L, K)
    log_p = logits.log_softmax(dim=-1)
    nll = -log_p.gather(-1, ids.unsqueeze(-1)).squeeze(-1).mean(dim=-1)  # (B,)
    return {"E_seq": nll.cpu()}


def _roc_auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """Mann–Whitney U / 1-AUC equivalent.

    pos = scores from the *positive class* (corrupted, where higher score
    should mean more anomalous). neg = scores from the *negative class*
    (clean). Returns ROC-AUC ∈ [0, 1]; 0.5 = chance, 1.0 = perfect.
    """
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = torch.cat([pos, neg])
    order = combined.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, combined.numel() + 1, dtype=torch.float)
    pos_ranks = ranks[: n_pos]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _abs_auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """ROC-AUC computed on |score|. The energy may be negative for clean
    inputs and positive for corrupted, or vice versa; absolute value lets a
    bidirectional pole register as separation in either direction. We
    report both signed AUC and |AUC| in the JSON.
    """
    return _roc_auc(neg.abs(), pos.abs())


def evaluate_ood(
    ckpt_path: str | Path,
    *,
    n_samples: int = 256,
    seed: int = 1234,
) -> dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    device = cfg.training.device
    K = cfg.text8_dataset.K
    label_smoothing = cfg.transformation.label_smoothing

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    val_ids = dm.splits.val.long()
    rng = torch.Generator().manual_seed(seed)
    pick = torch.randperm(val_ids.shape[0], generator=rng)[:n_samples]
    clean_ids = val_ids[pick].to(device)

    corrupted = _build_corruptions(clean_ids, K=K, seed=seed)

    is_eqm_like = hasattr(model, "energy")
    is_dfm = not is_eqm_like and not hasattr(model, "decode_to_logprobs")
    scorer = _score_eqm_like if is_eqm_like else (_score_dfm if is_dfm else None)
    if scorer is None:
        raise RuntimeError(
            f"model {type(model).__name__} has neither .energy() nor a DFM-style "
            "denoiser interface; eval_ood doesn't know how to score it."
        )

    scores: dict[str, dict[str, torch.Tensor]] = {}
    for name, ids in corrupted.items():
        if is_eqm_like:
            s = _score_eqm_like(
                model, ids, K=K, label_smoothing=label_smoothing
            )
        else:
            s = _score_dfm(model, ids)
        scores[name] = s

    stat_names = sorted(scores["clean"].keys())

    summary: dict[str, Any] = {
        "ckpt": str(ckpt_path),
        "model_name": cfg.training.model_name,
        "n_samples": n_samples,
        "is_eqm_like": is_eqm_like,
        "is_dfm": is_dfm,
        "stats": {},
        "auc": {},
    }

    for stat in stat_names:
        per_corruption = {}
        for name, s in scores.items():
            v = s[stat]
            per_corruption[name] = {
                "mean": float(v.mean()),
                "std": float(v.std()) if v.numel() > 1 else 0.0,
                "median": float(v.median()),
            }
        summary["stats"][stat] = per_corruption

        aucs = {}
        clean = scores["clean"][stat]
        for contrast in ("subst_0.5", "shuffle_0.5", "rand"):
            if contrast not in scores:
                continue
            corrupt = scores[contrast][stat]
            aucs[contrast] = {
                "auc": _roc_auc(clean, corrupt),
                "auc_abs": _abs_auc(clean, corrupt),
            }
        summary["auc"][stat] = aucs

    summary["raw_scores"] = {
        name: {stat: scores[name][stat].tolist() for stat in stat_names}
        for name in scores
    }

    return summary


def _make_figure(summary: dict[str, Any], out_path: Path) -> None:
    """Two-panel violin + ROC plot. matplotlib is optional; skip if missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[ood] matplotlib not available; skipping figure")
        return

    raw = summary["raw_scores"]
    stat = "E_seq"  # headline statistic on the figure
    panels = ["clean", "subst_0.5", "shuffle_0.5", "rand"]
    panels = [p for p in panels if p in raw]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    data = [np.array(raw[p][stat]) for p in panels]
    parts = axes[0].violinplot(data, showmeans=True, showmedians=False)
    axes[0].set_xticks(range(1, len(panels) + 1))
    axes[0].set_xticklabels(panels, rotation=15)
    axes[0].set_ylabel(f"{stat} ({summary['model_name']})")
    axes[0].set_title("Score distributions by corruption")
    axes[0].grid(alpha=0.3)

    clean = np.array(raw["clean"][stat])
    for contrast in ("subst_0.5", "shuffle_0.5", "rand"):
        if contrast not in raw:
            continue
        corrupt = np.array(raw[contrast][stat])
        thresholds = np.sort(np.concatenate([clean, corrupt]))[::-1]
        tpr, fpr = [], []
        for thr in thresholds:
            tpr.append((corrupt >= thr).mean())
            fpr.append((clean >= thr).mean())
        auc = summary["auc"][stat][contrast]["auc"]
        axes[1].plot(fpr, tpr, label=f"{contrast} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k:", alpha=0.5)
    axes[1].set_xlabel("FPR (clean misclassified)")
    axes[1].set_ylabel("TPR (corrupted detected)")
    axes[1].set_title(f"ROC curves on {stat}")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"OOD detection — {summary['model_name']} "
        f"({Path(summary['ckpt']).parent.name})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=256, dest="n_samples")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    summary = evaluate_ood(
        args.ckpt, n_samples=args.n_samples, seed=args.seed
    )

    print(
        f"ckpt={Path(args.ckpt).parent.name}/{Path(args.ckpt).name}  "
        f"model={summary['model_name']}  n={summary['n_samples']}"
    )
    for stat, aucs in summary["auc"].items():
        for contrast, vals in aucs.items():
            print(
                f"  {stat:>12s}  {contrast:>14s}  "
                f"AUC={vals['auc']:.3f}  |AUC|={vals['auc_abs']:.3f}"
            )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_path}")
        if not args.no_figure:
            fig_path = out_path.with_suffix(".png")
            _make_figure(summary, fig_path)
            print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
