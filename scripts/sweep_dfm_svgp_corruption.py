"""Sweep the OOD AUROC of a trained DFM+SVGP across corruption rates.

For each rate r ∈ [0.1, 0.2, ..., 1.0] and each corruption scheme
(replace | shuffle | both), generate OOD batches by applying that scheme
at rate r to held-out val sequences, score them through ``model.svgp``,
and compute AUROC(prob, label=OOD) against an in-distribution baseline.

Writes a tabular JSON + a single matplotlib plot of AUROC vs rate per
scheme. Run after the canonical Stage 2 (fit_dfm_svgp_hinge.py) has
populated ``model_with_svgp_hinge.pt``.

Usage:
    python scripts/sweep_dfm_svgp_corruption.py \\
        --ckpt runs/dfm_svgp_pure50_lr3e4/model_with_svgp_hinge.pt \\
        --n 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

import aitchinson_flow.models  # noqa: E402,F401
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if (labels == 1).sum() < 2 or (labels == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _apply_corruption(
    tok: torch.Tensor, *, scheme: str, rate: float, K: int, seed: int
) -> torch.Tensor:
    if scheme == "replace":
        return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
    if scheme == "both":
        out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        out = partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)
        return out
    raise ValueError(f"unknown scheme: {scheme!r}")


def _collect_prob(
    model, loader, *, t_eval: float, n: int, device: torch.device
) -> torch.Tensor:
    """Score positive val batches up to n samples."""
    probs: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        tok = batch["token_ids"].to(device).long()
        s = model.ood_score(tok, t_eval=t_eval)
        probs.append(s["prob"].detach().cpu())
        seen += tok.shape[0]
        if seen >= n:
            break
    return torch.cat(probs)[:n]


def _collect_ood_prob(
    model, loader, *, t_eval: float, n: int, device: torch.device,
    scheme: str, rate: float, K: int, seed: int,
) -> torch.Tensor:
    probs: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        tok = batch["token_ids"].to(device).long()
        # Apply on CPU to use the existing seed-driven RNG, then move back.
        tok_corr = _apply_corruption(tok.cpu(), scheme=scheme, rate=rate, K=K,
                                     seed=seed + seen).to(device)
        s = model.ood_score(tok_corr, t_eval=t_eval)
        probs.append(s["prob"].detach().cpu())
        seen += tok.shape[0]
        if seen >= n:
            break
    return torch.cat(probs)[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="model_with_svgp_hinge.pt (or any DFM+SVGP ckpt with fitted SVGP)")
    ap.add_argument("--out", type=str, default=None,
                    help="JSON output path; defaults to <ckpt-dir>/svgp_corruption_sweep.json")
    ap.add_argument("--n", type=int, default=500,
                    help="number of val sequences per row")
    ap.add_argument("--rates", type=str,
                    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-eval", type=float, default=None,
                    help="override cfg.dfm_svgp.t_eval")
    args = ap.parse_args()

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    schemes = [s for s in args.schemes.split(",") if s.strip()]

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # ckpt may be a pure state_dict (model_with_svgp_hinge.pt) without cfg;
    # fall back to sibling epoch_final.pt for the cfg.
    if isinstance(payload, dict) and "cfg" in payload:
        cfg = _config_from_payload(payload)
    else:
        sibling = Path(args.ckpt).parent / "epoch_final.pt"
        if not sibling.exists():
            raise SystemExit(f"no cfg in {args.ckpt}, no sibling epoch_final.pt either")
        sibling_payload = torch.load(sibling, map_location="cpu", weights_only=False)
        cfg = _config_from_payload(sibling_payload)

    device = cfg.training.device
    torch.manual_seed(args.seed)
    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=False)
    model.eval()

    dm, _ = build_training_datamodule(cfg)
    val_loader = dm.val_dataloader() or dm.train_dataloader()
    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    K = cfg.text8_dataset.K

    # Score positives once.
    pos_prob = _collect_prob(model, val_loader, t_eval=t_eval, n=args.n, device=device)
    print(f"positive_val (n={pos_prob.numel()})  mean(prob)={pos_prob.mean():.4f}")

    rows: list[dict] = []
    rows.append({
        "scheme": None, "rate": 0.0, "n": int(pos_prob.numel()),
        "mean_prob": float(pos_prob.mean()), "std_prob": float(pos_prob.std()),
        "auroc_prob_as_ood": float("nan"),
    })

    print(f"\n{'scheme':>10} {'rate':>6} {'mean_prob':>10} {'AUROC(prob,OOD)':>16}")
    print(f"{'------':>10} {'----':>6} {'---------':>10} {'---------------':>16}")
    for scheme in schemes:
        for r in rates:
            ood_prob = _collect_ood_prob(
                model, val_loader, t_eval=t_eval, n=args.n, device=device,
                scheme=scheme, rate=r, K=K, seed=args.seed + int(1000*r),
            )
            scores = torch.cat([pos_prob, ood_prob]).numpy()
            labels = np.concatenate([
                np.zeros(pos_prob.numel()), np.ones(ood_prob.numel())
            ])
            auroc = _auroc(scores, labels)
            mean_o = float(ood_prob.mean())
            print(f"{scheme:>10} {r:>6.2f} {mean_o:>10.4f} {auroc:>16.4f}")
            rows.append({
                "scheme": scheme, "rate": r,
                "n": int(ood_prob.numel()),
                "mean_prob": mean_o, "std_prob": float(ood_prob.std()),
                "auroc_prob_as_ood": auroc,
            })

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "svgp_corruption_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "t_eval": t_eval,
        "n_per_split": int(args.n),
        "rates": rates,
        "schemes": schemes,
        "rows": rows,
        "_convention": (
            "auroc_prob_as_ood: AUROC of model.svgp prob output as the OOD "
            "score with label=1 for OOD. The hinge trained E_invalid > E_clean, "
            "so prob > 0.5 indicates OOD. Higher AUROC means better OOD detection."
        ),
    }, indent=2))
    print(f"\nWrote {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for scheme in schemes:
            xs, ys = [], []
            for row in rows:
                if row.get("scheme") == scheme:
                    xs.append(row["rate"]); ys.append(row["auroc_prob_as_ood"])
            ax.plot(xs, ys, marker="o", label=scheme, linewidth=1.6)
        ax.axhline(0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.5, label="chance")
        ax.set_xlabel("Corruption rate r  (fraction of positions)")
        ax.set_ylabel("AUROC(prob, label=OOD)")
        ax.set_title("DFM+SVGP OOD AUROC vs corruption rate\n"
                     "(replace = random-token; shuffle = in-row permutation; both = combined)")
        ax.set_xlim(0, 1.05); ax.set_ylim(0.45, 1.01)
        ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
        plot_path = out_path.with_suffix(".png")
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {plot_path}")
    except Exception as e:
        print(f"[warn] plot generation failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
