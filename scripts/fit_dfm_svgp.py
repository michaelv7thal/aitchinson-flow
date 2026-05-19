"""Stage 2 for DirichletFMSvgp: fit the SVGP head + report OOD AUROC.

After Stage 1 (standard DFM training via ``scripts/run_sweep.py``), call:

    python scripts/fit_dfm_svgp.py --ckpt runs/<cell>/epoch_final.pt \\
        --neg-strategy scrambled --n-pos 4000 --n-iters 200

Writes ``svgp_state.pt`` (the fitted GP + buffers) and ``svgp_eval.json``
(AUROC of latent std vs. scrambled/random-simplex/cross-corpus OOD).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
from aitchinson_flow.models import build_model  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from scripts.eval_full import _config_from_payload  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U via rank-sum; positive class label = 1.

    Returns NaN if either class has < 2 samples.
    """
    from sklearn.metrics import roc_auc_score
    if (labels == 1).sum() < 2 or (labels == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="defaults to dirname(ckpt)")
    ap.add_argument("--n-pos", type=int, default=None,
                    help="override cfg.dfm_svgp.n_pos")
    ap.add_argument("--n-iters", type=int, default=None,
                    help="override cfg.dfm_svgp.n_iters")
    ap.add_argument("--neg-strategy", type=str, default=None,
                    choices=["scrambled", "random_simplex", "mixed"],
                    help="override cfg.dfm_svgp.neg_strategy for the fit")
    ap.add_argument("--t-eval", type=float, default=None)
    ap.add_argument("--eval-n", type=int, default=500,
                    help="number of val sequences for AUROC eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _config_from_payload(payload)
    # CLI overrides
    dfs_overrides: dict = {}
    if args.n_pos is not None: dfs_overrides["n_pos"] = args.n_pos
    if args.n_iters is not None: dfs_overrides["n_iters"] = args.n_iters
    if args.neg_strategy is not None: dfs_overrides["neg_strategy"] = args.neg_strategy
    if args.t_eval is not None: dfs_overrides["t_eval"] = float(args.t_eval)
    if dfs_overrides:
        cfg.dfm_svgp = replace(cfg.dfm_svgp, **dfs_overrides)

    device = cfg.training.device
    torch.manual_seed(args.seed)

    model = build_model(cfg).to(device)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        # SVGP buffers initialized fresh — that's expected for a Stage-1-only ckpt.
        non_svgp_unexpected = [k for k in unexpected if "svgp" not in k]
        if non_svgp_unexpected:
            print(f"[warn] unexpected non-SVGP keys: {non_svgp_unexpected[:3]}")
    if missing:
        non_svgp_missing = [k for k in missing if "svgp" not in k]
        if non_svgp_missing:
            print(f"[warn] missing non-SVGP keys: {non_svgp_missing[:3]}")
    model.eval()

    dm, _ = build_training_datamodule(cfg)

    # --- Stage 2: fit ---------------------------------------------------
    print(f"[stage 2] fitting SVGP head — strategy={cfg.dfm_svgp.neg_strategy}, "
          f"n_pos={cfg.dfm_svgp.n_pos}, n_iters={cfg.dfm_svgp.n_iters}, "
          f"t_eval={cfg.dfm_svgp.t_eval}")
    fit_summary = model.fit_svgp(
        dm.train_dataloader(),
        t_eval=cfg.dfm_svgp.t_eval,
        max_pos=cfg.dfm_svgp.n_pos,
        neg_strategy=cfg.dfm_svgp.neg_strategy,
        neg_per_pos_ratio=cfg.dfm_svgp.neg_per_pos_ratio,
        n_iters=cfg.dfm_svgp.n_iters,
        lr=cfg.dfm_svgp.lr,
        verbose=True,
    )
    print(f"[stage 2] done. ELBO={fit_summary['elbo_loss']:.4f} "
          f"n_pos={fit_summary['n_pos']} n_neg={fit_summary['n_neg']}")

    # Save fitted SVGP weights.
    torch.save(model.state_dict(), out_dir / "model_with_svgp.pt")

    # --- Eval: AUROC of SVGP std as an OOD scorer ----------------------
    val_loader = dm.val_dataloader()
    if val_loader is None:
        val_loader = dm.train_dataloader()
    print(f"[eval] collecting positive features from val for AUROC")

    pos_std: list[torch.Tensor] = []
    pos_mean: list[torch.Tensor] = []
    pos_prob: list[torch.Tensor] = []
    n_eval = 0
    for batch in val_loader:
        tok = batch["token_ids"].to(device).long()
        s = model.ood_score(tok, t_eval=cfg.dfm_svgp.t_eval)
        pos_std.append(s["std"].cpu()); pos_mean.append(s["mean"].cpu()); pos_prob.append(s["prob"].cpu())
        n_eval += tok.shape[0]
        if n_eval >= args.eval_n:
            break

    # Generate matching OOD via each strategy and score them.
    rows: list[dict] = []
    pos_std_v = torch.cat(pos_std)[:args.eval_n]
    pos_mean_v = torch.cat(pos_mean)[:args.eval_n]
    pos_prob_v = torch.cat(pos_prob)[:args.eval_n]
    print(f"  positive mean(std)={pos_std_v.mean():.4f}  mean(prob)={pos_prob_v.mean():.4f}")
    rows.append({
        "split": "positive_val",
        "n": int(pos_std_v.numel()),
        "mean_std": float(pos_std_v.mean()), "std_std": float(pos_std_v.std()),
        "mean_prob": float(pos_prob_v.mean()),
        "mean_latent": float(pos_mean_v.mean()),
    })

    for neg_strategy in ("scrambled", "random_simplex"):
        ood_std: list[torch.Tensor] = []
        ood_prob: list[torch.Tensor] = []
        ood_mean: list[torch.Tensor] = []
        n = 0
        for batch in val_loader:
            tok = batch["token_ids"].to(device).long()
            B, L = tok.shape
            t = torch.full((B,), float(cfg.dfm_svgp.t_eval), device=device)
            if neg_strategy == "scrambled":
                # Per-row shuffle (preserves unigram, breaks bigram)
                perms = torch.argsort(torch.rand(B, L, device=device), dim=-1)
                tok_ood = tok.gather(1, perms)
                z = model.pool_features(tok_ood, t)
            else:  # random_simplex
                from torch.distributions import Dirichlet as _D
                K = model.K
                x_t = _D(torch.ones(K, device=device)).sample((B, L))
                # Bypass pool_features (which would re-Dirichlet); pool directly
                h = model.get_hidden_states(x_t, t)
                z = model.pooler(h)
            prob, mean, std = model.svgp(z)
            ood_std.append(std.cpu()); ood_prob.append(prob.cpu()); ood_mean.append(mean.cpu())
            n += B
            if n >= args.eval_n:
                break
        ood_std_v = torch.cat(ood_std)[:args.eval_n]
        ood_prob_v = torch.cat(ood_prob)[:args.eval_n]
        ood_mean_v = torch.cat(ood_mean)[:args.eval_n]
        # AUROC: OOD = label 1 (high std should rank OOD above positives)
        all_scores_std = torch.cat([pos_std_v, ood_std_v]).numpy()
        all_scores_prob = torch.cat([pos_prob_v, ood_prob_v]).numpy()
        all_labels = np.concatenate([
            np.zeros(pos_std_v.numel()), np.ones(ood_std_v.numel())
        ])
        auroc_std = _auroc(all_scores_std, all_labels)
        auroc_negprob = _auroc(-all_scores_prob, all_labels)  # low in-dist prob ↔ OOD
        print(f"  [{neg_strategy:>15}] mean(std)={ood_std_v.mean():.4f}  "
              f"AUROC(std)={auroc_std:.4f}  AUROC(-prob)={auroc_negprob:.4f}")
        rows.append({
            "split": f"ood_{neg_strategy}",
            "n": int(ood_std_v.numel()),
            "mean_std": float(ood_std_v.mean()), "std_std": float(ood_std_v.std()),
            "mean_prob": float(ood_prob_v.mean()),
            "mean_latent": float(ood_mean_v.mean()),
            "auroc_std": auroc_std,
            "auroc_neg_prob": auroc_negprob,
            "auroc_prob_as_ood": 1.0 - auroc_negprob,
        })

    out_path = out_dir / "svgp_eval.json"
    out_path.write_text(json.dumps({"fit": fit_summary, "rows": rows}, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
