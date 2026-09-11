"""Per-position MAHALANOBIS-distance OOD detector ("PerPosMaha") on DirichletFM.

Concentration-of-measure-robust alternative to PerPosVarGP. Fit a single Gaussian
to IN-DISTRIBUTION per-position DirichletFM features (mean mu, shrinkage covariance
Sigma) and score each token by the Mahalanobis distance

    D^2(z) = (z - mu)^T Sigma^{-1} (z - mu).

Unlike a GP's (near-isotropic) predictive variance, the Sigma^{-1} whitening
normalises each direction by its in-distribution spread, so the distance stays
discriminative in high dimensions where the GP variance saturates (Lee et al.
2018, "A Simple Unified Framework for Detecting OOD Samples"). Per-position (no
pooling): in-distribution token -> small D^2, OOD token (replaced, or shuffled
into a wrong context) -> large D^2. Aggregate max/mean to sequence; also report
token-level localization AUROC.

DISTINCT detector from HingeSVGP (trained mean/prob) and PerPosVarGP (GP
variance) — different score, different name, different outputs (maha_perpos_sweep.json).
DirichletFM == DirichletFlowMatching (Stark et al. 2024). NOT the Discrete-FM
'DFM' arm (DiscreteFlowMatching); the loader guards against it.

Usage:
    python scripts/ood_mahalanobis_perpos.py \
        --ckpt runs/sflm_bench_a100_20g_L256/DirichletFM_ep30_d30k/epoch_final.pt \
        --out  runs/ood_maha_perpos_dirichletfm/bigger_ep30_d30k/maha_perpos_sweep.json
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

# Reuse the variance detector's load+guard and feature extraction so the two
# per-position detectors are guaranteed identical on those steps.
from scripts.ood_variance_perpos import (  # noqa: E402
    _load_dirichletfm,
    _perpos_feats,
    _auroc,
)
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-eval", type=float, default=None)
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="# in-distribution sequences to fit the Gaussian")
    ap.add_argument("--n", type=int, default=256,
                    help="# eval sequences per split")
    ap.add_argument("--max-fit-pos", type=int, default=80000,
                    help="cap on # per-position vectors used to estimate mu/Sigma")
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="0 = full-d_model Mahalanobis (default). >0 truncates to "
                         "top-k PCs of the ID cloud before fitting the Gaussian.")
    ap.add_argument("--shrinkage", type=float, default=0.1,
                    help="Ledoit-Wolf-style shrinkage of Sigma toward "
                         "(tr Sigma / d) I; stabilises Sigma^{-1} in high-d "
                         "(small eigenvalues are noisy). 0 = raw empirical cov.")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device0 = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device0)
    device = next(model.parameters()).device
    K = cfg.text8_dataset.K
    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    dm, _ = build_training_datamodule(cfg)
    val_loader = dm.val_dataloader() or dm.train_dataloader()

    # ---- in-distribution sequences (fit + eval-positive) ------------------
    seqs = []
    for batch in val_loader:
        seqs.append(batch["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs, dim=0)
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs: args.fit_seqs + args.n]

    # ---- fit single Gaussian on ID per-position features ------------------
    fit_feats = _perpos_feats(model, fit_tok, t_eval, device)           # (Nf,L,d)
    d_model = fit_feats.shape[-1]
    X = fit_feats.reshape(-1, d_model)                                   # (Nf*L,d)
    if X.shape[0] > args.max_fit_pos:
        X = X[torch.randperm(X.shape[0])[: args.max_fit_pos]]
    X = X.to(device).double()
    mu = X.mean(0)                                                       # (d,)
    Xc = X - mu
    pca_V = None
    if args.pca_dim and args.pca_dim < d_model:
        _, _, V = torch.pca_lowrank(Xc, q=min(args.pca_dim, Xc.shape[1]))
        pca_V = V[:, : args.pca_dim]                                     # (d,p)
        Xc = Xc @ pca_V
    m = Xc.shape[1]
    Sigma = (Xc.T @ Xc) / (Xc.shape[0] - 1)                             # (m,m)
    if args.shrinkage > 0:
        a = float(args.shrinkage)
        Sigma = (1 - a) * Sigma + a * (Sigma.trace() / m) * torch.eye(m, device=device, dtype=Sigma.dtype)
    # jitter for SPD safety, then Cholesky
    eps = 1e-6 * Sigma.trace() / m
    L = torch.linalg.cholesky(Sigma + eps * torch.eye(m, device=device, dtype=Sigma.dtype))
    print(f"[maha] t_eval={t_eval} d_model={d_model} feat_dim={m} "
          f"shrinkage={args.shrinkage} fit_pos={X.shape[0]}")

    @torch.no_grad()
    def perpos_maha(tok, chunk=16):
        """token_ids (B,L) -> per-position Mahalanobis D^2 (B,L)."""
        outs = []
        for i in range(0, tok.shape[0], chunk):
            tb = tok[i:i + chunk].to(device).long()
            B, Lq = tb.shape
            t = torch.full((B,), t_eval, device=device)
            beta = torch.ones(B, Lq, model.K, device=device)
            beta.scatter_(-1, tb.unsqueeze(-1), float(t_eval))
            x_t = beta / beta.sum(-1, keepdim=True)      # deterministic (match _perpos_feats)
            h = model.get_hidden_states(x_t, t).reshape(B * Lq, d_model).double()
            xc = h - mu
            if pca_V is not None:
                xc = xc @ pca_V
            # D^2 = || L^{-1} xc^T ||^2  (whitened squared norm)
            w = torch.linalg.solve_triangular(L, xc.T, upper=False)      # (m, B*Lq)
            d2 = (w * w).sum(0)                                          # (B*Lq,)
            outs.append(d2.reshape(B, Lq).float().cpu())
        return torch.cat(outs, dim=0)

    # ---- score positives + corruption ladder ------------------------------
    pos = perpos_maha(pos_tok)
    pos_max = pos.max(dim=1).values.numpy()
    pos_mean = pos.mean(dim=1).numpy()
    print(f"[maha] positive D^2: per-pos mean={pos.mean():.3f} seq-max mean={pos_max.mean():.3f}")

    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    rows = [{"scheme": None, "rate": 0.0, "n": int(pos_tok.shape[0]),
             "mean_d2": float(pos.mean()), "seq_max_mean": float(pos_max.mean())}]

    def _corrupt(tok, scheme, rate, seed):
        if scheme == "replace":
            return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        if scheme == "shuffle":
            return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
        out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)

    print(f"\n{'scheme':>10} {'rate':>5} {'AUROC_max':>10} {'AUROC_mean':>11} "
          f"{'tokloc_AUROC':>13}")
    for scheme in schemes:
        for r in rates:
            ood_tok = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
            ood = perpos_maha(ood_tok)
            ood_max = ood.max(dim=1).values.numpy()
            ood_mean = ood.mean(dim=1).numpy()
            lab = np.concatenate([np.zeros(len(pos_max)), np.ones(len(ood_max))])
            au_max = _auroc(np.concatenate([pos_max, ood_max]), lab)
            au_mean = _auroc(np.concatenate([pos_mean, ood_mean]), lab)
            changed = (ood_tok != pos_tok)
            tokloc = float("nan")
            if changed.any() and (~changed).any():
                tokloc = _auroc(ood.numpy().reshape(-1),
                                changed.numpy().reshape(-1).astype(int))
            print(f"{scheme:>10} {r:>5.2f} {au_max:>10.4f} {au_mean:>11.4f} "
                  f"{tokloc:>13.4f}")
            rows.append({"scheme": scheme, "rate": r, "n": int(ood_tok.shape[0]),
                         "mean_d2": float(ood.mean()),
                         "seq_max_mean": float(ood_max.mean()),
                         "auroc_maha_max": au_max, "auroc_maha_mean": au_mean,
                         "auroc_token_localization": tokloc})

    out_path = Path(args.out or (Path(args.ckpt).parent / "maha_perpos_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "t_eval": t_eval, "d_model": d_model, "feat_dim": m,
        "pca_dim": args.pca_dim, "shrinkage": args.shrinkage,
        "detector": "PerPosMaha",
        "detector_long": "per-position Mahalanobis distance to ID feature Gaussian",
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
