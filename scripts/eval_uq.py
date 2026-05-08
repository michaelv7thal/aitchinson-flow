"""Generalised UQ evaluation on cached LM features.

Phase K (TRAINING_PROTOCOL_v2.md §5) — train and compare four UQ
methods on cached (h_LLM, label) pairs from a hallucination-style
dataset (HaluEval-QA, TruthfulQA, …). Reuses the closed-form
classifiers from `scripts/phaseF_uq.py`.

The cache schema is the one produced by `scripts/cache_hallueval.py`
(see `src/aitchinson_flow/data/hallueval.py` docstring).

Methods compared:

1. **Linear probe** — closed-form ridge logistic. Mean only.
2. **Mahalanobis distance** from the clean training mean (zero-supervised).
3. **Bayesian LR Laplace** — closed-form Hessian, calibrated probs + UQ.
4. **Deep ensemble × 5** — bootstrap diversity.
5. **SVGP** — gpytorch, RBF kernel, 64 inducing points.

Per-row features come from pooling h_LLM over the *answer span*
(everything after the prompt prefix). Pool methods: meanpool,
lasttoken, maxpool.

Outputs:

* `<out>.json` — raw numbers (AUROC + ECE per method).
* `<out>` — JSON same as above; the canonical name.
* Optional figure via `scripts/plot_uq_calibration.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aitchinson_flow.data.hallueval import load_cache  # noqa: E402

from scripts.phaseF_uq import (  # noqa: E402
    linear_probe_ridge, predict_linear,
    mahalanobis,
    deep_ensemble_linear, predict_ensemble,
    train_blr_laplace, predict_blr,
    train_svgp, predict_svgp,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _roc_auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = torch.cat([pos.flatten(), neg.flatten()]).float()
    order = combined.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, combined.numel() + 1, dtype=torch.float)
    pos_ranks = ranks[: n_pos]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    p = probs.flatten().numpy()
    y = labels.flatten().numpy().astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (p >= bins[b]) & (p < bins[b + 1] + (1e-9 if b == n_bins - 1 else 0))
        if m.sum() == 0:
            continue
        avg_p = p[m].mean()
        avg_y = y[m].mean()
        ece += (m.sum() / len(p)) * abs(avg_p - avg_y)
    return float(ece)


def _pool_features(
    h: torch.Tensor, mask: torch.Tensor, *, method: str
) -> torch.Tensor:
    """Pool a per-position hidden state into one vector per row.

    h: (N, L, H), mask: (N, L) bool. Returns (N, H).
    """
    if method == "meanpool":
        denom = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (h * mask.float().unsqueeze(-1)).sum(dim=-2) / denom
    if method == "lasttoken":
        # Pick the last True position per row.
        last = (mask.float() * torch.arange(mask.shape[-1], dtype=torch.float)).max(dim=-1).indices
        return h[torch.arange(h.shape[0]), last]
    if method == "maxpool":
        h_masked = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return h_masked.max(dim=-2).values
    raise ValueError(f"unknown pool method {method!r}")


def _split_by_pair(
    pair_id: torch.Tensor, train_frac: float, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train/val split such that both rows in a pair go to the same split.

    Returns (train_row_indices, val_row_indices).
    """
    n_pairs = int(pair_id.max().item()) + 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_pairs)
    n_train = int(train_frac * n_pairs)
    train_pairs = set(perm[:n_train].tolist())
    train_idx, val_idx = [], []
    for r, pid in enumerate(pair_id.tolist()):
        (train_idx if pid in train_pairs else val_idx).append(r)
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="path to e.g. data/hallueval_cache_gpt2.pt")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--pool", default="meanpool",
                   choices=["meanpool", "lasttoken", "maxpool"])
    p.add_argument("--svgp-iters", type=int, default=200)
    p.add_argument("--svgp-inducing", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    print(f"[uq] loading {args.cache} …")
    cache = load_cache(args.cache)
    n_rows = int(cache["full_ids"].shape[0])
    L = int(cache["L"]); H = int(cache["H"])
    print(f"  n_rows={n_rows}  L={L}  H={H}  lm={cache['lm']}  pool={args.pool}")

    h = cache["hidden_states"].float()       # (N, L, H)
    answer_mask = cache["answer_mask"].bool() # (N, L)
    label = cache["label"].bool()            # (N,) — True if hallucinated
    pair_id = cache["pair_id"].long()        # (N,)
    SE_pos = cache["SE_pos"].float()         # (N, L)

    # ---- Per-row pooling ----
    print(f"[uq] pooling features ({args.pool}) …")
    feats = _pool_features(h, answer_mask, method=args.pool)
    # SE per row: mean over answer-span tokens (sequence-level surprise).
    n_ans_tokens = answer_mask.float().sum(dim=-1).clamp(min=1)
    se_seq = (SE_pos * answer_mask.float()).sum(dim=-1) / n_ans_tokens

    # ---- Train/val split (paired) ----
    train_idx, val_idx = _split_by_pair(pair_id, args.train_frac, seed=args.seed)
    print(f"  train rows: {len(train_idx)}, val rows: {len(val_idx)}")

    feats_train = feats[train_idx]; label_train = label[train_idx]
    feats_val = feats[val_idx]; label_val = label[val_idx]
    se_seq_val = se_seq[val_idx]

    X_pos_train = feats_train[label_train]      # hallucinated answers
    X_neg_train = feats_train[~label_train]     # correct answers
    X_pos_val   = feats_val[label_val]
    X_neg_val   = feats_val[~label_val]
    print(f"  train: {X_pos_train.shape[0]} pos, {X_neg_train.shape[0]} neg")
    print(f"  val:   {X_pos_val.shape[0]} pos, {X_neg_val.shape[0]} neg")

    # ---- (1) Linear probe ----
    print("[uq] (1) linear probe …")
    w, mu, sigma = linear_probe_ridge(X_pos_train, X_neg_train)
    s_lin_pos = predict_linear(w, mu, sigma, X_pos_val)
    s_lin_neg = predict_linear(w, mu, sigma, X_neg_val)
    auc_lin = _roc_auc(s_lin_neg, s_lin_pos)
    lin_probs = torch.sigmoid(torch.cat([s_lin_neg, s_lin_pos]))
    lin_labels = torch.cat([torch.zeros(s_lin_neg.shape[0]), torch.ones(s_lin_pos.shape[0])])
    ece_lin = _ece(lin_probs, lin_labels)
    print(f"    AUROC={auc_lin:.4f}  ECE={ece_lin:.4f}")

    # ---- (2) Mahalanobis ----
    print("[uq] (2) Mahalanobis …")
    M_pos = mahalanobis(X_neg_train, X_pos_val)
    M_neg = mahalanobis(X_neg_train, X_neg_val)
    auc_maha = _roc_auc(M_neg, M_pos)
    print(f"    AUROC={auc_maha:.4f}  (zero-supervised)")

    # ---- (3) Bayesian LR Laplace ----
    print("[uq] (3) Bayesian LR Laplace …")
    w_blr, post_cov, mu_blr, sigma_blr = train_blr_laplace(
        X_pos_train, X_neg_train, sigma2_prior=1.0, n_iters=30,
    )
    p_blr_pos, m_blr_pos, s_blr_pos_std = predict_blr(
        w_blr, post_cov, mu_blr, sigma_blr, X_pos_val
    )
    p_blr_neg, m_blr_neg, s_blr_neg_std = predict_blr(
        w_blr, post_cov, mu_blr, sigma_blr, X_neg_val
    )
    auc_blr_prob = _roc_auc(p_blr_neg, p_blr_pos)
    auc_blr_mean = _roc_auc(m_blr_neg, m_blr_pos)
    auc_blr_std = _roc_auc(s_blr_neg_std, s_blr_pos_std)
    blr_probs = torch.cat([p_blr_neg, p_blr_pos])
    blr_labels = torch.cat([torch.zeros(p_blr_neg.shape[0]), torch.ones(p_blr_pos.shape[0])])
    ece_blr = _ece(blr_probs, blr_labels)
    print(f"    AUROC(prob)={auc_blr_prob:.4f}  AUROC(mean)={auc_blr_mean:.4f}  "
          f"AUROC(std as score)={auc_blr_std:.4f}  ECE={ece_blr:.4f}")

    # ---- (4) Deep ensemble ----
    print("[uq] (4) deep ensemble × 5 …")
    ensemble = deep_ensemble_linear(X_pos_train, X_neg_train, n_models=5, seed=args.seed)
    e_mean_pos, e_std_pos = predict_ensemble(ensemble, X_pos_val)
    e_mean_neg, e_std_neg = predict_ensemble(ensemble, X_neg_val)
    auc_ens_mean = _roc_auc(e_mean_neg, e_mean_pos)
    auc_ens_std = _roc_auc(e_std_neg, e_std_pos)
    ens_probs = torch.sigmoid(torch.cat([e_mean_neg, e_mean_pos]))
    ens_labels = torch.cat([torch.zeros(e_mean_neg.shape[0]), torch.ones(e_mean_pos.shape[0])])
    ece_ens = _ece(ens_probs, ens_labels)
    print(f"    AUROC(mean)={auc_ens_mean:.4f}  AUROC(std)={auc_ens_std:.4f}  ECE={ece_ens:.4f}")

    # ---- (5) SVGP ----
    print(f"[uq] (5) SVGP ({args.svgp_inducing} inducing, {args.svgp_iters} iters) …")
    sv_model, sv_lik, sv_mu, sv_sigma = train_svgp(
        X_pos_train, X_neg_train,
        n_inducing=args.svgp_inducing, n_iters=args.svgp_iters,
    )
    p_svgp_pos, m_svgp_pos, s_svgp_pos = predict_svgp(
        sv_model, sv_lik, sv_mu, sv_sigma, X_pos_val
    )
    p_svgp_neg, m_svgp_neg, s_svgp_neg = predict_svgp(
        sv_model, sv_lik, sv_mu, sv_sigma, X_neg_val
    )
    auc_svgp_prob = _roc_auc(p_svgp_neg, p_svgp_pos)
    auc_svgp_mean = _roc_auc(m_svgp_neg, m_svgp_pos)
    auc_svgp_std = _roc_auc(s_svgp_neg, s_svgp_pos)
    svgp_probs = torch.cat([p_svgp_neg, p_svgp_pos])
    svgp_labels = torch.cat([torch.zeros(p_svgp_neg.shape[0]), torch.ones(p_svgp_pos.shape[0])])
    ece_svgp = _ece(svgp_probs, svgp_labels)
    print(f"    AUROC(prob)={auc_svgp_prob:.4f}  AUROC(mean)={auc_svgp_mean:.4f}  "
          f"AUROC(std as score)={auc_svgp_std:.4f}  ECE={ece_svgp:.4f}")

    # ---- (6) Spilled Energy baseline (zero-train, sequence-level) ----
    print("[uq] (6) Spilled Energy seq baseline …")
    se_pos_val_seq = se_seq_val[label_val]
    se_neg_val_seq = se_seq_val[~label_val]
    auc_se = _roc_auc(se_neg_val_seq, se_pos_val_seq)
    print(f"    AUROC={auc_se:.4f} (mean SE across answer-span tokens)")

    # ---- Aggregate ----
    summary = {
        "cache": str(args.cache),
        "lm": cache["lm"],
        "n_train_pos": int(X_pos_train.shape[0]),
        "n_val_pos": int(X_pos_val.shape[0]),
        "pool": args.pool,
        "linear_probe": {
            "auc": float(auc_lin),
            "ece": float(ece_lin),
        },
        "mahalanobis": {
            "auc": float(auc_maha),
        },
        "blr_laplace": {
            "auc_prob": float(auc_blr_prob),
            "auc_mean": float(auc_blr_mean),
            "auc_std":  float(auc_blr_std),
            "ece":      float(ece_blr),
        },
        "ensemble_5": {
            "auc_mean": float(auc_ens_mean),
            "auc_std":  float(auc_ens_std),
            "ece":      float(ece_ens),
        },
        "svgp": {
            "auc_prob": float(auc_svgp_prob),
            "auc_mean": float(auc_svgp_mean),
            "auc_std":  float(auc_svgp_std),
            "ece":      float(ece_svgp),
        },
        "spilled_energy_seq": {
            "auc": float(auc_se),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[uq] wrote {args.out}")
    print()
    print("=== summary ===")
    print(f"  Linear probe:        AUROC={auc_lin:.4f}  ECE={ece_lin:.4f}")
    print(f"  Mahalanobis:         AUROC={auc_maha:.4f}")
    print(f"  Bayesian LR Laplace: AUROC={auc_blr_prob:.4f}  ECE={ece_blr:.4f}")
    print(f"  Ensemble × 5:        AUROC={auc_ens_mean:.4f}  ECE={ece_ens:.4f}")
    print(f"  SVGP:                AUROC={auc_svgp_prob:.4f}  ECE={ece_svgp:.4f}")
    print(f"  Spilled Energy seq:  AUROC={auc_se:.4f}  (zero-train baseline)")


if __name__ == "__main__":
    main()
