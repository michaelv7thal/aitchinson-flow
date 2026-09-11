"""Per-position Bayes floor for the EqM regression target — the line that
separates "uses the neighbours" from "does not".

The flow loss alone cannot say whether a cell learned anything, because moving
γ into the strip also raises the *irreducible* error: c(γ) is largest exactly
where the interpolant is least informative. This computes the reference.

For the deterministic-CLR path with x_γ = (1-γ)x_0 + γ x_1 and x_0 ~ σN(0,I)
projected to the zero-mean subspace, the target is

    u = c(γ)·(x_0 - x_1) = c(γ)/(1-γ)·(x_γ - x_1),

so the optimal prediction is c/(1-γ)·(x_γ - E[x_1|·]). Both x_1 = v(k) and the
noise live in the zero-mean subspace and the noise covariance does not depend
on k, so the exact per-position posterior is a plain softmax:

    log p(k | x_{γ,i}) = log p_uni(k) - ‖x_{γ,i} - γ v(k)‖² / (2(1-γ)²σ²) + const

Evaluating that gives the loss of the best predictor that sees ONE position and
is additionally *told* γ, which the trained field is not. It is therefore a
LOWER bound on what any position-wise field can achieve, and the test is
one-sided but sound:

    model loss  <  floor   ⇒  the field is using the neighbours
    model loss  ≈  floor   ⇒  position-wise only
    model loss  >  floor   ⇒  has not even reached the position-wise optimum

Also reports the no-information reference (always predict E[x_1] = μ_1) and a
per-γ breakdown, which shows where each schedule's loss actually comes from.

Run (CPU, seconds):
    uv run python scripts/band_bayes_floor.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from aitchinson_flow.config import Config  # noqa: E402
from aitchinson_flow.data.transforms import token_ids_to_features  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402

# (name, gamma_lo, gamma_hi, gamma_star or None for the linear 1-γ schedule)
CELLS = [
    ("ctrl_noce", 0.0, 1.0, None),
    ("band_g05", 0.0, 0.05, 0.05),
    ("band_strict", 0.005, 0.03, 0.03),
]
GRID = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04,
        0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def c_of(gamma: float, gstar: float | None) -> float:
    return (1.0 - gamma) if gstar is None else max(0.0, 1.0 - gamma / gstar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="windows per γ")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config()
    dm, _ = build_training_datamodule(cfg)
    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma
    eps = cfg.transformation.label_smoothing

    ids = dm.splits.val.long()[: args.n]                      # (N, L)
    x1 = token_ids_to_features(ids, K, label_smoothing=eps)   # (N, L, K)
    V = token_ids_to_features(torch.arange(K).unsqueeze(0), K,
                              label_smoothing=eps)[0]         # (K, K) rows = v(k)

    train_ids = dm.splits.train.long().reshape(-1)
    counts = torch.bincount(train_ids, minlength=K).double()
    p_uni = (counts / counts.sum()).float()
    log_prior = p_uni.log()
    mu1 = p_uni @ V                                           # E[x_1], the unigram tilt

    print(f"K={K} L={L} σ={sigma} windows={len(ids)}")
    print(f"‖v(k)‖={V.norm(dim=-1).mean():.4f}  ‖μ1‖={mu1.norm():.4f}  "
          f"Σp²={(p_uni**2).sum():.4f}  H_uni={-(p_uni*log_prior).sum():.4f} nats")

    g = torch.Generator().manual_seed(args.seed)
    rows = {}
    print(f"\n{'γ':>7}{'acc_in':>9}{'floor_1':>10}{'noinfo_1':>10}  "
          f"(per-coordinate MSE at c=1)")
    for gam in GRID:
        x0 = sigma * torch.randn(x1.shape, generator=g)
        x0 = x0 - x0.mean(-1, keepdim=True)
        xg = (1.0 - gam) * x0 + gam * x1

        # exact per-position posterior over the K candidate vertices
        d2 = ((xg.unsqueeze(-2) - gam * V) ** 2).sum(-1)      # (N, L, K)
        denom = 2.0 * max((1.0 - gam) ** 2 * sigma ** 2, 1e-12)
        post = torch.softmax(log_prior - d2 / denom, dim=-1)
        x1_hat = post @ V                                     # (N, L, K)

        scale = 1.0 / max(1.0 - gam, 1e-6)                    # c/(1-γ), at c=1
        floor = (scale ** 2 * (x1 - x1_hat) ** 2).mean().item()
        noinfo = (scale ** 2 * (x1 - mu1) ** 2).mean().item()
        acc = (xg.argmax(-1) == ids).float().mean().item()
        rows[gam] = (acc, floor, noinfo)
        print(f"{gam:>7.4f}{acc:>9.4f}{floor:>10.4f}{noinfo:>10.4f}")

    print(f"\n{'cell':14}{'γ range':>14}{'floor':>9}{'no-info':>9}   "
          f"share of floor from γ<0.005")
    out = {}
    for name, lo, hi, gstar in CELLS:
        gs = [x for x in GRID if lo - 1e-9 <= x <= hi + 1e-9]
        w = []
        for i, x in enumerate(gs):                            # trapezoid weights
            a = gs[i - 1] if i else x
            b = gs[i + 1] if i + 1 < len(gs) else x
            w.append((b - a) / 2 if 0 < i < len(gs) - 1 else (b - a) if i == 0 else (x - a))
        tot = sum(w) or 1.0
        fl = sum(wi * c_of(x, gstar) ** 2 * rows[x][1] for wi, x in zip(w, gs)) / tot
        ni = sum(wi * c_of(x, gstar) ** 2 * rows[x][2] for wi, x in zip(w, gs)) / tot
        dead = sum(wi * c_of(x, gstar) ** 2 * rows[x][1]
                   for wi, x in zip(w, gs) if x < 0.005) / tot
        out[name] = {"floor": fl, "noinfo": ni, "dead_share": dead / fl if fl else 0.0}
        print(f"{name:14}{f'[{lo},{hi}]':>14}{fl:>9.4f}{ni:>9.4f}"
              f"{dead / fl if fl else 0:>14.2f}")

    json.dump({"per_gamma": {str(k): v for k, v in rows.items()}, "cells": out},
              open(ROOT / "runs" / "band_bayes_floor.json", "w"), indent=2)
    print(f"\nwrote {ROOT / 'runs' / 'band_bayes_floor.json'}")
    print("\nCompare each cell's floor with its converged flow_loss:")
    print("  below the floor ⇒ neighbours are being used (the floor already grants γ)")


if __name__ == "__main__":
    main()
