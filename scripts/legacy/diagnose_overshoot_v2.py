"""Iteration 2: test NAG variants that specifically counter overshoot.

Builds on diagnose_overshoot.py with additions:
  - Adaptive restart: reset momentum when grad and (x - x_prev) point opposite ways
  - Momentum decay schedule: mu_t = (t-1)/(t+2) (Nesterov classic)
  - Step-size decay: eta_t = eta_0 / sqrt(1 + t / decay)
  - Gradient clipping by per-position norm
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

import aitchinson_flow.models  # noqa: F401
from aitchinson_flow.config import Config
from aitchinson_flow.data.char_window_dataset import CHAR2ID
from aitchinson_flow.models import build_model
from aitchinson_flow.training import build_training_datamodule

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def unigram_stats(ids: torch.Tensor, K: int) -> tuple[float, torch.Tensor]:
    flat = ids.reshape(-1)
    cnt = torch.zeros(K).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    p = (cnt + 1e-9) / cnt.sum()
    return float(-(p * p.log()).sum()), p


def kl_unigram(p_gen: torch.Tensor, p_ref: torch.Tensor) -> float:
    return float((p_gen * (p_gen.log() - p_ref.log())).sum())


def bigram_kl(gen: torch.Tensor, ref: torch.Tensor, K: int) -> float:
    def counts(ids: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(K, K)
        for row in ids.reshape(-1, ids.shape[-1]):
            for a, b in zip(row[:-1], row[1:]):
                c[int(a), int(b)] += 1
        return c
    gc = counts(gen)
    rc = counts(ref)
    gp = (gc + 1e-6) / (gc.sum() + K * K * 1e-6)
    rp = (rc + 1e-6) / (rc.sum() + K * K * 1e-6)
    return float((gp * (gp.log() - rp.log())).sum())


def sample_advanced(
    model,
    *,
    B: int,
    L: int,
    K: int,
    eta0: float,
    mu0: float,
    max_steps: int,
    sigma_init: float,
    seed: int = 42,
    schedule: str = "constant",          # constant | nesterov_classic | adaptive_restart
    eta_decay_steps: int | None = None,  # if set, eta_t = eta0 / sqrt(1 + t/eta_decay_steps)
    grad_clip: float | None = None,      # per-position L2 clip
    return_best: bool = True,
) -> dict:
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(B, L, K, device=device)
    x = x - x.mean(-1, keepdim=True)
    x_prev = x.clone()
    grad = model._compute_grad(x)
    if grad_clip is not None:
        n = grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        grad = grad * (n.clamp(max=grad_clip) / n)

    g0 = grad.norm(dim=-1).mean().item()
    best_x = x.clone()
    best_g = g0
    traj = [g0]

    n_restart = 0
    for t in range(max_steps):
        # Schedule mu
        if schedule == "nesterov_classic":
            mu = (t) / (t + 3.0)  # rises from 0 → ~1
        else:
            mu = mu0

        # Schedule eta
        eta = eta0
        if eta_decay_steps is not None:
            eta = eta0 / (1.0 + t / eta_decay_steps) ** 0.5

        x_prev_new = x
        # Adaptive restart: if last update direction (x - x_prev) and -grad
        # point opposite → we're going uphill → reset momentum.
        if schedule == "adaptive_restart" and t > 0:
            update = x - x_prev
            inner = (-grad * update).sum()
            if inner.item() < 0:
                mu = 0.0
                n_restart += 1

        x = x - eta * grad
        lookahead = x + mu * (x - x_prev_new) if mu > 0 else x
        grad = model._compute_grad(lookahead)
        if grad_clip is not None:
            n = grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            grad = grad * (n.clamp(max=grad_clip) / n)
        g = grad.norm(dim=-1).mean().item()
        traj.append(g)
        if g < best_g:
            best_g = g
            best_x = x.clone()
        x_prev = x_prev_new

    return {
        "g0": g0,
        "final_g": traj[-1],
        "min_g": min(traj),
        "argmin_step": int(min(range(len(traj)), key=lambda i: traj[i])),
        "max_after_min_g": max(traj[traj.index(min(traj)):]) if len(traj) > 1 else traj[-1],
        "x": best_x if return_best else x,
        "traj": traj,
        "n_restart": n_restart,
    }


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/baseline_5ep/epoch_final.pt"
    cfg = Config()
    device = cfg.training.device
    model = build_model(cfg).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    K, L = cfg.text8_dataset.K, cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma
    B = 32

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    _, p_ref = unigram_stats(train_ids, K)

    print(f"checkpoint={ckpt}  B={B}  K={K}  L={L}  sigma_init={sigma}")
    header = (
        f"{'config':<48s}"
        f"{'g0':>7s}{'g_min':>8s}{'@step':>7s}{'g_final':>8s}"
        f"{'overshoot':>11s}{'restart':>8s}"
        f"{'H_uni':>7s}{'KL_u':>7s}{'KL_b':>7s}  sample"
    )
    print(header)
    print("-" * len(header))

    configs = [
        # baseline current default
        dict(name="NAG eta=0.1 mu=0.9 200 +best",  eta0=0.1,  mu0=0.9, max_steps=200,
             schedule="constant",          return_best=True),
        # adaptive restart
        dict(name="NAG eta=0.1 adaptive-restart 200",  eta0=0.1, mu0=0.9, max_steps=200,
             schedule="adaptive_restart",  return_best=True),
        # nesterov classic schedule
        dict(name="NAG nesterov-mu eta=0.1 200",      eta0=0.1, mu0=0.0, max_steps=200,
             schedule="nesterov_classic",  return_best=True),
        # eta decay only
        dict(name="NAG eta_decay=50 mu=0.9 200",      eta0=0.1, mu0=0.9, max_steps=200,
             schedule="constant", eta_decay_steps=50, return_best=True),
        # gradient clip
        dict(name="NAG eta=0.1 mu=0.9 clip=1.0 200",  eta0=0.1, mu0=0.9, max_steps=200,
             schedule="constant", grad_clip=1.0, return_best=True),
        # combo
        dict(name="adaptive-restart + eta_decay=50",  eta0=0.1, mu0=0.9, max_steps=200,
             schedule="adaptive_restart", eta_decay_steps=50, return_best=True),
        # plain GD baseline
        dict(name="plain GD eta=0.05 500",           eta0=0.05, mu0=0.0, max_steps=500,
             schedule="constant",          return_best=True),
    ]

    for c in configs:
        t0 = time.time()
        with torch.no_grad():
            r = sample_advanced(model, B=B, L=L, K=K, sigma_init=sigma, **{k: v for k, v in c.items() if k != "name"})
        x = r["x"]
        ids = model.decode_to_logprobs(x).argmax(-1).cpu()
        H, p_gen = unigram_stats(ids, K)
        kl_u = kl_unigram(p_gen, p_ref)
        kl_b = bigram_kl(ids, train_ids, K)
        overshoot = r["max_after_min_g"] - r["min_g"]
        sample = "".join(ALPHABET[int(i)] for i in ids[0])[:30]
        print(
            f"{c['name']:<48s}"
            f"{r['g0']:>7.3f}{r['min_g']:>8.4f}{r['argmin_step']:>7d}"
            f"{r['final_g']:>8.4f}{overshoot:>11.4f}{r['n_restart']:>8d}"
            f"{H:>7.3f}{kl_u:>7.4f}{kl_b:>7.3f}  '{sample}'  ({time.time()-t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
