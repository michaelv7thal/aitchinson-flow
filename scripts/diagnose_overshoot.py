"""Faster sampler diagnostic that focuses on detecting NAG overshoot.

For each config we record:
  - Per-step max ||grad|| trajectory (overshoot = bouncing up after going down)
  - Final ||grad||, unigram H, KL_unigram, KL_bigram
  - "best-iterate" vs final-iterate divergence (large gap → overshoot)

B=32, fewer configs than diagnose_sampler_variants.py so the run finishes
inside a session.
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


def run_sampler(
    model,
    *,
    B: int,
    L: int,
    K: int,
    eta: float,
    mu: float,
    max_steps: int,
    sigma_init: float,
    return_best: bool,
    seed: int = 42,
    track_traj: bool = True,
) -> dict:
    """One NAG/GD run. Records grad-norm trajectory + final + best-iterate."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    x = sigma_init * torch.randn(B, L, K, device=device)
    x = x - x.mean(-1, keepdim=True)
    x_prev = x.clone()
    grad = model._compute_grad(x)
    g0 = grad.norm(dim=-1).mean().item()

    best_x = x.clone()
    best_g = g0
    traj = [g0]
    final_x = x.clone()

    for _ in range(max_steps):
        x_prev_new = x
        x = x - eta * grad
        if mu > 0:
            grad = model._compute_grad(x + mu * (x - x_prev_new))
        else:
            grad = model._compute_grad(x)
        g = grad.norm(dim=-1).mean().item()
        if track_traj:
            traj.append(g)
        if g < best_g:
            best_g = g
            best_x = x.clone()
        x_prev = x_prev_new
        final_x = x

    out_x = best_x if return_best else final_x
    return {
        "final_g": traj[-1],
        "min_g": min(traj),
        "argmin_step": int(min(range(len(traj)), key=lambda i: traj[i])),
        "max_after_min_g": max(traj[traj.index(min(traj)):]) if len(traj) > 1 else traj[-1],
        "g0": g0,
        "x": out_x,
        "traj": traj,
    }


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/baseline_5ep/epoch_final.pt"
    cfg = Config()
    device = cfg.training.device
    model = build_model(cfg).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    sigma = cfg.eqm.source_sigma
    B = 32

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()
    _, p_ref = unigram_stats(train_ids, K)

    print(f"checkpoint={ckpt}  B={B}  K={K}  L={L}  sigma_init={sigma}")
    print()
    header = (
        f"{'config':<36s}"
        f"{'g0':>7s}{'g_min':>8s}{'@step':>7s}{'g_final':>8s}"
        f"{'overshoot':>11s}{'H_uni':>7s}{'KL_u':>7s}{'KL_b':>7s}"
        f"  sample"
    )
    print(header)
    print("-" * len(header))

    # Configs:
    #   (name, eta, mu, steps, return_best)
    configs = [
        ("NAG eta=0.1 mu=0.9 200 (cur)",   0.1,  0.9, 200, False),
        ("NAG eta=0.1 mu=0.9 200 +best",   0.1,  0.9, 200, True),
        ("NAG eta=0.1 mu=0.5 200 +best",   0.1,  0.5, 200, True),
        ("NAG eta=0.05 mu=0.9 200 +best",  0.05, 0.9, 200, True),
        ("plain GD eta=0.1  200",          0.1,  0.0, 200, False),
        ("plain GD eta=0.1  500",          0.1,  0.0, 500, False),
        ("plain GD eta=0.05 500",          0.05, 0.0, 500, False),
    ]

    rows = []
    for name, eta, mu, steps, ret_best in configs:
        t0 = time.time()
        with torch.no_grad():
            r = run_sampler(
                model, B=B, L=L, K=K,
                eta=eta, mu=mu, max_steps=steps,
                sigma_init=sigma, return_best=ret_best,
            )
        x = r["x"]
        ids = model.decode_to_logprobs(x).argmax(-1).cpu()
        H, p_gen = unigram_stats(ids, K)
        kl_u = kl_unigram(p_gen, p_ref)
        kl_b = bigram_kl(ids, train_ids, K)
        overshoot = r["max_after_min_g"] - r["min_g"]
        sample = "".join(ALPHABET[int(i)] for i in ids[0])[:30]
        rows.append((name, r, H, kl_u, kl_b, sample))
        print(
            f"{name:<36s}"
            f"{r['g0']:>7.3f}{r['min_g']:>8.4f}{r['argmin_step']:>7d}"
            f"{r['final_g']:>8.4f}{overshoot:>11.4f}"
            f"{H:>7.3f}{kl_u:>7.4f}{kl_b:>7.3f}  '{sample}'  ({time.time()-t0:.1f}s)"
        )

    # Print trajectory of the (current) NAG config
    print("\n=== grad-norm trajectory for current NAG config (eta=0.1 mu=0.9) ===")
    name, r, H, kl_u, kl_b, _ = rows[0]
    traj = r["traj"]
    sampled = [traj[i] for i in (0, 1, 2, 5, 10, 20, 50, 100, 150, 199)]
    for i, g in zip([0, 1, 2, 5, 10, 20, 50, 100, 150, 199], sampled):
        print(f"  step {i:3d}: ||grad|| = {g:.4f}")


if __name__ == "__main__":
    main()
