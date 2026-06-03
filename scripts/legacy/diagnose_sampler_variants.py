"""Test sampler variants: plain GD, NAG-with-best-iterate, NAG-with-restart, etc."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

import aitchinson_flow.models  # noqa: F401
from aitchinson_flow.config import Config
from aitchinson_flow.data.char_window_dataset import CHAR2ID, VOCAB_SIZE
from aitchinson_flow.models import build_model
from aitchinson_flow.training import build_training_datamodule

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def unigram_H(ids: torch.Tensor, K: int) -> float:
    flat = ids.reshape(-1)
    cnt = torch.zeros(K).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    p = (cnt + 1e-9) / cnt.sum()
    return float(-(p * p.log()).sum())


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


def gd_sample(model, B, L, K, *, eta, mu, max_steps, sigma_init, return_best: bool):
    """Plain GD or NAG with optional best-iterate tracking."""
    device = next(model.parameters()).device
    x = sigma_init * torch.randn(B, L, K, device=device)
    x = x - x.mean(-1, keepdim=True)
    x_last = x.clone()
    grad = model._compute_grad(x)
    best_x = x.clone()
    best_g = grad.norm(dim=-1).mean().item()
    for _ in range(max_steps):
        x_last_new = x
        x = x - eta * grad
        if mu > 0:
            grad = model._compute_grad(x + mu * (x - x_last_new))
        else:
            grad = model._compute_grad(x)
        gnorm = grad.norm(dim=-1).mean().item()
        if return_best and gnorm < best_g:
            best_g = gnorm
            best_x = x.clone()
        x_last = x_last_new
    return best_x if return_best else x


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
    B = 256

    dm, _ = build_training_datamodule(cfg)
    train_ids = dm.splits.train.long()

    print(f"checkpoint={ckpt}  B={B}  K={K}  L={L}")
    print(f"\n{'config':<40s}{'||grad||':>10s}{'H_uni':>8s}{'KL_uni':>8s}{'KL_bi':>8s}  sample")
    print("-" * 100)

    configs = [
        ("NAG η=0.1 μ=0.9 200steps (current)", 0.1, 0.9, 200, False),
        ("NAG η=0.1 μ=0.9 50steps",            0.1, 0.9, 50,  False),
        ("NAG η=0.1 μ=0.9 200steps + best",    0.1, 0.9, 200, True),
        ("plain GD η=0.1 200steps",            0.1, 0.0, 200, False),
        ("plain GD η=0.1 500steps",            0.1, 0.0, 500, False),
        ("plain GD η=0.05 500steps",           0.05, 0.0, 500, False),
        ("plain GD η=0.2 200steps",            0.2, 0.0, 200, False),
        ("NAG η=0.05 μ=0.5 200steps + best",   0.05, 0.5, 200, True),
        ("NAG η=0.05 μ=0.7 200steps + best",   0.05, 0.7, 200, True),
        ("NAG η=0.1 μ=0.5 500steps + best",    0.1, 0.5, 500, True),
    ]

    for name, eta, mu, steps, ret_best in configs:
        torch.manual_seed(42)
        with torch.no_grad():
            x = gd_sample(model, B, L, K, eta=eta, mu=mu, max_steps=steps,
                          sigma_init=sigma, return_best=ret_best)
        gnorm = model.position_uncertainty(x).mean().item()
        ids = model.decode_to_logprobs(x).argmax(-1).cpu()
        # KL(gen || gt) per sample stat
        cnt_g = torch.zeros(K).scatter_add_(0, ids.reshape(-1), torch.ones_like(ids.reshape(-1), dtype=torch.float))
        cnt_r = torch.zeros(K).scatter_add_(0, train_ids.reshape(-1), torch.ones_like(train_ids.reshape(-1), dtype=torch.float))
        pg = (cnt_g + 1e-9) / (cnt_g.sum() + K * 1e-9)
        pr = (cnt_r + 1e-9) / (cnt_r.sum() + K * 1e-9)
        kl_u = float((pg * (pg.log() - pr.log())).sum())
        H_u = unigram_H(ids, K)
        kl_b = bigram_kl(ids, train_ids, K)
        sample = "".join(ALPHABET[int(i)] for i in ids[0])
        print(f"{name:<40s}{gnorm:>10.4f}{H_u:>8.3f}{kl_u:>8.4f}{kl_b:>8.3f}  '{sample[:30]}'")


if __name__ == "__main__":
    main()
