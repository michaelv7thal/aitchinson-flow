"""Diagnose sampling: track gradient norm over iterations + test step-size sensitivity.

Tells us whether NAG-GD is converging or just running out of steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

import aitchinson_flow.models  # noqa: F401 — populate registry
from aitchinson_flow.config import Config
from aitchinson_flow.data.char_window_dataset import CHAR2ID, VOCAB_SIZE
from aitchinson_flow.models import build_model
from aitchinson_flow.training import build_training_datamodule

ALPHABET = "".join(sorted(CHAR2ID, key=CHAR2ID.__getitem__))


def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/baseline_5ep/epoch_final.pt"
    print(f"checkpoint={ckpt_path}")

    cfg = Config()
    device = cfg.training.device
    model = build_model(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    K = cfg.text8_dataset.K
    L = cfg.text8_dataset.L
    s = cfg.eqm

    # === Trajectory: track per-position grad norm over iterations ===
    print("\n=== Sampling trajectory (B=64) ===")
    print(f"η={s.sample_eta} μ={s.sample_mu} σ_init={s.source_sigma}")
    B = 64
    x = s.source_sigma * torch.randn(B, L, K, device=device)
    x = x - x.mean(-1, keepdim=True)
    x_last = x.clone()
    grad = model._compute_grad(x)

    for step in range(1, 1001):
        gnorm_per_pos = grad.norm(dim=-1).mean().item()
        if step in (1, 5, 10, 25, 50, 100, 200, 500, 1000):
            print(f"  step {step:4d}: ||grad||_pos = {gnorm_per_pos:.4f}")
        if gnorm_per_pos < 0.01:
            print(f"  converged at step {step}")
            break
        x_last_new = x
        x = x - s.sample_eta * grad
        grad = model._compute_grad(x + s.sample_mu * (x - x_last_new))
        x_last = x_last_new

    # Decode final.
    log_probs = model.decode_to_logprobs(x)
    ids = log_probs.argmax(-1)
    print("\n  3 samples after long run:")
    for row in ids[:3].cpu():
        print(f"    '{''.join(ALPHABET[int(i)] for i in row)}'")

    # === Step-size sweep ===
    print("\n=== η sweep (200 steps each) ===")
    for eta in [0.05, 0.1, 0.2, 0.5, 1.0]:
        x_init = s.source_sigma * torch.randn(B, L, K, device=device)
        x_init = x_init - x_init.mean(-1, keepdim=True)
        with torch.no_grad():
            x_out = model.sample(B, L, eta=eta, max_steps=200, x_init=x_init.clone())
        gnorm = model.position_uncertainty(x_out).mean().item()
        log_probs = model.decode_to_logprobs(x_out)
        ids = log_probs.argmax(-1).cpu()
        # unigram
        flat = ids.reshape(-1)
        cnt = torch.zeros(K).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        p = (cnt + 1e-9) / cnt.sum()
        H = float(-(p * p.log()).sum())
        print(f"  η={eta:.2f}: ||grad||={gnorm:.4f}  H_unigram={H:.3f}  '{''.join(ALPHABET[int(i)] for i in ids[0])[:40]}'")

    # === GT-anchored: how close does sampling get when we start near data? ===
    print("\n=== GT-anchored sampling (init = GT + small noise) ===")
    dm, _ = build_training_datamodule(cfg)
    val_features = dm._val_ds._features[:64].to(device)
    for sigma_pert in [0.01, 0.1, 0.5, 1.0, 5.0]:
        x_init = val_features + sigma_pert * torch.randn_like(val_features)
        x_init = x_init - x_init.mean(-1, keepdim=True)
        with torch.no_grad():
            x_out = model.sample(B=64, L=L, max_steps=200, x_init=x_init.clone())
        gnorm = model.position_uncertainty(x_out).mean().item()
        recon_acc = (model.decode_to_logprobs(x_out).argmax(-1) == dm._val_ds._windows[:64].to(device)).float().mean().item()
        print(f"  σ_pert={sigma_pert:.2f}: ||grad||={gnorm:.4f}  reconstruction_acc={recon_acc:.3f}")


if __name__ == "__main__":
    main()
