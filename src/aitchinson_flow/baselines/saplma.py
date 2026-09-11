"""Phase V (CAPSTONE_EXPERIMENTS.md §8) — SAPLMA-style probe.

Per Azaria & Mitchell 2023 §3:
* 3-layer MLP, hidden dims 256 → 128 → 64, ReLU, sigmoid output.
* Input: GPT-2 last-hidden-state (768-dim) at each position.
* Trained on per-token labels (corrupted vs clean) with BCE.
* Inference: per-token sigmoid + mean-pool over the answer span (the
  published per-claim aggregation). For Phase U we keep the per-position
  scores for direct comparison with the linear-probe / SE columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaplmaProbe(nn.Module):
    """3-layer MLP per Azaria & Mitchell 2023."""

    def __init__(self, in_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def train_saplma(
    cache: dict[str, Any],
    *,
    epochs: int = 25,
    lr: float = 3e-4,
    batch_size: int = 4096,
    train_frac: float = 0.8,
    device: str = "cuda",
    seed: int = 1234,
) -> SaplmaProbe:
    """Train a SaplmaProbe on a wiki cache. Returns the trained probe."""
    h_clean = cache["clean_h"].float()
    h_invalid = cache["invalid_h"].float()
    mask_corrupt = cache["mask_corrupt"].bool()

    H = h_clean.shape[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = h_clean.shape[0]
    n_train = int(train_frac * n)
    perm = torch.randperm(n, generator=g)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    h_inv_tr = h_invalid[train_idx].flatten(0, 1).to(device)
    y_inv_tr = mask_corrupt[train_idx].flatten().float().to(device)
    h_cln_tr = h_clean[train_idx].flatten(0, 1).to(device)
    y_cln_tr = torch.zeros(h_cln_tr.shape[0], device=device)
    h_all = torch.cat([h_inv_tr, h_cln_tr], 0)
    y_all = torch.cat([y_inv_tr, y_cln_tr], 0)

    probe = SaplmaProbe(in_dim=H).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n_pool = h_all.shape[0]
    for ep in range(epochs):
        permi = torch.randperm(n_pool, device=device)
        ep_loss = 0.0
        for i in range(0, n_pool, batch_size):
            idx = permi[i:i+batch_size]
            logits = probe(h_all[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * idx.numel()
        ep_loss /= max(n_pool, 1)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  saplma ep {ep+1}/{epochs}  loss={ep_loss:.4f}")
    return probe


@torch.no_grad()
def score_saplma(
    ckpt: str | Path, cache: dict[str, Any], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a trained SaplmaProbe and return (B, L) per-position scores
    on (clean, invalid) sequences."""
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    H = payload.get("in_dim", cache["clean_h"].shape[-1])
    probe = SaplmaProbe(in_dim=H).to(device)
    probe.load_state_dict(payload["state_dict"])
    probe.eval()
    h_cln = cache["clean_h"].float().to(device)
    h_inv = cache["invalid_h"].float().to(device)
    B, L, _ = h_cln.shape
    s_cln = probe(h_cln.flatten(0, 1)).view(B, L).cpu()
    s_inv = probe(h_inv.flatten(0, 1)).view(B, L).cpu()
    return s_cln, s_inv
