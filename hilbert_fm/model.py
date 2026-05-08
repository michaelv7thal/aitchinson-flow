"""Small transformer with a softmax head — the x_1-prediction backbone.

Inputs are ``log p_t`` (a continuous distribution), outputs are ``log p_1_hat``.
Time is conditioned via a sinusoidal embedding added to every position after
the input projection. No causal mask: this is a non-autoregressive denoiser.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """``t: (B,)`` in ``[0, 1]`` -> ``(B, dim)`` sin/cos features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        pad = torch.zeros(emb.shape[0], dim - emb.shape[-1], device=t.device, dtype=t.dtype)
        emb = torch.cat([emb, pad], dim=-1)
    return emb


class HilbertFMTransformer(nn.Module):
    """Standard pre-norm transformer encoder with a continuous-distribution input.

    Args:
        K, L:        vocab size and sequence length.
        d_model:     hidden width.
        n_layers:    encoder depth.
        n_heads:     attention heads.
        d_ff:        feed-forward width (defaults to ``4 * d_model``).
        dropout:     dropout probability.
    """

    def __init__(
        self,
        K: int,
        L: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.K = K
        self.L = L
        self.d_model = d_model

        self.in_proj = nn.Linear(K, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        d_ff = d_ff or 4 * d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, K)

    def forward(self, log_pt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``log p_1_hat`` of shape ``(B, L, K)``.

        Args:
            log_pt: ``(B, L, K)`` log-probabilities along the path.
            t:      ``(B,)`` time in ``[0, 1]``.
        """
        h = self.in_proj(log_pt) + self.pos_emb[:, : log_pt.shape[1]]
        t_emb = self.t_mlp(sinusoidal_time_embedding(t, self.d_model))
        h = h + t_emb[:, None, :]
        h = self.encoder(h)
        h = self.norm(h)
        return F.log_softmax(self.head(h), dim=-1)
