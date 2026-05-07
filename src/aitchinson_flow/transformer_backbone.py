from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend

from aitchinson_flow.config import Config


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone.

    x (B, L, K) → hidden (B, L, d_model)

    Input: CLR feature coordinates, linearly projected into d_model.
    Positional encoding: learned. Optional sinusoidal γ time-conditioning
    when ``cfg.eqm.time_conditioning`` is "add" or "concat".

    Note: uses the standard (non-flash) attention backend so that
    create_graph=True second-order autograd works through attention.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.transformer.d_model
        self.input_proj = nn.Linear(cfg.text8_dataset.K, d)
        self.pos_emb = nn.Embedding(cfg.text8_dataset.L, d)

        self._time_mode = getattr(cfg.eqm, "time_conditioning", "off")
        if self._time_mode not in ("off", "add", "concat"):
            raise ValueError(
                f"unknown eqm.time_conditioning={self._time_mode!r}; "
                "expected 'off', 'add', or 'concat'"
            )
        if self._time_mode == "add":
            self.t_proj = nn.Linear(d, d)
        elif self._time_mode == "concat":
            self.t_proj = nn.Linear(2 * d, d)
        else:
            self.t_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.transformer.nhead,
            dim_feedforward=d * 4,
            dropout=cfg.transformer.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(d),
            enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
        B, L, K = x.shape
        h = self.input_proj(x)
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)

        if self._time_mode != "off":
            if gamma is None:
                # Sample-time fallback: condition on γ=1 (data-manifold endpoint).
                gamma = torch.ones(B, device=x.device, dtype=x.dtype)
            t_emb = _sinusoidal_embedding(gamma, h.shape[-1])  # (B, d_model)
            if self._time_mode == "add":
                h = h + self.t_proj(t_emb)[:, None, :]
            else:  # "concat"
                t_broadcast = t_emb[:, None, :].expand(B, L, -1)
                h = self.t_proj(torch.cat([h, t_broadcast], dim=-1))

        with sdpa_kernel(SDPBackend.MATH):
            return self.transformer(h)


class VelocityHead(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.transformer.d_model, cfg.text8_dataset.K)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = self.proj(h)
        return v - v.mean(dim=-1, keepdim=True)


class PooledLatentHead(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.transformer.d_model, cfg.transformer.d_latent)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h.mean(dim=1))


def _sinusoidal_embedding(t: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal timestep embedding.

    t: (B,) float in [0, 1]
    returns: (B, d_model)
    """
    assert d_model % 2 == 0
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)


class DFMBackbone(nn.Module):
    """Transformer encoder with sinusoidal timestep conditioning for DFM.

    (x_onehot: B,L,K), (t: B,) → hidden (B, L, d_model)

    Uses standard SDP (no MATH backend needed — no second-order autograd).
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.transformer.d_model
        self.input_proj = nn.Linear(cfg.text8_dataset.K, d)
        self.pos_emb = nn.Embedding(cfg.text8_dataset.L, d)
        self.t_proj = nn.Linear(d, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.transformer.nhead,
            dim_feedforward=d * 4,
            dropout=cfg.dfm.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(d),
            enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, L, K) one-hot floats, t: (B,) floats in [0, 1]."""
        B, L, K = x.shape
        h = self.input_proj(x)
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        t_emb = self.t_proj(_sinusoidal_embedding(t, h.shape[-1]))  # (B, d_model)
        h = h + t_emb[:, None, :]  # broadcast across sequence length
        return self.transformer(h)


class DFMHead(nn.Module):
    """Linear head projecting d_model → K logits (no centering)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.transformer.d_model, cfg.text8_dataset.K)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
