"""Shared transformer backbone used by all generative models."""

from __future__ import annotations
import math

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .config import Config
from .data.feature_dim import feature_dim


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone.

    log_x (B, L, D) [+ optional t (B,)] → hidden (B, L, d_model)

    Input: feature coordinates (e.g., full ILR), linearly projected into d_model.
    Positional encoding: learned.
    Time conditioning: sinusoidal time embedding added when time_conditioned=True.

    Note: uses the standard (non-flash) attention backend so that
    create_graph=True second-order autograd works through attention.
    """

    def __init__(
        self, cfg: Config, time_conditioned: bool = False, *, sdp_math_for_autograd: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_conditioned = time_conditioned
        self._sdp_math = sdp_math_for_autograd

        in_dim = feature_dim(cfg)
        self.input_proj = nn.Linear(in_dim, cfg.transformer.d_model)
        self.pos_emb = nn.Embedding(cfg.dataset.L, cfg.transformer.d_model)

        if time_conditioned:
            self.time_proj = nn.Sequential(
                nn.Linear(cfg.transformer.d_model, cfg.transformer.d_model),
                nn.SiLU(),
                nn.Linear(cfg.transformer.d_model, cfg.transformer.d_model),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer.d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=cfg.transformer.d_model * 4,
            dropout=cfg.transformer.dropout,
            batch_first=True,
            norm_first=True,  ## Superior training dynamics vs Post-LN
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.transformer.num_layers,
            enable_nested_tensor=False,
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """t (B,) -> sinusoidal embedding (B, d_model)."""
        d = self.cfg.transformer.d_model
        half = d // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if d % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
        return emb

    def forward(self, log_x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            log_x: (B, L, D)
            t:     (B,)  — required when time_conditioned=True, ignored otherwise
        Returns:
            hidden: (B, L, d_model)
        """
        _b, L, _d = log_x.shape
        h = self.input_proj(log_x)
        pos = torch.arange(L, device=log_x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        if self.time_conditioned:
            if t is None:
                raise ValueError("t is required when time_conditioned=True")
            t_emb = self.time_proj(self._sinusoidal(t))
            h = h + t_emb.unsqueeze(1)

        if self._sdp_math:
            with sdpa_kernel(SDPBackend.MATH):
                return self.transformer(h)

        return self.transformer(h)


class VelocityHead(nn.Module):
    """(B, L, d_model) → (B, L, D) — direct velocity output.

    For CLR mode (K-dimensional, constrained): the output is projected onto
    the Aitchison tangent space V_K = {v ∈ R^K : Σv_i = 0} by subtracting
    the feature-dimension mean.

    For ILR mode (K-1 dimensional, unconstrained): the ILR transform already
    maps V_K → R^(K-1), removing the sum-to-zero constraint. Velocity vectors
    in R^(K-1) require no further projection, so the mean subtraction is
    skipped.

    The output dimension matches the data feature dim (``K-1`` for ILR,
    ``K`` for CLR; see ``aitchinson_flow.data.feature_dim``).
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        out_dim = feature_dim(cfg)
        self.proj = nn.Linear(cfg.transformer.d_model, out_dim)
        self._clr_mode = cfg.hf_dataset.transform_mode.lower() == "clr"

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = self.proj(h)
        if self._clr_mode:
            v = v - v.mean(dim=-1, keepdim=True)
        return v


class MaskReconHead(nn.Module):
    """(B, L, d_model) → (B, L, D) — masked-reconstruction projection.

    Structurally identical to ``VelocityHead`` (linear projection onto the
    feature dimension + optional CLR zero-sum centering) but semantically
    distinct: the outputs are clean log-simplex feature targets used by the
    Stage 1 MLM-style auxiliary objective, not velocities. Keeping it as a
    separate module makes the Stage 1 state dict unambiguous and allows
    composition code to treat the two heads independently.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        out_dim = feature_dim(cfg)
        self.proj = nn.Linear(cfg.transformer.d_model, out_dim)
        self._clr_mode = cfg.hf_dataset.transform_mode.lower() == "clr"

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = self.proj(h)
        if self._clr_mode:
            v = v - v.mean(dim=-1, keepdim=True)
        return v


class TokenLatentHead(nn.Module):
    """(B, L, d_model) → (B, L, d_latent) — per-token linear projection.

    Used when downstream consumers (Stage 2 token-level GP, composed
    `BayesianAuditor` per-token queries) need one latent per position.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.transformer.d_model, cfg.transformer.d_latent)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class PooledLatentHead(nn.Module):
    """(B, L, d_model) → (B, d_latent) — mean-pool across sequence + linear projection.

    Used when downstream consumers need a single sequence-level latent. Because
    ``nn.Linear`` is affine, this is numerically identical to
    ``TokenLatentHead(h).mean(dim=1)``; the two classes share the same
    ``self.proj`` parameter layout (``proj.weight``/``proj.bias``), so their
    state dicts are interchangeable.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.transformer.d_model, cfg.transformer.d_latent)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h.mean(dim=1))


# Backwards-compatible alias for external code that imported the old name.
# New code should choose TokenLatentHead or PooledLatentHead explicitly.
LatentHead = PooledLatentHead
