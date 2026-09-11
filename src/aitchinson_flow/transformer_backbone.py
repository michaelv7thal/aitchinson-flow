from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt
from torch.nn.attention import sdpa_kernel, SDPBackend

from aitchinson_flow.config import Config


def run_encoder(
    encoder: nn.TransformerEncoder,
    h: torch.Tensor,
    *,
    grad_checkpointing: bool = False,
) -> torch.Tensor:
    """Run a ``nn.TransformerEncoder`` under the MATH SDPA backend, optionally
    with per-layer activation checkpointing.

    The MATH backend is mandatory across the EqM family: FlashAttention does
    not support the ``create_graph=True`` second-order autograd of the
    conservative-gradient path (see CLAUDE.md). When ``grad_checkpointing`` is
    on AND we are in a grad-tracking forward, each encoder layer is wrapped in
    ``torch.utils.checkpoint`` (``use_reentrant=False`` so it composes with
    second-order autograd); the MATH context is re-entered inside the
    checkpointed callable so it is active on the backward recompute too. The
    result is numerically identical to the non-checkpointed path."""
    use_ck = grad_checkpointing and torch.is_grad_enabled() and h.requires_grad
    if not use_ck:
        with sdpa_kernel(SDPBackend.MATH):
            return encoder(h)
    for layer in encoder.layers:
        def _run(x, _layer=layer):
            with sdpa_kernel(SDPBackend.MATH):
                return _layer(x)
        h = _ckpt.checkpoint(_run, h, use_reentrant=False)
    if encoder.norm is not None:
        h = encoder.norm(h)
    return h


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone.

    x (B, L, K) → hidden (B, L, d_model)

    Input: CLR feature coordinates, linearly projected into d_model.
    Positional encoding: learned. Optional sinusoidal γ time-conditioning
    when ``cfg.eqm.time_conditioning`` is "add" or "concat". Optional
    LM-hidden context conditioning when ``cfg.eqm.context_features`` is
    "hidden_only" or "product_concat" (Phase F auditor).

    Note: uses the standard (non-flash) attention backend so that
    create_graph=True second-order autograd works through attention.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.transformer.d_model
        K = cfg.text8_dataset.K
        self._d_model = d
        self._K = K

        self._ctx_mode = getattr(cfg.eqm, "context_features", "off")
        if self._ctx_mode not in ("off", "hidden_only", "product_concat"):
            raise ValueError(
                f"unknown eqm.context_features={self._ctx_mode!r}; "
                "expected 'off', 'hidden_only', or 'product_concat'"
            )
        ctx_hidden = int(getattr(cfg.eqm, "ctx_hidden", 768))
        ctx_proj_dim = getattr(cfg.eqm, "ctx_proj_dim", None) or (d // 2)

        if self._ctx_mode == "off":
            self.input_proj = nn.Linear(K, d)
            self.h_proj = None
        elif self._ctx_mode == "hidden_only":
            self.input_proj = nn.Linear(ctx_hidden, d)
            self.h_proj = None
        elif self._ctx_mode == "product_concat":
            self.h_proj = nn.Linear(ctx_hidden, ctx_proj_dim)
            self.input_proj = nn.Linear(K + ctx_proj_dim, d)

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

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        if self._ctx_mode == "off":
            h = self.input_proj(x)
        elif self._ctx_mode == "hidden_only":
            if h_ctx is None:
                raise ValueError(
                    "context_features='hidden_only' requires h_ctx; got None"
                )
            h = self.input_proj(h_ctx.to(dtype=x.dtype))
        else:  # "product_concat"
            if h_ctx is None:
                raise ValueError(
                    "context_features='product_concat' requires h_ctx; got None"
                )
            h_proj = self.h_proj(h_ctx.to(dtype=x.dtype))
            h = self.input_proj(torch.cat([x, h_proj], dim=-1))

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

        return run_encoder(
            self.transformer, h,
            grad_checkpointing=self.cfg.transformer.grad_checkpointing,
        )


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
