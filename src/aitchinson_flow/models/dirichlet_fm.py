"""Dirichlet Flow Matching (Stark et al. 2024, arXiv:2402.05841).

The conditional probability path on the simplex is

    p_{t|1}(x | x_1) = Dir(x ; β(t, x_1)),    β_i = α(t)·δ_{i,x_1} + (1 - δ_{i,x_1})

with α(t) = t for t ∈ [1, t_max]. At t=1 this is Dir(1,...,1) (uniform on the
simplex); as t grows the distribution concentrates at e_{x_1}.

Training: sample t ~ U[1, t_max], sample x_t from Dir, predict p(x_1 | x_t, t)
with a cross-entropy denoiser.

Sampling: Euler integration along t ∈ [1, t_max] of the marginal vector field

    u_t(x) = Σ_c p̂(x_1=c | x_t, t) · u_t(x | c)

where the conditional vector field (Theorem 3.1) is

    u_t(x | c) = ċ_t(x_c) · (e_c - x) / (1 - x_c),
    ċ_t(u)    = -∂_α I_u(α(t), K-1) / Beta_pdf(u; α(t), K-1)   · α'(t).

For α(t) = t the time-derivative reduces to the ∂_α partial; we evaluate it
with a central finite difference over `betainc` (scipy — torch.special doesn't
expose betainc).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import betainc, gammaln
from torch.distributions import Dirichlet

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.transformer_backbone import DFMBackbone, DFMHead


def _beta_log_pdf_np(u: np.ndarray, a: float, b: float) -> np.ndarray:
    u_safe = np.clip(u, 1e-10, 1.0 - 1e-10)
    log_B = gammaln(a) + gammaln(b) - gammaln(a + b)
    return (a - 1.0) * np.log(u_safe) + (b - 1.0) * np.log1p(-u_safe) - log_B


def _conditional_velocity_factor(
    x: torch.Tensor, t: float, K: int, eps: float = 1e-3
) -> torch.Tensor:
    """Compute ċ_t(u) = -∂_α I_u(α, K-1) / Beta_pdf(u; α, K-1) at α = t.

    x: (..., K) values in (0, 1). Returns same shape.
    Computed on CPU via scipy; safe to call inside @torch.no_grad sampling.
    """
    device = x.device
    dtype = x.dtype
    u_np = x.detach().cpu().double().numpy()

    a_plus = t + eps
    a_minus = max(t - eps, 1e-3)
    F_plus = betainc(a_plus, K - 1, u_np)
    F_minus = betainc(a_minus, K - 1, u_np)
    dF_da = (F_plus - F_minus) / (a_plus - a_minus)

    log_q = _beta_log_pdf_np(u_np, t, K - 1)
    log_q = np.clip(log_q, -60.0, 60.0)
    q = np.exp(log_q)

    factor = -dF_da / np.maximum(q, 1e-12)
    return torch.from_numpy(factor).to(device=device, dtype=dtype)


class DirichletFlowMatching(nn.Module):
    """Dirichlet Flow Matching as defined in Stark et al. (arXiv:2402.05841)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DFMBackbone(cfg=cfg)
        self.head = DFMHead(cfg=cfg)

    @property
    def t_max(self) -> float:
        return float(self.cfg.dirichlet_fm.t_max)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, L, K) on simplex, t: (B,) ∈ [1, t_max] → logits (B, L, K)."""
        # Backbone's sinusoidal embedding expects t in roughly [0, 1].
        denom = max(self.t_max - 1.0, 1.0)
        t_norm = ((t - 1.0) / denom).to(dtype=x_t.dtype)
        h = self.backbone(x_t, t_norm)
        return self.head(h)

    def _sample_xt(
        self, token_ids: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Sample x_t ~ Dir(β(t, token_ids)).

        token_ids: (B, L) long; t: (B,) float in [1, t_max].
        Returns (B, L, K) simplex points.
        """
        B, L = token_ids.shape
        K = self.cfg.text8_dataset.K
        beta = torch.ones(B, L, K, device=token_ids.device, dtype=t.dtype)
        idx = token_ids.unsqueeze(-1)
        beta_target = t[:, None, None].expand(B, L, 1).to(beta.dtype)
        beta = beta.scatter(-1, idx, beta_target)
        return Dirichlet(beta).sample()

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        token_ids = batch["token_ids"].long()
        B = token_ids.shape[0]
        device = token_ids.device
        K = self.cfg.text8_dataset.K

        t = 1.0 + (self.t_max - 1.0) * torch.rand(B, device=device)
        x_t = self._sample_xt(token_ids, t)
        logits = self.forward(x_t, t)
        loss = F.cross_entropy(logits.reshape(-1, K), token_ids.reshape(-1))
        return {TRAINING_LOSS_KEY: loss}

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        token_ids = batch["token_ids"].long()
        B = token_ids.shape[0]
        device = token_ids.device
        K = self.cfg.text8_dataset.K

        t = 1.0 + (self.t_max - 1.0) * torch.rand(B, device=device)
        x_t = self._sample_xt(token_ids, t)
        logits = self.forward(x_t, t)
        loss = F.cross_entropy(logits.reshape(-1, K), token_ids.reshape(-1))
        return {TRAINING_LOSS_KEY: loss}

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        nfe: int | None = None,
        x_init: torch.Tensor | None = None,
        t_start: float | None = None,
        max_steps: int | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Euler integration of the marginal vector field, t: 1 → t_max.

        Returns argmax token IDs (B, L).

        Optional ``x_init`` + ``t_start`` enable the *partial-path*
        recovery test used by :mod:`scripts.eval_generation` — instead
        of starting from Dir(1,…,1) at t=1, start from a user-supplied
        simplex point at ``t_start ∈ [1, t_max]`` and integrate forward.
        The natural recovery construction is ``x_init = _sample_xt(
        clean_ids, t_start)`` so the perturbation level is set by the
        choice of ``t_start`` (lower t = noisier, t_max = clean).

        ``max_steps`` is accepted for cross-arm API compatibility with
        ``recovery_check.py`` / ``eval_generation`` and is aliased to
        ``nfe`` when ``nfe`` itself is not provided.
        """
        if nfe is None:
            nfe = max_steps if max_steps is not None else self.cfg.dirichlet_fm.sample_nfe
        K = self.cfg.text8_dataset.K
        device = next(self.parameters()).device
        start_t = float(t_start) if t_start is not None else 1.0
        if not (1.0 <= start_t <= self.t_max):
            raise ValueError(
                f"t_start={start_t} must lie in [1, t_max={self.t_max}]"
            )

        if x_init is not None:
            x = x_init.to(device).to(dtype=torch.float32)
        else:
            # Initial: Dir(1,...,1) = uniform on the simplex.
            x = Dirichlet(torch.ones(B, L, K, device=device)).sample()

        t_grid = torch.linspace(start_t, self.t_max, nfe + 1, device=device)
        for i in range(nfe):
            t = float(t_grid[i].item())
            dt = float((t_grid[i + 1] - t_grid[i]).item())

            t_batch = torch.full((B,), t, device=device, dtype=x.dtype)
            logits = self.forward(x, t_batch)
            p1 = logits.softmax(dim=-1)  # (B, L, K)

            # ċ_t evaluated at every coordinate of x. Shape (B, L, K).
            x_clamped = x.clamp(min=1e-6, max=1.0 - 1e-6)
            c_dot = _conditional_velocity_factor(x_clamped, t, K)

            # u_t(x|c) = ċ_t(x_c) · (e_c - x) / (1 - x_c)
            # Marginal: Σ_c p1[c] · u_t(x|c)
            #         = Σ_c (p1[c] · ċ_t(x_c) / (1 - x_c)) · (e_c - x)
            # Let w[c] = p1[c] · ċ_t(x_c) / (1 - x_c). Then:
            #   Σ_c w[c] · e_c = w     (since e_c are standard basis vectors)
            #   Σ_c w[c] · x   = x · Σ_c w[c]
            denom = (1.0 - x_clamped).clamp(min=1e-6)
            w = p1 * c_dot / denom  # (B, L, K)
            sum_w = w.sum(dim=-1, keepdim=True)
            v = w - x * sum_w

            x = x + dt * v
            # Numerical safety: re-project onto simplex.
            x = x.clamp(min=1e-6)
            x = x / x.sum(dim=-1, keepdim=True)

        return x.argmax(dim=-1)


@register("DirichletFM")
def build_dirichlet_fm(cfg: Config) -> DirichletFlowMatching:
    return DirichletFlowMatching(cfg)
