"""Equilibrium Matching (EqM): time-independent velocity field — adapted from legacy EquilibriumFlow."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.loss import build_velocity_loss
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.factory import register
from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead


def _uniform_log_x0(B: int, L: int, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((B, L, K), math.log(1.0 / K), device=device, dtype=dtype)


class EquilibriumAuditor(nn.Module):
    """
    Equilibrium Matching: predictor is a time-independent velocity on ``log_x``.

    Network: ``TransformerBackbone(time_conditioned=False)`` + ``VelocityHead``.
    Target: ``c_t * (log_x0 - log_x1)`` on the linear path ``log_xt = (1-t) log_x0 + t log_x1``.
    Optional batch key ``valid_mask`` of shape ``(B, L)`` selects positions (True = included in loss).
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        eq = cfg.equilibrium
        if not (0.0 <= eq.eqm_interp < 1.0):
            raise ValueError(f"equilibrium.eqm_interp must be in [0, 1), got {eq.eqm_interp}")
        self._scale = 1.0 / (1.0 - eq.eqm_interp)

        self.backbone = TransformerBackbone(cfg=cfg, time_conditioned=False)
        self.velocity_head = VelocityHead(cfg=cfg)
        self._loss_fn = build_velocity_loss(
            cfg.training.velocity_loss, soft_hilbert_alpha=cfg.training.soft_hilbert_alpha
        )

    def forward(self, log_x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """``log_x`` (B, L, K) → velocity (B, L, K). ``t`` is ignored (time-independent field)."""
        del t
        return self.velocity_head(self.backbone(log_x))

    def _c_t(self, t: torch.Tensor) -> torch.Tensor:
        """``c_t`` schedule as (B, 1, 1); downweights signal toward t → 1 (legacy)."""
        cap = torch.as_tensor(self.cfg.equilibrium.eqm_start, device=t.device, dtype=t.dtype)
        ct = torch.clamp(torch.minimum(cap, self._scale * (1.0 - t)), min=0.0)
        return ct[:, None, None] * 4.0

    def _eqm_loss(self, log_x1: torch.Tensor, valid_mask: torch.Tensor | None) -> LossDict:
        B, L, K = log_x1.shape
        device, dt = log_x1.device, log_x1.dtype
        log_x0 = _uniform_log_x0(B, L, K, device, dt)
        t = torch.rand(B, device=device, dtype=dt)
        log_xt = (1.0 - t[:, None, None]) * log_x0 + t[:, None, None] * log_x1
        u_tgt = self._c_t(t) * (log_x0 - log_x1)
        v_pred = self.forward(log_xt)

        if valid_mask is not None:
            if valid_mask.shape != (B, L):
                raise ValueError(f"valid_mask must be (B, L), got {tuple(valid_mask.shape)}")
            m = valid_mask.unsqueeze(-1).expand(B, L, K).to(dtype=torch.bool)
            total = self._loss_fn(v_pred[m], u_tgt[m])
        else:
            total = self._loss_fn(v_pred, u_tgt)

        return {TRAINING_LOSS_KEY: total, "eqm_loss": total.detach()}

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        vm = batch.get("valid_mask")
        return self._eqm_loss(batch["log_x"], vm)

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        vm = batch.get("valid_mask")
        return self._eqm_loss(batch["log_x"], vm)

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        return self.eval_step(batch)

    @torch.no_grad()
    def generate(
        self,
        n: int,
        steps: int | None = None,
        stepsize: float | None = None,
    ) -> torch.Tensor:
        """Pure fixed-point style updates in log-space: ``x ← x - v(x) * stepsize``; return discrete ids."""
        eq = self.cfg.equilibrium
        steps = steps if steps is not None else eq.generate_steps
        stepsize = stepsize if stepsize is not None else eq.generate_stepsize
        device = self.cfg.training.device
        L, K = self.cfg.dataset.L, self.cfg.dataset.K
        dtype = torch.float32

        self.eval()
        x = _uniform_log_x0(n, L, K, device, dtype)
        x = x + eq.generate_init_noise * torch.randn_like(x)
        for _ in range(steps):
            x = x - self.forward(x) * stepsize
        self.train()
        return x.softmax(dim=-1).argmax(dim=-1)


@register("equilibrium")
def build_equilibrium_auditor(cfg: Config) -> EquilibriumAuditor:
    return EquilibriumAuditor(cfg)
