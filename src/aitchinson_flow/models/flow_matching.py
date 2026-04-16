"""Direct velocity flow-matching auditor: time-conditioned backbone + velocity head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.loss import build_velocity_loss
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.factory import register
from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead


def _uniform_log_x0(B: int, L: int, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((B, L, D), device=device, dtype=dtype)


class FlowMatchingAuditor(nn.Module):
    """Supervise predicted velocity against linear path in log-simplex (CFM-style, no GP)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._velocity_loss_fn = build_velocity_loss(
            cfg.training.velocity_loss, soft_hilbert_alpha=cfg.training.soft_hilbert_alpha
        )
        self.backbone = TransformerBackbone(cfg=cfg, time_conditioned=True)
        self.head = VelocityHead(cfg=cfg)

    def _predict_velocity(self, log_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.backbone(log_x, t)
        return self.head(h)

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        log_x1 = batch["log_x"]
        B, L, D = log_x1.shape
        device, dt = log_x1.device, log_x1.dtype
        log_x0 = _uniform_log_x0(B, L, D, device, dt)
        t = torch.rand(B, device=device, dtype=dt)
        log_xt = (1.0 - t[:, None, None]) * log_x0 + t[:, None, None] * log_x1
        u_tgt = log_x1 - log_x0
        v_pred = self._predict_velocity(log_xt, t)
        loss = self._velocity_loss_fn(v_pred, u_tgt)
        return {TRAINING_LOSS_KEY: loss, "velocity_loss": loss.detach()}

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        log_x1 = batch["log_x"]
        B, L, D = log_x1.shape
        device, dt = log_x1.device, log_x1.dtype
        log_x0 = _uniform_log_x0(B, L, D, device, dt)
        t = torch.rand(B, device=device, dtype=dt)
        log_xt = (1.0 - t[:, None, None]) * log_x0 + t[:, None, None] * log_x1
        u_tgt = log_x1 - log_x0
        v_pred = self._predict_velocity(log_xt, t)
        err = (v_pred - u_tgt).pow(2).mean()
        return {"mse_velocity": err, TRAINING_LOSS_KEY: err}

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        return self.eval_step(batch)


@register("flow_matching")
def build_flow_matching_auditor(cfg: Config) -> FlowMatchingAuditor:
    return FlowMatchingAuditor(cfg)
