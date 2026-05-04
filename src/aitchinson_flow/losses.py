from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.geometry import hilbert_distance, nielsen_soft_hilbert_distance
from aitchinson_flow.utils import Registry


LossFn = nn.Module

_REGISTRY: Registry = Registry("loss")


class HilbertLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return hilbert_distance(pred, target).mean()


class SoftHilbertLoss(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target

        # LSE(T * (x - x_hat))
        term1 = torch.logsumexp(self.alpha * diff, dim=-1)

        # LSE(T * (x_hat - x))
        term2 = torch.logsumexp(self.alpha * (-diff), dim=-1)

        # Symmetrized loss: 1/T * (LSE1 + LSE2)
        loss = (term1 + term2) / self.alpha

        # Note: The paper mentions p=q results in (2/T) * log(d).
        # You may optionally subtract this constant if you want loss=0 at optimum.
        # vocab_size = pred_log.size(-1)
        # loss = loss - (2.0 / self.T) * torch.log(torch.tensor(vocab_size))
        vocab_size = pred.size(-1)
        loss = loss - (2.0 / self.alpha) * torch.log(torch.tensor(vocab_size))

        return loss.mean()


class MSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


@_REGISTRY.register("hilbert")
def _build_hilbert(cfg: Config) -> HilbertLoss:
    return HilbertLoss()


@_REGISTRY.register("hilbert_soft")
def _build_soft_hilbert(cfg: Config) -> SoftHilbertLoss:
    return SoftHilbertLoss(alpha=cfg.loss.hilbert_alpha)


@_REGISTRY.register("mse")
def _build_mse(cfg: Config) -> MSELoss:
    return MSELoss()


def build_loss(cfg: Config) -> nn.Module:
    return _REGISTRY.get(cfg.loss.mode)(cfg)


def anneal_alpha(model: nn.Module, cfg: Config, epoch: int) -> float | None:
    """Linearly anneal SoftHilbertLoss alpha from hilbert_alpha_start to hilbert_alpha.

    Returns the new alpha value, or None if the loss is not SoftHilbertLoss.
    """
    if cfg.loss.mode != "hilbert_soft":
        return None
    loss_fn = getattr(model, "loss_fn", None)
    if not isinstance(loss_fn, SoftHilbertLoss):
        return None
    n = cfg.loss.hilbert_alpha_anneal_epochs
    if n is None:
        n = cfg.training.epochs
    t = min(epoch / max(n, 1), 1.0)
    alpha = (
        cfg.loss.hilbert_alpha_start
        + (cfg.loss.hilbert_alpha - cfg.loss.hilbert_alpha_start) * t
    )
    loss_fn.alpha = alpha
    return alpha
