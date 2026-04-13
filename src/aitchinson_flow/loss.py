from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F

from .geometry import (
    hilbert_distance,
    ilr,
    nielsen_soft_hilbert_distance,
)

VelocityLossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _center_last(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def soft_hilbert_flow_loss(
    v_pred: torch.Tensor, v_tgt: torch.Tensor, *, alpha: float = 1.0
) -> torch.Tensor:
    """Current `flow_loss` style: centered + Nielsen soft Hilbert, mean over batch."""
    vp, vt = _center_last(v_pred), _center_last(v_tgt)
    return nielsen_soft_hilbert_distance(vp, vt, alpha=alpha).mean()


def hard_hilbert_flow_loss(v_pred: torch.Tensor, v_tgt: torch.Tensor) -> torch.Tensor:
    vp, vt = _center_last(v_pred), _center_last(v_tgt)
    return hilbert_distance(vp, vt).mean()


def mse_loss(v_pred: torch.Tensor, v_tgt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_pred, v_tgt)


def clr_residual_mse(v_pred: torch.Tensor, v_tgt: torch.Tensor) -> torch.Tensor:
    """Remove the irrelevant component along 1 in log-space (compositional flavor)."""
    clr_p = v_pred - v_pred.mean(dim=-1, keepdim=True)
    clr_t = v_tgt - v_tgt.mean(dim=-1, keepdim=True)
    return F.mse_loss(clr_p, clr_t)


def ilr_residual_mse(v_pred: torch.Tensor, v_tgt: torch.Tensor) -> torch.Tensor:
    """Euclidean metric on ILR coordinates of CLR residuals (K-1); 'Aitchison-flavored' vector loss."""
    K = v_pred.shape[-1]
    clr_p = v_pred - v_pred.mean(dim=-1, keepdim=True)
    clr_t = v_tgt - v_tgt.mean(dim=-1, keepdim=True)
    yp = ilr(clr_p)  # (..., K-1)
    yt = ilr(clr_t)
    return F.mse_loss(yp, yt)


def build_velocity_loss(
    name: str,
    *,
    soft_hilbert_alpha: float = 1.0,
) -> VelocityLossFn:
    """Factory used from Config so experiments stay declarative."""
    if name == "mse":
        return mse_loss
    if name == "soft_hilbert":
        return lambda vp, vt: soft_hilbert_flow_loss(vp, vt, alpha=soft_hilbert_alpha)
    if name == "hard_hilbert":
        return hard_hilbert_flow_loss
    if name == "clr_mse":
        return clr_residual_mse
    if name == "ilr_mse":
        return ilr_residual_mse
    raise ValueError(f"Unknown velocity loss {name!r}")
