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

        return loss.mean()


class MSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class SoftmaxSoftHilbertLoss(nn.Module):
    """Project (pred, target) onto the K-simplex via softmax, then compute the
    Nielsen soft-Hilbert distance on the resulting probability vectors.

    Mathematically distinct from ``SoftHilbertLoss`` in CLR space: the
    log-softmax form is invariant under additive constants and equals the
    raw variation seminorm, but plugging *raw* softmax outputs (probability
    vectors with values in [0, 1]) into the variation seminorm gives a
    bounded, sharper geometry than the unconstrained CLR-space loss.

    Empirically the dynamic range of ``softmax(pred) − softmax(target)`` is
    much smaller (≤1 in absolute value) than ``pred − target`` in CLR
    coords (~10), so ``alpha`` here needs to be roughly an order of
    magnitude larger than for ``hilbert_soft`` for the LSE max-term to
    dominate.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.softmax(dim=-1)
        q = target.softmax(dim=-1)
        diff = p - q  # sums to 0 along K; |diff_i| ≤ 1.

        term1 = torch.logsumexp(self.alpha * diff, dim=-1)
        term2 = torch.logsumexp(self.alpha * (-diff), dim=-1)

        loss = (term1 + term2) / self.alpha

        # Constant offset so loss = 0 when p = q.
        K = pred.size(-1)
        loss = loss - (2.0 / self.alpha) * torch.log(torch.tensor(K))

        return loss.mean()


def sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Sparsemax projection onto the simplex (Martins & Astudillo 2016).

    For an input ``z`` of shape (..., K), returns probabilities of the same
    shape that sum to 1 along the last axis and may contain exact zeros.

    Pure-PyTorch, second-order differentiable (sort + cumsum + clamp).
    """
    K = z.shape[-1]
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    z_cumsum = z_sorted.cumsum(dim=-1)
    k = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    # Support condition: 1 + k * z_sorted_k > sum_{j<=k} z_sorted_j
    support = (1.0 + k * z_sorted) > z_cumsum  # (..., K) bool
    cardinality = support.long().sum(dim=-1, keepdim=True).clamp(min=1)
    # tau = (sum_{j in S} z_j - 1) / |S|
    z_cumsum_k = z_cumsum.gather(-1, cardinality - 1)
    tau = (z_cumsum_k - 1.0) / cardinality.to(z.dtype)
    p = (z - tau).clamp(min=0.0)
    return p


def sparsemax_loss(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Martins & Astudillo 2016 sparsemax loss (Eq. 10).

    ``z`` is (..., K) logits, ``target`` is (...) integer class indices.
    Returns the mean loss across the leading dimensions. Convex, has zero
    gradient on the support so the model can be exactly confident.

    Closed form (verified at the limit p = e_y):
      L(z, y) = -z_y + 0.5 * sum_{j in S(z)} z_j^2 - 0.5 * τ^2 * |S(z)| + 0.5

    where τ and S(z) are the sparsemax threshold and support of ``z``.
    """
    p = sparsemax(z)
    support_f = (p > 0).to(z.dtype)
    cardinality = support_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    # τ = (sum_{j in S} z_j - 1) / |S|
    tau = (support_f * z).sum(dim=-1, keepdim=True) / cardinality - 1.0 / cardinality

    z_target = z.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    sum_z2_on_support = (support_f * z * z).sum(dim=-1)
    tau_sq_card = (tau.squeeze(-1) ** 2) * cardinality.squeeze(-1)

    loss = -z_target + 0.5 * (sum_z2_on_support - tau_sq_card + 1.0)
    return loss.mean()


@_REGISTRY.register("hilbert")
def _build_hilbert(cfg: Config) -> HilbertLoss:
    return HilbertLoss()


@_REGISTRY.register("hilbert_soft")
def _build_soft_hilbert(cfg: Config) -> SoftHilbertLoss:
    return SoftHilbertLoss(alpha=cfg.loss.hilbert_alpha)


@_REGISTRY.register("hilbert_soft_softmax")
def _build_softmax_soft_hilbert(cfg: Config) -> SoftmaxSoftHilbertLoss:
    return SoftmaxSoftHilbertLoss(alpha=cfg.loss.hilbert_alpha)


@_REGISTRY.register("mse")
def _build_mse(cfg: Config) -> MSELoss:
    return MSELoss()


def build_loss(cfg: Config) -> nn.Module:
    return _REGISTRY.get(cfg.loss.mode)(cfg)


def anneal_alpha(model: nn.Module, cfg: Config, epoch: int) -> float | None:
    """Linearly anneal SoftHilbertLoss alpha from hilbert_alpha_start to hilbert_alpha.

    Returns the new alpha value, or None if the loss is not SoftHilbertLoss.
    """
    if cfg.loss.mode not in ("hilbert_soft", "hilbert_soft_softmax"):
        return None
    loss_fn = getattr(model, "loss_fn", None)
    if not isinstance(loss_fn, (SoftHilbertLoss, SoftmaxSoftHilbertLoss)):
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
