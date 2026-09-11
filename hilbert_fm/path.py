"""Log-linear (Aitchison/Hilbert) geodesic on the simplex and soft Hilbert loss.

All operations work in log-space. Probabilities are never normalised by hand; we
always go through ``log_softmax`` to stay numerically stable near the vertices.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def log_p1_from_ids(ids: torch.Tensor, K: int, eps: float = 0.01) -> torch.Tensor:
    """Build label-smoothed log-target ``log p_1`` from token ids.

    The correct coordinate gets mass ``1 - eps``; the remaining ``eps`` is
    spread uniformly over the other ``K - 1`` coordinates.

    Args:
        ids: ``(B, L)`` long tensor of token ids in ``[0, K)``.
        K:   vocabulary size.
        eps: total smoothing mass redistributed off the correct vertex.

    Returns:
        ``log p_1`` of shape ``(B, L, K)``.
    """
    B, L = ids.shape
    p1 = torch.full((B, L, K), eps / (K - 1), device=ids.device, dtype=torch.float32)
    p1.scatter_(-1, ids.unsqueeze(-1), 1.0 - eps)
    return p1.log()


def log_pt(log_p0: torch.Tensor, log_p1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Log-linear interpolant ``log_softmax((1 - t) log p0 + t log p1)``.

    Args:
        log_p0: ``(B, L, K)`` source log-probs (uniform in this experiment).
        log_p1: ``(B, L, K)`` target log-probs.
        t:      ``(B,)`` time in ``[0, 1]`` (should be capped strictly < 1 in train).

    Returns:
        ``log p_t`` of shape ``(B, L, K)``.
    """
    t_b = t.view(-1, 1, 1)
    return F.log_softmax((1.0 - t_b) * log_p0 + t_b * log_p1, dim=-1)


def soft_hilbert(
    log_p_hat: torch.Tensor, log_p_target: torch.Tensor, tau: float = 0.3
) -> torch.Tensor:
    """LogSumExp soft relaxation of the Hilbert projective metric.

    With ``r = log p_hat - log p_target``, the true metric is
    ``max_i r_i - min_i r_i``. The soft form

        ``tau * lse(r / tau) + tau * lse(-r / tau)``

    converges to the true metric as ``tau -> 0`` and is everywhere smooth.

    Returns the mean over batch and positions.
    """
    r = log_p_hat - log_p_target
    pos = tau * torch.logsumexp(r / tau, dim=-1)
    neg = tau * torch.logsumexp(-r / tau, dim=-1)
    return (pos + neg).mean()


def advance(
    log_pt_now: torch.Tensor,
    log_p1_hat: torch.Tensor,
    t_now: float,
    t_next: float,
) -> torch.Tensor:
    """Walk one step along the log-linear path with re-prediction.

    Closed-form increment along the geodesic from ``p_{t_now}`` to ``p_1_hat``::

        log p_{t_next} = log_softmax(
            ((1 - t_next) / (1 - t_now)) * log p_{t_now}
          + ((t_next - t_now) / (1 - t_now)) * log p_1_hat
        )
    """
    a = (1.0 - t_next) / (1.0 - t_now)
    b = (t_next - t_now) / (1.0 - t_now)
    return F.log_softmax(a * log_pt_now + b * log_p1_hat, dim=-1)


def hilbert_distance(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """True (non-soft) Hilbert distance ``max_i r_i - min_i r_i`` per position.

    Returns shape ``log_p.shape[:-1]``.
    """
    r = log_p - log_q
    return r.max(dim=-1).values - r.min(dim=-1).values


def uniform_log_p0(B: int, L: int, K: int, device: str | torch.device) -> torch.Tensor:
    """Uniform source ``log(1/K)`` of shape ``(B, L, K)``.

    Note: this source is *deterministic* — every sample starts from the same
    point, so the marginal flow has no per-sample diversity. Use
    :func:`random_log_p0` for sampling that produces distinct sequences.
    """
    return torch.full((B, L, K), -math.log(K), device=device, dtype=torch.float32)


def random_log_p0(
    B: int, L: int, K: int, eps: float, device: str | torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-``(b, l)`` random label-smoothed one-hot — a stochastic interior source.

    Each position draws an independent token from ``Uniform({0..K-1})`` and is
    converted to a label-smoothed log-distribution with mass ``1 - eps`` on the
    drawn coordinate and ``eps / (K - 1)`` elsewhere. This replaces the literal
    uniform source so sampling has per-sample diversity, while keeping the
    starting point in the strict interior of the simplex.
    """
    if generator is not None:
        ids = torch.randint(0, K, (B, L), device=device, generator=generator)
    else:
        ids = torch.randint(0, K, (B, L), device=device)
    return log_p1_from_ids(ids, K, eps)


def make_log_p0(
    kind: str, B: int, L: int, K: int, eps: float, device: str | torch.device,
) -> torch.Tensor:
    """Dispatcher: ``"uniform"`` -> :func:`uniform_log_p0`, ``"random_token"`` -> :func:`random_log_p0`."""
    if kind == "uniform":
        return uniform_log_p0(B, L, K, device)
    if kind == "random_token":
        return random_log_p0(B, L, K, eps, device)
    raise ValueError(f"unknown source kind: {kind!r}")
