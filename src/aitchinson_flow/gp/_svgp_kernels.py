"""Kernels and helpers for the sparse variational GP (``gp.py``).

Isotropic Matérn 5/2 covariances, adaptive Cholesky with diagonal jitter, and
a lower-triangular parameterisation of the variational covariance with
strictly positive diagonal entries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def matern52(
    x1: torch.Tensor,
    x2: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_outputscale: torch.Tensor,
) -> torch.Tensor:
    """Isotropic Matérn covariance (smoothness nu = 5/2) for all input pairs.

    Inputs are divided by the length scale before distances are taken. Squared
    distances use a matmul expansion instead of ``torch.cdist`` so
    second-order autograd (e.g. ``create_graph=True``) remains available.

    Args:
        x1: Left batch of points, shape ``(n, d)``.
        x2: Right batch of points, shape ``(m, d)``.
        log_lengthscale: Scalar tensor, natural log of the length scale.
        log_outputscale: Scalar tensor, natural log of the signal variance.

    Returns:
        Covariance matrix of shape ``(n, m)``: entry ``(i, j)`` is the kernel
        between row ``i`` of ``x1`` and row ``j`` of ``x2``.
    """
    ls = log_lengthscale.exp()
    os = log_outputscale.exp()
    x1_ = x1 / ls
    x2_ = x2 / ls
    sq = (
        (x1_ * x1_).sum(-1, keepdim=True)
        + (x2_ * x2_).sum(-1, keepdim=True).T
        - 2.0 * x1_ @ x2_.T
    ).clamp_min(0.0)
    dist = (sq + 1e-8).sqrt()
    r5 = 5.0**0.5
    return os * (1.0 + r5 * dist + (5.0 / 3.0) * sq) * torch.exp(-r5 * dist)


def cholesky_adaptive(K: torch.Tensor, base_jitter: float, *, name: str) -> torch.Tensor:
    """Lower Cholesky factor after adding enough diagonal jitter to succeed.

    Tries ``K + eps * I`` for ``eps`` equal to ``base_jitter`` times
    ``1, 10, 100, 1000`` until ``torch.linalg.cholesky`` succeeds.

    Args:
        K: Symmetric positive semidefinite matrix (typically the inducing Gram
            matrix), square.
        base_jitter: Smallest jitter tried first; scaled up on failure.
        name: Included in the error message if every scale fails.

    Returns:
        Lower-triangular ``L`` with ``L @ L.T`` equal to ``K`` plus the jitter
        that worked, same device and dtype as ``K``.

    Raises:
        RuntimeError: If no jitter in the schedule yields a successful factorization.
    """
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    for scale in (1.0, 10.0, 100.0, 1000.0):
        try:
            return torch.linalg.cholesky(K + (base_jitter * scale) * eye)
        except RuntimeError:
            continue

    raise RuntimeError(f"{name}: K + jitter still not positive definite")


def var_L_from_raw(var_L_raw: torch.Tensor, diag_floor: float) -> torch.Tensor:
    """Lower Cholesky factor of the variational covariance from raw parameters.

    Off-diagonals on and below the diagonal come from the lower triangle of
    ``var_L_raw``; each diagonal entry is softplus of the raw diagonal plus
    ``diag_floor``, so the factor stays lower triangular with positive diagonal
    and the implied covariance stays positive definite.

    Args:
        var_L_raw: Unconstrained parameters, shape ``(M, M)``; only the lower
            triangle contributes (diagonal feeds softplus).
        diag_floor: Small positive offset added after softplus on the diagonal.

    Returns:
        Lower-triangular ``L_var`` of shape ``(M, M)`` with
        ``L_var @ L_var.T`` the variational covariance.
    """
    L = var_L_raw.tril(-1)
    diag_pos = F.softplus(var_L_raw.diagonal()) + diag_floor
    return L + torch.diag(diag_pos)
