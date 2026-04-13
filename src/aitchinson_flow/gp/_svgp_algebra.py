"""Sparse variational GP algebra (Titsias-style, non-whitened).

Predictive mean and variance at test points given kernel blocks and a Gaussian
variational distribution over inducing variables, plus the KL between that
approximate posterior and the inducing prior. Matches the parameterisation used
in ``gp.py`` (variational covariance via a lower Cholesky factor).
"""

from __future__ import annotations

import torch

from ._svgp_kernels import cholesky_adaptive


def svgp_predictive(
    K_ZZ: torch.Tensor,
    K_xZ: torch.Tensor,
    K_xx_diag: torch.Tensor,
    mean_x: torch.Tensor,
    mean_Z: torch.Tensor,
    var_mean: torch.Tensor,
    var_L: torch.Tensor,
    base_jitter: float,
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictive mean and variance of the latent function at test points.

    Uses a Cholesky factor of the inducing Gram matrix and triangular solves
    instead of explicit matrix inverses. Predictive variances are clamped
    slightly above zero for numerical stability.

    Args:
        K_ZZ: Kernel matrix at inducing inputs, shape ``(M, M)``.
        K_xZ: Cross-covariance between each test point and inducing inputs,
            shape ``(B, M)``.
        K_xx_diag: Prior kernel variance at each test point, shape ``(B,)``.
        mean_x: Prior mean at test points, shape ``(B,)``.
        mean_Z: Prior mean at inducing points, shape ``(M,)``.
        var_mean: Variational mean over inducing variables, shape ``(M,)``.
        var_L: Lower Cholesky factor of the variational covariance, shape
            ``(M, M)``.
        base_jitter: Initial jitter passed to ``cholesky_adaptive`` when
            factoring ``K_ZZ``.
        name: Label forwarded to ``cholesky_adaptive`` for diagnostics.

    Returns:
        ``(pred_mean, pred_var)`` each of shape ``(B,)``: approximate predictive
        mean and variance of the latent GP under the variational distribution.
    """
    L_ZZ = cholesky_adaptive(K_ZZ, base_jitter, name=name)
    diff = (var_mean - mean_Z).unsqueeze(-1)
    v = torch.linalg.solve_triangular(L_ZZ, diff, upper=False)
    alpha = torch.linalg.solve_triangular(L_ZZ.T, v, upper=True).squeeze(-1)
    pred_mean = mean_x + K_xZ @ alpha
    A = torch.linalg.solve_triangular(L_ZZ, K_xZ.T, upper=False)
    C = torch.linalg.solve_triangular(L_ZZ, var_L, upper=False)
    pred_var = (K_xx_diag - (A * A).sum(0) + (C.T @ A).pow(2).sum(0)).clamp_min(1e-6)
    return pred_mean, pred_var


def svgp_kl(
    M: int,
    K_ZZ: torch.Tensor,
    mean_Z: torch.Tensor,
    var_mean: torch.Tensor,
    var_L: torch.Tensor,
    base_jitter: float,
    *,
    name: str,
) -> torch.Tensor:
    """KL divergence from the variational Gaussian to the prior on inducing variables.

    Closed form for two multivariate normals of the same dimension; implemented
    with Cholesky factors and triangular solves on ``K_ZZ``.

    Args:
        M: Number of inducing variables (length of ``u``).
        K_ZZ: Prior covariance at inducing points, shape ``(M, M)``.
        mean_Z: Prior mean at inducing points, shape ``(M,)``.
        var_mean: Variational mean, shape ``(M,)``.
        var_L: Lower Cholesky factor of the variational covariance, shape
            ``(M, M)``.
        base_jitter: Initial jitter for factoring ``K_ZZ``.
        name: Label forwarded to ``cholesky_adaptive``.

    Returns:
        Scalar tensor (0-dimensional): ``KL(q(u) || p(u))`` in nats, same
        device and dtype as ``K_ZZ``.
    """
    L_ZZ = cholesky_adaptive(K_ZZ, base_jitter, name=name)
    C = torch.linalg.solve_triangular(L_ZZ, var_L, upper=False)
    v = torch.linalg.solve_triangular(
        L_ZZ, (var_mean - mean_Z).unsqueeze(-1), upper=False
    ).squeeze(-1)
    log_det_K = 2.0 * L_ZZ.diagonal().log().sum()
    log_det_S = 2.0 * var_L.diagonal().log().sum()
    return 0.5 * (
        (C * C).sum()
        + (v * v).sum()
        - M
        + log_det_K
        - log_det_S
    )
