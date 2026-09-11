from __future__ import annotations

import math

import torch


def hilbert_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hilbert distance between two points in the latent space.

    Args:
        x: First point in the latent space.
        y: Second point in the latent space.
    Returns:
        The Hilbert distance between the two points.
    """
    diff = x - y
    return diff.amax(dim=-1) - diff.amin(dim=-1)


def nielsen_soft_hilbert_distance(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 10.0
) -> torch.Tensor:
    """Differentiable approximation of the Hilbert simplex distance via the variation norm.

    Computes ‖x - y‖_var = max_i(x_i - y_i) - min_i(x_i - y_i) using the
    LSE_T trick from Nielsen & Sun (2023) to approximate max/min differentiably.

    When x and y are log-space velocity vectors, this is equivalent to the
    Hilbert distance ρ_HG between their implied simplex points, since:
        ρ_HG(p, q) = ‖log p - log q‖_var  (Nielsen & Sun, 2023, Eq. 5)

    As alpha -> infinity, converges to the true variation norm (Hilbert distance).
    Approximation error is bounded by (2/alpha) * log(d) (Nielsen & Sun, 2023, Eq. 9).

    Args:
        x: First log-space vector (..., K).
        y: Second log-space vector (..., K).
        alpha: Temperature parameter. Higher = closer to true Hilbert distance
               but harder gradients. Recommended range: [1, 10].

    Returns:
        Soft Hilbert distance (...,), i.e. one scalar per (x, y) pair.
    """
    diff = x - y
    # τ logsumexp(x/τ) → max(x) as τ→0+; same structure for soft min via -max(-x).
    soft_max = torch.logsumexp(alpha * diff, dim=-1) / alpha
    soft_min = -torch.logsumexp(-alpha * diff, dim=-1) / alpha
    return (soft_max - soft_min).clamp(min=1e-8)


def ilr(log_x: torch.Tensor) -> torch.Tensor:
    """Isometric log-ratio (ILR) transform of compositional data.

    Applies CLR (centering log ratios) then an orthogonal Helmert contrast,
    yielding coordinates in ``R^{K-1}`` that preserve the Aitchison geometry.

    Args:
        log_x: Log-scale compositions, shape ``(..., K)`` (e.g. log proportions
            that differ by an additive constant from true log probabilities).

    Returns:
        ILR coordinates, shape ``(..., K-1)``.
    """
    clr = log_x - log_x.mean(dim=-1, keepdim=True)
    K = log_x.shape[-1]
    psi = _helmert_matrix(K, log_x.device, log_x.dtype)
    return clr @ psi


def _helmert_matrix(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build the Helmert contrast basis for compositional (simplex) coordinates.

    Columns form an orthonormal basis of the centered hyperplane
    ``{z ∈ R^K : sum(z) = 0}``, used to map CLR vectors to ILR coordinates.

    Args:
        K: Simplex dimension (number of parts).
        device: Device for the returned tensor.
        dtype: Floating dtype for the returned tensor.

    Returns:
        Matrix ``Ψ`` of shape ``(K, K-1)`` with orthonormal columns.
    """
    psi = torch.zeros(K, K - 1, device=device, dtype=dtype)

    # Orthonormal Helmert contrasts (Egozcue 2003, Eq. 1):
    #   column i (0-indexed) has entries +scale in rows 0..i and -(i+1)*scale
    #   in row i+1, with scale = 1 / sqrt((i+1)(i+2)). This makes Ψ^T Ψ = I.
    for i in range(K - 1):
        scale = 1.0 / math.sqrt((i + 1) * (i + 2))
        psi[: i + 1, i] = scale
        psi[i + 1, i] = -(i + 1) * scale

    return psi


def ilr_inv(y: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse isometric log-ratio map back to centered log-ratio (CLR) space.

    Args:
        y: ILR coordinates, shape (..., K-1).
        K: Simplex dimension.

    Returns:
        CLR vectors, shape (..., K), where sum(out) == 0.
    """
    psi = _helmert_matrix(K, y.device, y.dtype)
    # Projecting from (K-1) back to K dimensions
    clr = torch.matmul(y, psi.t())

    # clr is already zero-mean because columns of psi sum to zero.
    # No further normalization is required for the Aitchison inverse.
    return clr


def volume_penalty(
    x_log_space: torch.Tensor,
    alpha: float = 10.0,
    diag_approx: bool = True,
) -> torch.Tensor:
    """Penalty from the volume (log sqrt determinant) of a Hessian of squared soft range.

    For each batch row ``z``, uses a fixed ``origin`` (uniform direction scaled to
    unit Euclidean norm), defines a scalar soft Hilbert-style distance ``d(z)``
    via log-sum-exp on ``z - origin`` with sharpness ``alpha``, and forms
    ``G`` as the Hessian of ``d(z) ** 2`` plus a small diagonal ridge. Returns
    ``0.5 * log |G|`` per row (clamped for stability).

    If ``diag_approx`` is true, ``log |G|`` is approximated by summing logs of
    the diagonal of ``G`` only (fast, no batched ``D x D`` solves). Otherwise
    ``G`` is built explicitly and ``torch.linalg.slogdet`` is used batch-wise.

    Args:
        x_log_space: Points in log space, shape ``(B, D)``.
        alpha: Sharpness of the soft min / max along the last dimension; larger
            values track the hard Hilbert range more closely.
        diag_approx: If true, use the diagonal-only determinant approximation;
            if false, use the full ``(D, D)`` Hessian per batch row.

    Returns:
        Tensor of shape ``(B,)``, one penalty per row of ``x_log_space``.
    """
    Bx, D = x_log_space.shape
    origin = torch.ones(D, device=x_log_space.device, dtype=x_log_space.dtype) / (
        D**0.5
    )
    diff = x_log_space - origin
    p = torch.softmax(alpha * diff, dim=-1)
    q = torch.softmax(-alpha * diff, dim=-1)
    d = (
        torch.logsumexp(alpha * diff, dim=-1) - torch.logsumexp(-alpha * diff, dim=-1)
    ) / alpha

    if diag_approx:
        diag_G = (
            2.0 * (p + q) ** 2 + 2.0 * d[:, None] * alpha * (p - q - p**2 + q**2) + 1e-4
        )
        log_det = torch.log(diag_G.clamp(min=1e-8)).sum(dim=-1)
    else:
        H_d = alpha * (
            torch.diag_embed(p - q)
            - torch.bmm(p.unsqueeze(2), p.unsqueeze(1))
            + torch.bmm(q.unsqueeze(2), q.unsqueeze(1))
        )
        g = (p + q).unsqueeze(2)
        G = 2.0 * torch.bmm(g, g.transpose(1, 2)) + 2.0 * d[:, None, None] * H_d
        G = G + torch.eye(D, device=x_log_space.device, dtype=x_log_space.dtype) * 1e-4
        sign, log_det = torch.linalg.slogdet(G)
        log_det = torch.where(sign > 0, log_det, torch.zeros_like(log_det))

    return 0.5 * log_det.clamp(min=-50.0, max=50.0)
