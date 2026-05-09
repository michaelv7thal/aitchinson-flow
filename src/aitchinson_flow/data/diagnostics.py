"""Phase 0 instrumentation — describe a CLR data tensor and the FM target it induces.

Used by ``scripts/check_dirichlet_data.py`` to compare the deterministic
``token_ids_to_features`` output and the new
``token_ids_to_features_dirichlet`` output side-by-side, and to verify
``alpha_peak`` calibration during Phase 2.
"""

from __future__ import annotations

import torch

from aitchinson_flow.data.transforms import token_ids_to_features


def variation_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Aitchison variation seminorm: max - min along ``dim``.

    For CLR-coded points this is the standard ``hilbert_distance`` (see
    geometry.py); we keep a small in-module copy to avoid reaching into
    the model package from data utilities.
    """
    return x.amax(dim=dim) - x.amin(dim=dim)


def describe_clr(x: torch.Tensor, name: str = "x") -> dict[str, float]:
    """Per-coordinate / per-position descriptors of a CLR tensor.

    Inputs of shape (B, L, K) or (N, K). Returns a flat dict suitable for
    pretty-printing or for collecting into a JSON record.
    """
    if x.ndim < 2:
        raise ValueError(f"expected (..., K) tensor, got shape {tuple(x.shape)}")
    K = x.shape[-1]
    flat = x.reshape(-1, K)  # (N, K)
    var = variation_norm(flat)  # (N,)
    l2 = flat.norm(dim=-1)  # (N,)
    return {
        f"{name}.K": float(K),
        f"{name}.n_points": float(flat.shape[0]),
        f"{name}.coord_min": float(flat.min()),
        f"{name}.coord_max": float(flat.max()),
        f"{name}.coord_mean": float(flat.mean()),
        f"{name}.var_norm_mean": float(var.mean()),
        f"{name}.var_norm_std": float(var.std()),
        f"{name}.var_norm_min": float(var.min()),
        f"{name}.var_norm_max": float(var.max()),
        f"{name}.l2_mean": float(l2.mean()),
        f"{name}.l2_std": float(l2.std()),
        f"{name}.zero_mean_residual": float(flat.mean(dim=-1).abs().max()),
        f"{name}.has_nan": float(torch.isnan(flat).any().item()),
        f"{name}.has_inf": float(torch.isinf(flat).any().item()),
    }


def distance_to_token_basis(
    x: torch.Tensor,
    token_ids: torch.Tensor,
    K: int,
    *,
    label_smoothing: float = 1e-4,
) -> dict[str, float]:
    """Per-position CLR distance from x[..., k] to the deterministic basis vector
    of the *true* token id, and to the *closest* basis vector (argmin).

    Returns a small summary: how far the data sits from the on-token vertex
    on average, and what fraction of positions still nearest-neighbour back
    to the correct token.
    """
    flat_ids = token_ids.reshape(-1)  # (N,)
    flat_x = x.reshape(-1, K)         # (N, K)
    eye_ids = torch.arange(K, device=x.device)
    basis = token_ids_to_features(eye_ids, K, label_smoothing=label_smoothing)  # (K, K)

    diffs = flat_x.unsqueeze(1) - basis.unsqueeze(0)  # (N, K, K) — heavy at K=27 it's fine
    dists = diffs.pow(2).sum(dim=-1).sqrt()           # (N, K) Euclidean in CLR
    nearest = dists.argmin(dim=-1)
    on_token = dists.gather(-1, flat_ids.long().unsqueeze(-1)).squeeze(-1)
    return {
        "dist_on_token_mean": float(on_token.mean()),
        "dist_on_token_std": float(on_token.std()),
        "nn_recovery_acc": float((nearest == flat_ids).float().mean()),
    }


def fm_target_diagnostic(
    x1: torch.Tensor,
    *,
    source_sigma: float,
    n_gamma: int = 8,
) -> dict[str, float]:
    """Statistics of the FM target ``c(γ)·(x_0 − x_1)`` for a fixed batch.

    With deterministic-CLR data, ``x_1`` is one of K vertices and the
    target field is piecewise-constant per-token ⇒ near-zero variance
    of ``x_1`` magnitude across same-token positions. With Dirichlet
    data, ``x_1`` is sampled fresh ⇒ non-zero variance, which is the
    discontinuity-smoothing the Dirichlet hypothesis predicts.

    We report just the magnitudes of ``x_1`` and ``u_tgt = c(γ)·(x_0 − x_1)``
    averaged over a few γ values so the user can sanity-check that the
    target's L2 hasn't blown up. Linear decay c(γ)=1−γ is assumed.
    """
    x1 = x1.detach()
    device = x1.device

    # x0 ~ N(0, σ²) projected to V_d, mirroring _eqm_loss.
    x0 = source_sigma * torch.randn_like(x1)
    x0 = x0 - x0.mean(dim=-1, keepdim=True)

    x1_norm = x1.norm(dim=-1).reshape(-1)  # (N,)

    gammas = torch.linspace(0.1, 0.95, n_gamma, device=device)
    u_norm_means: list[float] = []
    u_norm_stds: list[float] = []
    for g in gammas:
        c = 1.0 - g
        u = c * (x0 - x1)
        u_n = u.norm(dim=-1).reshape(-1)
        u_norm_means.append(float(u_n.mean()))
        u_norm_stds.append(float(u_n.std()))
    return {
        "x1.l2_mean": float(x1_norm.mean()),
        "x1.l2_std": float(x1_norm.std()),
        "u_tgt.l2_mean_avg_over_gamma": float(sum(u_norm_means) / len(u_norm_means)),
        "u_tgt.l2_std_avg_over_gamma": float(sum(u_norm_stds) / len(u_norm_stds)),
    }


def per_token_x1_variance(
    x1: torch.Tensor,
    token_ids: torch.Tensor,
    K: int,
) -> dict[str, float]:
    """Across-batch variance of ``x_1`` for *the same* token id.

    For deterministic data this is identically zero (modulo floating point);
    for Dirichlet data it should be a non-trivial positive number — that's
    the discontinuity-smoothing we want.
    """
    flat_x = x1.reshape(-1, K)
    flat_ids = token_ids.reshape(-1).long()

    # Per-token mean across positions sharing that token id.
    sums = torch.zeros(K, K, device=x1.device).index_add(0, flat_ids, flat_x)
    counts = torch.zeros(K, device=x1.device).index_add(
        0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32)
    )
    means = sums / counts.clamp(min=1).unsqueeze(-1)
    diffs = flat_x - means[flat_ids]              # (N, K)
    sq = diffs.pow(2).sum(dim=-1)                 # (N,) per-position squared dev
    return {
        "x1.per_token_l2_var_mean": float(sq.mean()),
        "x1.per_token_l2_var_max": float(sq.max()),
    }


__all__ = [
    "describe_clr",
    "distance_to_token_basis",
    "fm_target_diagnostic",
    "per_token_x1_variance",
    "variation_norm",
]
