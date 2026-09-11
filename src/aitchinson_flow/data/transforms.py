import torch
import torch.nn.functional as F


def token_ids_to_features(
    ids: torch.Tensor,
    K: int,
    *,
    label_smoothing: float = 1e-4,
) -> torch.Tensor:
    """Token IDs → CLR features.

    Pipeline: one-hot (L×K) → label-smooth → log → CLR (subtract log-geometric-mean).
    Accepts (L,) or (N, L); returns (..., K).
    """
    if ids.ndim not in (1, 2):
        raise ValueError(f"ids must be 1D or 2D, got shape {tuple(ids.shape)}")
    if not 0.0 < label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in (0, 1), got {label_smoothing}")
    ids = ids.clamp(0, K - 1)
    oh = torch.zeros(*ids.shape, K, dtype=torch.float32, device=ids.device)
    oh.scatter_(-1, ids.unsqueeze(-1), 1.0)
    x = (1.0 - label_smoothing) * oh + label_smoothing / K
    log_x = x.log()
    return log_x - log_x.mean(dim=-1, keepdim=True)


def token_ids_to_features_dirichlet(
    ids: torch.Tensor,
    K: int,
    *,
    alpha_peak: float,
    alpha_base: float,
) -> torch.Tensor:
    """Token IDs → CLR features via Dirichlet sampling around each vertex.

    For token i, samples ``p ~ Dirichlet(α_i)`` with concentration vector
    ``α_i = α_base · 1 + α_peak · e_i`` (peaked at the target class but
    non-zero everywhere on the simplex), then maps p → CLR. A fresh draw
    happens on every call — caller must NOT cache the result. Accepts (L,)
    or (N, L); returns (..., K).

    The target token's concentration is ``α_base + α_peak``; the off-target
    classes get ``α_base``. Recovering the token by argmax requires
    ``α_peak`` large enough that the on-target draw dominates K-1 i.i.d.
    Gamma(α_base) draws — see scripts/check_dirichlet_data.py.
    """
    if ids.ndim not in (1, 2):
        raise ValueError(f"ids must be 1D or 2D, got shape {tuple(ids.shape)}")
    if alpha_peak <= 0.0 or alpha_base <= 0.0:
        raise ValueError(
            f"alpha_peak/alpha_base must be > 0, got peak={alpha_peak}, base={alpha_base}"
        )
    ids = ids.clamp(0, K - 1)
    one_hot = F.one_hot(ids.long(), num_classes=K).to(torch.float32)
    alpha = alpha_base + alpha_peak * one_hot
    # Reparameterised Dirichlet via independent Gammas. _standard_gamma is
    # faster than torch.distributions.Dirichlet and we don't need autograd
    # through the concentrations here (data input only). Seed globally via
    # torch.manual_seed for reproducibility.
    gamma = torch._standard_gamma(alpha)
    p = gamma / gamma.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    log_p = p.clamp(min=1e-10).log()
    return log_p - log_p.mean(dim=-1, keepdim=True)
