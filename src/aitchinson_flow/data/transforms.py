import torch


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
