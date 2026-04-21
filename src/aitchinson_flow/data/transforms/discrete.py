"""Map HF rows with integer token ids → continuous simplex coordinates.

Two pathways move discrete token ids into the open simplex interior:

1. **Epsilon path** (``label_smoothing == 0``): one-hot + small additive
   ``eps`` then log. Backward-compatible default; rows do not sum to one
   exactly but stay strictly positive so ``log`` is well defined.
2. **Label-smoothing path** (``label_smoothing > 0``): one-hot is mixed with
   the uniform distribution on K via ``(1 - α) · oh + α / K``. Rows sum to
   one exactly and lie strictly inside the simplex; this is the
   geometrically motivated transform described in the two-stage auditor
   plan.

The chosen log-space row is then projected with either the isometric
log-ratio (``ilr``, output dim ``K-1``) or the centered log-ratio (``clr``,
output dim ``K``) depending on ``cfg.hf_dataset.transform_mode``. The CLR
mode is used for the ILR-off ablation.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from aitchinson_flow.config import Config
from aitchinson_flow.geometry import ilr


def _smoothed_log_row(
    ids: torch.Tensor,
    K: int,
    *,
    eps: float,
    label_smoothing: float,
) -> torch.Tensor:
    """One-hot ids → log-probabilities row with the configured smoothing path.

    Args:
        ids: 1D long tensor of token ids, shape ``(L,)``.
        K: Vocabulary size (last dim of the one-hot row).
        eps: Additive smoothing for the legacy epsilon path
            (used when ``label_smoothing == 0``).
        label_smoothing: Mix coefficient α with uniform distribution on K.
            When ``α > 0`` the row is ``(1 - α) · one_hot + α / K`` and
            ``eps`` is ignored.

    Returns:
        Log-probability tensor of shape ``(L, K)`` (i.e. before any
        ILR/CLR projection).
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            f"label_smoothing must be in [0, 1), got {label_smoothing}"
        )
    L = ids.shape[0]
    oh = torch.zeros(L, K, dtype=torch.float32, device=ids.device)
    oh.scatter_(dim=-1, index=ids.unsqueeze(-1), value=1.0)
    if label_smoothing > 0.0:
        x = (1.0 - label_smoothing) * oh + label_smoothing / K
    else:
        x = oh + eps
    return x.log()


def _project(log_x: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Apply the configured simplex projection to a log-prob row.

    Args:
        log_x: Log-probabilities of shape ``(..., K)``.
        mode: ``"ilr"`` (return ``(..., K-1)`` Euclidean coords) or
            ``"clr"`` (return ``(..., K)`` centered log-ratios).

    Raises:
        ValueError: if ``mode`` is not a recognized value.
    """
    m = mode.lower()
    if m == "ilr":
        return ilr(log_x)
    if m == "clr":
        return log_x - log_x.mean(dim=-1, keepdim=True)
    raise ValueError(f"unknown transform_mode={mode!r}; expected 'ilr' or 'clr'")


def token_ids_to_ilr_x(
    ids: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """One-hot from ids → smoothed log-row → ILR coords ``(L, K-1)``.

    Backward-compatible signature: ``label_smoothing=0`` reproduces the
    previous ``one_hot + eps + log + ilr`` pipeline exactly.
    """
    if ids.ndim != 1:
        raise ValueError(f"token ids must be 1D, got shape {tuple(ids.shape)}")
    ids = ids.clamp(min=0, max=K - 1)
    log_x = _smoothed_log_row(ids, K, eps=eps, label_smoothing=label_smoothing)
    return ilr(log_x)


def token_ids_to_clr_x(
    ids: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """One-hot from ids → smoothed log-row → CLR coords ``(L, K)``.

    The output is the centered log-ratio (each row sums to zero) — used for
    the ILR-off ablation. Distances on this representation are constrained
    (sum-to-zero) but otherwise live in ``R^K``.
    """
    if ids.ndim != 1:
        raise ValueError(f"token ids must be 1D, got shape {tuple(ids.shape)}")
    ids = ids.clamp(min=0, max=K - 1)
    log_x = _smoothed_log_row(ids, K, eps=eps, label_smoothing=label_smoothing)
    return _project(log_x, mode="clr")


def token_ids_to_features(
    ids: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
    transform_mode: str = "ilr",
) -> torch.Tensor:
    """Mode-dispatching helper used by data modules.

    Honors both ``label_smoothing`` and ``transform_mode`` so callers can
    drive the entire discrete→continuous pipeline from config alone.
    """
    if ids.ndim != 1:
        raise ValueError(f"token ids must be 1D, got shape {tuple(ids.shape)}")
    ids = ids.clamp(min=0, max=K - 1)
    log_x = _smoothed_log_row(ids, K, eps=eps, label_smoothing=label_smoothing)
    return _project(log_x, mode=transform_mode)


def token_probs_to_features(
    probs: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    transform_mode: str = "ilr",
) -> torch.Tensor:
    """Simplex probabilities -> ILR/CLR features.

    Accepts ``(L, V)`` or ``(B, L, V)`` and projects the first ``K`` channels.
    """
    if probs.ndim not in (2, 3):
        raise ValueError(f"probs must be 2D or 3D, got shape {tuple(probs.shape)}")
    if probs.shape[-1] < K:
        raise ValueError(f"probs last dim must be >= K ({K}), got {probs.shape[-1]}")
    p = probs[..., :K].to(dtype=torch.float32)
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    log_p = p.log()
    return _project(log_p, mode=transform_mode)


def token_logits_to_features(
    logits: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    transform_mode: str = "ilr",
) -> torch.Tensor:
    """LM logits -> simplex probabilities -> ILR/CLR features."""
    if logits.ndim not in (2, 3):
        raise ValueError(f"logits must be 2D or 3D, got shape {tuple(logits.shape)}")
    if logits.shape[-1] < K:
        raise ValueError(f"logits last dim must be >= K ({K}), got {logits.shape[-1]}")
    probs = torch.softmax(logits[..., :K].to(dtype=torch.float32), dim=-1)
    return token_probs_to_features(probs, K=K, eps=eps, transform_mode=transform_mode)


def sorted_topk_logits_to_features(
    logits: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    transform_mode: str = "ilr",
) -> torch.Tensor:
    """LM logits -> per-position sorted top-K probabilities -> ILR/CLR features.

    At each position the top-``K`` softmax probabilities are gathered (descending),
    renormalized so they sum to one, and projected through the ILR/CLR map. Unlike
    :func:`token_logits_to_features`, which slices the first ``K`` tokenizer
    indices, this helper treats the ``K`` outputs as rank slots (slot 0 = most
    likely token at that position, slot 1 = second, …). This is the Path B
    feature used for LLM-driven auditors.

    Accepts ``(L, V)`` or ``(B, L, V)`` logits. Output shape matches the chosen
    ``transform_mode`` (``K-1`` for ILR, ``K`` for CLR).
    """
    if logits.ndim not in (2, 3):
        raise ValueError(f"logits must be 2D or 3D, got shape {tuple(logits.shape)}")
    if logits.shape[-1] < K:
        raise ValueError(f"logits last dim must be >= K ({K}), got {logits.shape[-1]}")
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    top_probs = probs.topk(K, dim=-1).values
    top_probs = top_probs.clamp_min(eps)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    log_p = top_probs.log()
    return _project(log_p, mode=transform_mode)


def token_ids_to_log_x(
    ids: torch.Tensor,
    K: int,
    *,
    eps: float = 1e-8,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Backward-compatible alias for ILR features (kept to avoid breaking imports)."""
    return token_ids_to_ilr_x(ids, K, eps=eps, label_smoothing=label_smoothing)


def _to_1d_long(value: Any) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.long)
    return t.reshape(-1)


def make_discrete_row_transform(cfg: Config) -> Callable[[dict[str, Any]], dict[str, torch.Tensor]]:
    """Build transform: HF row dict → ``{"log_x": (L, F)}`` using ``cfg.dataset`` + ``cfg.hf_dataset``.

    ``F`` is ``K-1`` for ILR mode and ``K`` for CLR mode (see
    ``aitchinson_flow.data.feature_dim.feature_dim``).
    """
    L = cfg.dataset.L
    K = cfg.dataset.K
    key = cfg.hf_dataset.row_input_key
    pad_id = cfg.hf_dataset.pad_token_id
    eps = cfg.hf_dataset.log_simplex_eps
    label_smoothing = cfg.hf_dataset.label_smoothing
    transform_mode = cfg.hf_dataset.transform_mode

    def transform(row: dict[str, Any]) -> dict[str, torch.Tensor]:
        if key not in row:
            raise KeyError(f"Row missing {key!r}; keys: {list(row.keys())}")

        ids = _to_1d_long(row[key])

        if ids.numel() > L:
            ids = ids[:L]

        elif ids.numel() < L:
            pad = torch.full((L - ids.numel(),), pad_id, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=0)
        log_x = token_ids_to_features(
            ids,
            K,
            eps=eps,
            label_smoothing=label_smoothing,
            transform_mode=transform_mode,
        )
        return {"log_x": log_x}

    return transform
