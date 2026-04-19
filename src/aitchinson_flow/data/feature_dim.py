"""Single source of truth for the per-token feature dimension.

The discrete→simplex transform produces tensors whose last-dim shape depends on
``cfg.hf_dataset.transform_mode`` (``"ilr"`` → ``K-1`` Euclidean coordinates,
``"clr"`` → ``K`` constrained coordinates). All model components (transformer
input projection, velocity head, generation seeds) read this helper instead of
hard-coding ``cfg.dataset.K - 1`` so the ILR-on/off ablation can be toggled
purely from config.
"""

from __future__ import annotations

from aitchinson_flow.config import Config

VALID_TRANSFORM_MODES: tuple[str, ...] = ("ilr", "clr")


def feature_dim(cfg: Config) -> int:
    """Return the per-token feature dimension implied by ``cfg.hf_dataset``.

    Args:
        cfg: Global config carrying ``dataset.K`` and ``hf_dataset.transform_mode``.

    Returns:
        ``max(1, K - 1)`` for ILR, ``max(1, K)`` for CLR.

    Raises:
        ValueError: if ``transform_mode`` is not a recognized value.
    """
    mode = cfg.hf_dataset.transform_mode.lower()
    if mode == "ilr":
        return max(1, cfg.dataset.K - 1)
    if mode == "clr":
        return max(1, cfg.dataset.K)
    raise ValueError(
        f"hf_dataset.transform_mode must be one of {VALID_TRANSFORM_MODES}, got {mode!r}"
    )
