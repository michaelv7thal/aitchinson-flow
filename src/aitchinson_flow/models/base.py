"""Protocols and types for generative models (Pattern A: training_step / eval_step)."""

from __future__ import annotations

import logging
from typing import Any, NotRequired, Protocol, TypedDict

import torch
import torch.nn as nn

TRAINING_LOSS_KEY = "loss"
"""Scalar tensor key required in `training_step` return dicts."""

_kl_normalizer_warned: set[int] = set()


def kl_normalizer(model: nn.Module, batch_size: int) -> float:
    """Return the divisor to apply to ``gp.kl_divergence()`` in the ELBO.

    Prefers ``model._kl_normalizer`` (set by the training runner to the true
    training-set size ``N``). When unavailable — e.g. in unit tests that
    bypass the runner — falls back to the batch size ``B`` and warns once
    per model instance so the degraded mode is visible without spamming
    logs.
    """
    n = getattr(model, "_kl_normalizer", None)
    if n is not None and n > 0:
        return float(n)
    key = type(model).__name__
    if key not in _kl_normalizer_warned:
        _kl_normalizer_warned.add(key)
        logging.getLogger(__name__).warning(
            "%s has no _kl_normalizer attribute set; falling back to batch size "
            "(B=%d) for KL normalization. Set model._kl_normalizer = len(train_dataset) "
            "(the training runner does this automatically) for the correct ELBO scale.",
            type(model).__name__,
            batch_size,
        )
    return float(batch_size)


class TrainingStepOutput(TypedDict):
    """Typical batch when using a dict collate; models may still take `Any` for custom types."""

    log_x: torch.Tensor  # (B, L, K)
    t: NotRequired[torch.Tensor]  # (B,) - optional time flow / diffusion
    mask: NotRequired[torch.Tensor]  # (B, L) - optional padding / validity


LossDict = dict[str, torch.Tensor]
"""Metric name → tensor. Must include `TRAINING_LOSS_KEY` for the optimization loss."""


class GenerativeTrainingModel(Protocol):
    """Structural contract for models used by the training loop.

    Usually implemented by an `nn.Module` that composes `TransformerBackbone` and heads.
    Prefer `training_step` / `eval_step` over a single overloaded `forward`.
    """

    def training_step(self, batch: Any, step: int) -> LossDict:
        """Return a dict containing a scalar `TRAINING_LOSS_KEY` for `.backward()`.

        `batch` is often `TrainingBatchDict` or a small dataclass; `Any` keeps families flexible.
        `step` is the global optimizer step (schedules, logging).
        """
        ...

    def eval_step(self, batch: Any) -> LossDict:
        """Validation / benchmark metrics; must not retain autograd graphs for training.

        Callers should invoke this inside `torch.no_grad()` (or implementers may use
        `@torch.no_grad()` on the concrete method).
        """
        ...


class AuditorModel(GenerativeTrainingModel, Protocol):
    """Models that score or audit batches at inference time (benchmarks, eval).

    Implementations should keep `audit` cheap and graph-free; benchmarks call it under `torch.no_grad()`.
    """

    def audit(self, batch: Any) -> LossDict:
        """Named scalar tensors suitable for averaging (same shape rules as `eval_step`)."""
        ...
