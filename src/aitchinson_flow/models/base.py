from __future__ import annotations

from typing import Protocol, Any

import torch

TRAINING_LOSS_KEY = "loss"
"""Scalar tensor key required in `training_step` return dicts."""

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
