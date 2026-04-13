from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from aitchinson_flow.models.base import TRAINING_LOSS_KEY, GenerativeTrainingModel, LossDict
from aitchinson_flow.training.metrics import detach_means, finalize_averages, running_average


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: Optimizer,
    *,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_clip_norm: float | None = None,
) -> tuple[dict[str, float], int]:
    """Returns (epoch_metrics, next_global_step)."""
    m = cast(GenerativeTrainingModel, model)
    model.train()
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    step = global_step

    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out: LossDict = m.training_step(batch, step)
        loss = out[TRAINING_LOSS_KEY]
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_average(agg, counts, out)
        step += 1

    return finalize_averages(agg, counts), step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> dict[str, float]:
    m = cast(GenerativeTrainingModel, model)
    model.eval()
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}

    for batch in loader:
        batch = _to_device(batch, device)
        out = m.eval_step(batch)
        running_average(agg, counts, out)

    return finalize_averages(agg, counts)


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)

    return batch
