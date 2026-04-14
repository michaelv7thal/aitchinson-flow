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
    use_tqdm: bool = True,
) -> tuple[dict[str, float], int]:
    """Returns (epoch_metrics, next_global_step)."""
    m = cast(GenerativeTrainingModel, model)
    model.train()
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    step = global_step
    running_loss_sum = 0.0
    running_n = 0

    from tqdm.auto import tqdm  # noqa: PLC0415

    pbar = tqdm(
        loader,
        desc=f"train epoch {epoch + 1}",
        disable=not use_tqdm,
        leave=False,
    )
    for batch in pbar:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out: LossDict = m.training_step(batch, step)
        loss = out[TRAINING_LOSS_KEY]
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_average(agg, counts, out)
        if use_tqdm:
            loss_f = float(loss.detach().cpu())
            running_loss_sum += loss_f
            running_n += 1
            avg_f = running_loss_sum / running_n
            pbar.set_postfix(last=f"{loss_f:.4f}", avg=f"{avg_f:.4f}")
        step += 1

    return finalize_averages(agg, counts), step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    use_tqdm: bool = True,
) -> dict[str, float]:
    m = cast(GenerativeTrainingModel, model)
    model.eval()
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}

    from tqdm.auto import tqdm  # noqa: PLC0415

    pbar = tqdm(loader, desc="validation", disable=not use_tqdm, leave=False)
    for batch in pbar:
        batch = _to_device(batch, device)
        out = m.eval_step(batch)
        running_average(agg, counts, out)
        if use_tqdm and TRAINING_LOSS_KEY in out:
            lt = out[TRAINING_LOSS_KEY].detach()
            lv = float(lt.mean().cpu() if lt.ndim > 0 else lt.cpu())
            pbar.set_postfix(loss=f"{lv:.4f}")

    return finalize_averages(agg, counts)


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)

    return batch
