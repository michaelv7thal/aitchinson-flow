from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from aitchinson_flow.models.base import TRAINING_LOSS_KEY, GenerativeTrainingModel, LossDict
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.training.metrics import detach_means, finalize_averages, running_average


def _to_float_scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        vt = value.detach()
        if vt.numel() == 0:
            return None
        return float(vt.mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _loss_postfix(out: LossDict) -> dict[str, str]:
    postfix: dict[str, str] = {}

    total = _to_float_scalar(out.get(TRAINING_LOSS_KEY))
    if total is not None:
        postfix["total_loss"] = f"{total:.4f}"

    velocity = _to_float_scalar(out.get("velocity_loss"))
    if velocity is None:
        # Legacy models may only expose flow_loss; keep tqdm stable for them.
        velocity = _to_float_scalar(out.get("flow_loss"))
    if velocity is not None:
        postfix["velocity_loss"] = f"{velocity:.4f}"

    mask = _to_float_scalar(out.get("mask_loss"))
    if mask is not None:
        postfix["mask_loss"] = f"{mask:.4f}"

    return postfix


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
    step_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict[str, float], int]:
    """Returns (epoch_metrics, next_global_step)."""
    m = cast(GenerativeTrainingModel, model)
    model.train()
    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    step = global_step
    from tqdm.auto import tqdm  # noqa: PLC0415

    pbar = tqdm(
        loader,
        desc=f"train epoch {epoch + 1}",
        disable=not use_tqdm,
        leave=False,
    )
    for batch in pbar:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out: LossDict = m.training_step(batch, step)
        if TRAINING_LOSS_KEY not in out:
            raise KeyError(
                f"training_step() for {type(model).__name__} must return a dict "
                f"containing the {TRAINING_LOSS_KEY!r} key (a scalar loss tensor); "
                f"got keys {sorted(out.keys())}."
            )
        loss = out[TRAINING_LOSS_KEY]
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_average(agg, counts, out)
        if step_callback is not None:
            step_callback(step, detach_means(out))
        if use_tqdm:
            postfix = _loss_postfix(out)
            if postfix:
                pbar.set_postfix(postfix)
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
        batch = to_device(batch, device)
        out = m.eval_step(batch)
        if TRAINING_LOSS_KEY not in out:
            raise KeyError(
                f"eval_step() for {type(model).__name__} must return a dict "
                f"containing the {TRAINING_LOSS_KEY!r} key; got keys {sorted(out.keys())}."
            )
        running_average(agg, counts, out)
        if use_tqdm:
            postfix = _loss_postfix(out)
            if postfix:
                pbar.set_postfix(postfix)

    return finalize_averages(agg, counts)
