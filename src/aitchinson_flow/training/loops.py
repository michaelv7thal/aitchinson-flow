from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from aitchinson_flow.training import (
    to_device,
    running_average,
    finalize_averages,
    detach_means,
)
from aitchinson_flow.models import GenerativeTrainingModel, LossDict, TRAINING_LOSS_KEY


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    *,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_clip_norm: float | None = None,
    step_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict[str, float], int]:
    """Returns (epoch_metrics, next_global_step)."""
    m = cast(GenerativeTrainingModel, model)
    model.train()

    agg: dict[str, float] = {}
    counts: dict[str, int] = {}
    step = global_step

    pbar = tqdm(
        loader,
        desc=f"train epoch: {epoch + 1}",
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

        postfix = _loss_postfix(out)

        if postfix:
            pbar.set_postfix(postfix)

        step += 1

    return finalize_averages(agg, counts), step


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    m = cast(GenerativeTrainingModel, model)
    model.eval()

    agg: dict[str, float] = {}
    counts: dict[str, int] = {}

    pbar = tqdm(loader, desc="validation", leave=False)

    for batch in pbar:
        batch = to_device(batch, device)
        out = m.eval_step(batch)
        if TRAINING_LOSS_KEY not in out:
            raise KeyError(
                f"eval_step() for {type(model).__name__} must return a dict "
                f"containing the {TRAINING_LOSS_KEY!r} key; got keys {sorted(out.keys())}."
            )

        running_average(agg, counts, out)
        postfix = _loss_postfix(out)
        if postfix:
            pbar.set_postfix(postfix)

    return finalize_averages(agg, counts)


_GAMMA_BIN_KEYS = ("g<.33", "g<.66", "g<1", "mse")


def _loss_postfix(out: LossDict) -> dict[str, str]:
    postfix: dict[str, str] = {}
    total = _to_float_scalar(out.get(TRAINING_LOSS_KEY))

    if total is not None:
        postfix["total_loss"] = f"{total:.4f}"

    for key in _GAMMA_BIN_KEYS:
        val = _to_float_scalar(out.get(key))
        if val is not None:
            postfix[key] = f"{val:.4f}"

    bpd = _to_float_scalar(out.get("bpd"))
    if bpd is not None:
        postfix["bpd"] = f"{bpd:.4f}"

    return postfix


def _to_float_scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        vt = value.detach()
        if vt.numel() == 0:
            return None
        return float(vt.mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None
