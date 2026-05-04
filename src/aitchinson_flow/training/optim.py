from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import (
    LRScheduler,
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from aitchinson_flow.config import Config


def build_optimizer(model: nn.Module, cfg: Config) -> AdamW:

    fn = getattr(model, "trainable_parameters", None)
    params: Iterable[nn.Parameter]

    if callable(fn):
        result = fn()
        params = list(result) if isinstance(result, Iterable) else []

        if not params:
            params = [p for p in model.parameters() if p.requires_grad]

    else:
        params = [p for p in model.parameters() if p.requires_grad]

        if not params:
            params = list(model.parameters())

    return AdamW(params=params, lr=cfg.training.lr)


def build_scheduler(optimizer: AdamW, cfg: Config) -> LRScheduler | None:

    raw = cfg.training.lr_sheduler
    name = (raw or "").strip().lower()

    if name in ("", "none", "constant"):
        return None

    if name == "cosine":
        return _build_cosine(optimizer, cfg, cfg.training.epochs)


def _build_cosine(optimizer: AdamW, cfg: Config, epochs: int) -> LRScheduler:
    warmup = max(0, cfg.training.scheduler_warmup_epochs)
    eta_min = cfg.training.cosine_eta_min

    if warmup > 0.0:
        if warmup >= epochs:
            return LinearLR(
                optimizer,
                start_factor=_clamp_unit_open(
                    cfg.training.scheduler_warmup_start_factor,
                    "scheduler_warmup_start_factor",
                ),
                end_factor=1.0,
                total_iters=max(1, epochs),
            )

        cos_epochs = max(1, epochs - warmup)
        t_max = (
            cfg.training.cosine_t_max_epochs
            if cfg.training.cosine_t_max_epochs is not None
            else cos_epochs
        )
        t_max = max(1, t_max)

        main = CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=eta_min)
        warm = LinearLR(
            optimizer=optimizer,
            start_factor=_clamp_unit_open(
                cfg.training.scheduler_warmup_start_factor,
                "scheduler_warmup_start_factor",
            ),
            end_factor=1.0,
            total_iters=warmup,
        )
        return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup])

    t_max = (
        cfg.training.cosine_t_max_epochs
        if cfg.training.cosine_t_max_epochs is not None
        else epochs
    )
    t_max = max(1, t_max)
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def _clamp_unit_open(x: float, field: str) -> float:
    """``LinearLR`` requires ``start_factor`` in (0, 1]."""
    if not (0.0, x <= 1.0):
        raise ValueError(f"{field} must be in (0,1], got {x})")

    return x


def _build_cosine_restarts(optimizer: AdamW, tr) -> CosineAnnealingWarmRestarts:
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, tr.cosine_restart_t0_epochs),
        T_mult=max(1, tr.cosine_restart_t_mult),
        eta_min=tr.cosine_eta_min,
    )


def _build_onecycle(optimizer: AdamW, tr, epochs: int) -> OneCycleLR:
    return OneCycleLR(
        optimizer,
        max_lr=tr.lr,
        total_steps=epochs,
        pct_start=tr.onecycle_pct_start,
        anneal_strategy="cos",
        div_factor=tr.onecycle_div_factor,
        final_div_factor=tr.onecycle_final_div_factor,
        three_phase=tr.onecycle_three_phase,
        cycle_momentum=tr.onecycle_cycle_momentum,
    )


def _build_linear(optimizer: AdamW, tr, epochs: int) -> LinearLR:
    ef = tr.linear_end_factor
    if ef < 0 or ef > 1:
        raise ValueError(f"training.linear_end_factor must be in [0, 1], got {ef}")
    return LinearLR(
        optimizer, start_factor=1.0, end_factor=ef, total_iters=max(1, epochs)
    )
