from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    SequentialLR,
)

from aitchinson_flow.config import Config


def build_optimizer(model: nn.Module, cfg: Config) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )


def build_scheduler(optimizer: AdamW, cfg: Config) -> LRScheduler | None:
    """Return an LR schedule stepped **once per epoch** (see ``training.runner.fit``).

    ``OneCycleLR`` is configured with ``total_steps == training.epochs`` so one cycle runs
    across the full training run when ``step()`` is called after each epoch.
    """
    raw = cfg.training.lr_scheduler
    name = (raw or "").strip().lower()
    if name in ("", "none", "constant"):
        return None

    tr = cfg.training
    epochs = max(1, tr.epochs)

    if name == "cosine":
        return _build_cosine(optimizer, tr, epochs)
    if name in ("cosine_restarts", "cosine_with_restarts"):
        return _build_cosine_restarts(optimizer, tr)
    if name in ("onecycle", "one_cycle"):
        return _build_onecycle(optimizer, tr, epochs)
    if name == "linear":
        return _build_linear(optimizer, tr, epochs)
    if name == "polynomial":
        return PolynomialLR(optimizer, total_iters=epochs, power=tr.polynomial_power)
    if name == "exponential":
        return ExponentialLR(optimizer, gamma=tr.exponential_gamma)
    if name in ("multistep", "multi_step"):
        if not tr.multistep_milestones:
            raise ValueError(
                "lr_scheduler='multistep' requires non-empty training.multistep_milestones "
                f"(got {tr.multistep_milestones!r})"
            )
        return MultiStepLR(optimizer, milestones=list(tr.multistep_milestones), gamma=tr.multistep_gamma)

    raise ValueError(
        f"Unknown lr_scheduler {raw!r}. Use None, constant, cosine, cosine_restarts, onecycle, "
        f"linear, polynomial, exponential, or multistep."
    )


def _build_cosine(optimizer: AdamW, tr, epochs: int) -> LRScheduler:
    warmup = max(0, tr.scheduler_warmup_epochs)
    eta_min = tr.cosine_eta_min

    if warmup > 0:
        if warmup >= epochs:
            return LinearLR(
                optimizer,
                start_factor=_clamp_unit_open(tr.scheduler_warmup_start_factor, "scheduler_warmup_start_factor"),
                end_factor=1.0,
                total_iters=epochs,
            )
        cos_epochs = epochs - warmup
        t_max = tr.cosine_t_max_epochs if tr.cosine_t_max_epochs is not None else cos_epochs
        t_max = max(1, t_max)
        main = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        warm = LinearLR(
            optimizer,
            start_factor=_clamp_unit_open(tr.scheduler_warmup_start_factor, "scheduler_warmup_start_factor"),
            end_factor=1.0,
            total_iters=warmup,
        )
        return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup])

    t_max = tr.cosine_t_max_epochs if tr.cosine_t_max_epochs is not None else epochs
    t_max = max(1, t_max)
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def _clamp_unit_open(x: float, field: str) -> float:
    """``LinearLR`` requires ``start_factor`` in (0, 1]."""
    if not (0.0 < x <= 1.0):
        raise ValueError(f"{field} must be in (0, 1], got {x}")
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
    return LinearLR(optimizer, start_factor=1.0, end_factor=ef, total_iters=max(1, epochs))
