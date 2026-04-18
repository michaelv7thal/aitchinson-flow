from __future__ import annotations

from typing import Iterable

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


# GP hyperparameter attribute names that should not receive weight decay.
# Decaying inducing locations or kernel hyperparameters toward zero is
# semantically wrong (it biases lengthscales, outputscale, and inducing
# positions to the origin) and empirically destabilises ELBO training.
_GP_HYPERPARAM_ATTRS: tuple[str, ...] = (
    "Z",
    "var_mean",
    "var_L_raw",
    "log_lengthscale",
    "log_outputscale",
    "log_noise_var",
)


def _collect_gp_param_ids(model: nn.Module) -> set[int]:
    """IDs of all tensors that live on a ``SparseGP`` / ``ProductSparseGP`` module.

    We match structurally rather than by name so subclasses and product-kernel
    variants are covered without an allow-list.
    """
    # Local import to avoid an import cycle at module load.
    from aitchinson_flow.gp.gp import ProductSparseGP, SparseGP

    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (SparseGP, ProductSparseGP)):
            for attr in _GP_HYPERPARAM_ATTRS:
                p = getattr(module, attr, None)
                if isinstance(p, nn.Parameter):
                    ids.add(id(p))
    return ids


def build_optimizer(model: nn.Module, cfg: Config) -> AdamW:
    """Build the optimizer over the model's *trainable* parameters.

    If the model exposes a ``trainable_parameters()`` callable (e.g. Stage 2
    of the two-stage auditor, where the backbone is frozen), use it; otherwise
    fall back to ``model.parameters()``. This keeps optimizer state from
    being allocated for frozen tensors and makes the trainable surface
    explicit.

    GP hyperparameters (inducing locations, kernel lengthscale/outputscale,
    noise variance, variational mean/covariance) are placed in a second
    parameter group with ``weight_decay=0.0`` (M5 fix). Other parameters use
    ``cfg.training.weight_decay`` as before.
    """
    fn = getattr(model, "trainable_parameters", None)
    params: Iterable[nn.Parameter]
    if callable(fn):
        params = list(fn())
        if not params:
            params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            params = list(model.parameters())

    gp_ids = _collect_gp_param_ids(model)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in params:
        (no_decay if id(p) in gp_ids else decay).append(p)

    param_groups: list[dict] = []
    if decay:
        param_groups.append(
            {"params": decay, "weight_decay": cfg.training.weight_decay}
        )
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    if not param_groups:
        # Preserve the historical fallback when a model has no parameters at all.
        param_groups = [{"params": list(model.parameters()), "weight_decay": cfg.training.weight_decay}]

    return AdamW(param_groups, lr=cfg.training.lr)


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
                total_iters=max(1, epochs),
            )
        # ``epochs`` is already clamped to >= 1 by ``build_scheduler``; with
        # ``warmup < epochs`` we have ``cos_epochs >= 1``, but keep the clamp
        # explicit so short runs (epochs=1, warmup=0 ⇒ cos_epochs=1) still
        # produce a valid ``T_max``.
        cos_epochs = max(1, epochs - warmup)
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
