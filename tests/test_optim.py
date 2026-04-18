"""Regression tests for optimizer and scheduler construction (M5, M10)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.training.optim import build_optimizer, build_scheduler


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.dataset.K = 5
    cfg.dataset.L = 4
    cfg.transformer.d_model = 16
    cfg.transformer.nhead = 2
    cfg.transformer.num_layers = 1
    cfg.transformer.d_latent = 8
    cfg.training.B = 2
    cfg.training.device = torch.device("cpu")
    cfg.gp.num_inducing = 4
    return cfg


class _DummyModel(nn.Module):
    """Single linear layer; used for scheduler tests that don't need a real model."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)


class TestSchedulerShortRuns:
    """M10 — cosine schedulers must not fail on epochs=1 / warmup edge cases."""

    @pytest.mark.parametrize(
        ("epochs", "warmup"),
        [(1, 0), (1, 1), (2, 1), (2, 2), (3, 1)],
    )
    def test_cosine_scheduler_handles_short_runs(self, epochs: int, warmup: int) -> None:
        cfg = _tiny_cfg()
        cfg.training.lr_scheduler = "cosine"
        cfg.training.epochs = epochs
        cfg.training.scheduler_warmup_epochs = warmup

        model = _DummyModel()
        opt = build_optimizer(model, cfg)
        sched = build_scheduler(opt, cfg)
        assert sched is not None, "expected a non-None scheduler for cosine"
        # One epoch of stepping should not raise (catches T_max=0 regressions).
        for _ in range(epochs):
            opt.step()
            sched.step()


class TestOptimizerParamGroups:
    """M5 — GP hyperparameters must live in a weight_decay=0 group."""

    def test_non_gp_model_has_single_or_zero_decay_group(self) -> None:
        cfg = _tiny_cfg()
        cfg.training.weight_decay = 0.25
        model = _DummyModel()
        opt = build_optimizer(model, cfg)
        # No GP module → everything is in the decay group.
        for group in opt.param_groups:
            if group["params"]:
                assert group["weight_decay"] == 0.25, (
                    "non-GP models should use cfg.training.weight_decay"
                )
