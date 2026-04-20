"""Shared smoke ``Config`` factory for ``scripts/`` entrypoints.

Every objective script (two-stage, single-stage, future distillation, ...)
should expose a ``--smoke`` flag that builds a tiny CPU-only config via
:func:`make_smoke_config`. Tests also import this directly to keep all smoke
runs on identical baseline hyperparameters.

Objectives that need to specialize smoke defaults (e.g. set
``cfg.training.model_name = "bayesian_auditor_stage1"``) should mutate the
returned ``Config`` in their own helper rather than duplicating the shared
baseline.
"""

from __future__ import annotations

import torch

from aitchinson_flow.config import Config


def make_smoke_config() -> Config:
    """Return a tiny CPU-only ``Config`` used by end-to-end smoke tests."""
    cfg = Config()
    cfg.training.device = torch.device("cpu")
    cfg.training.seed = 0
    cfg.training.B = 4
    cfg.training.lr = 1e-3
    cfg.training.checkpoint_every = 1
    cfg.training.lr_scheduler = None
    cfg.training.use_tqdm = False
    cfg.dataset.K = 27
    cfg.dataset.L = 8
    cfg.transformer.d_model = 16
    cfg.transformer.nhead = 2
    cfg.transformer.num_layers = 1
    cfg.transformer.d_latent = 16
    cfg.gp.num_inducing = 8
    cfg.benchmark.use_tqdm = False
    return cfg


__all__ = ["make_smoke_config"]
