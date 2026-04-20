"""Tests for AUROC computation in the text_audit task."""

from __future__ import annotations

import numpy as np
import torch

from aitchinson_flow.metrics.auroc import safe_auroc


class TestSafeAuroc:
    def test_perfectly_separable(self) -> None:
        v = np.zeros(100, dtype=np.float32)
        i = np.ones(100, dtype=np.float32)
        assert safe_auroc(v, i) == 1.0

    def test_swapped_returns_zero(self) -> None:
        v = np.ones(100, dtype=np.float32)
        i = np.zeros(100, dtype=np.float32)
        assert safe_auroc(v, i) == 0.0

    def test_random_overlap_around_half(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.normal(0, 1, size=500).astype(np.float32)
        i = rng.normal(0, 1, size=500).astype(np.float32)
        auc = safe_auroc(v, i)
        assert 0.4 < auc < 0.6

    def test_empty_returns_nan(self) -> None:
        auc = safe_auroc(np.empty(0, dtype=np.float32), np.ones(10, dtype=np.float32))
        assert np.isnan(auc)

    def test_shifted_gaussians_above_half(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.normal(0.0, 1.0, size=500).astype(np.float32)
        i = rng.normal(2.0, 1.0, size=500).astype(np.float32)
        auc = safe_auroc(v, i)
        assert auc > 0.85


class TestAuditorScoreHook:
    """`score_per_sample` on BayesianAuditor must return (B,)."""

    def test_bayesian_auditor(self) -> None:
        from aitchinson_flow.config import Config
        from aitchinson_flow.models.factory import build_model
        import aitchinson_flow.models  # noqa: F401

        cfg = Config()
        cfg.dataset.K = 8
        cfg.dataset.L = 6
        cfg.training.model_name = "bayesian_auditor"
        cfg.training.device = torch.device("cpu")
        cfg.transformer.d_model = 16
        cfg.transformer.num_layers = 1
        cfg.transformer.nhead = 2
        cfg.transformer.d_latent = 16

        model = build_model(cfg).eval()
        log_x = torch.randn(3, cfg.dataset.L, cfg.dataset.K)
        scores = model.score_per_sample(log_x)
        assert scores.shape == (3,)
