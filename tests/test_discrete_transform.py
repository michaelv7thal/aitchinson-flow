"""Unit tests for the discrete→simplex transform pipeline.

Covers the explicit label-smoothing path, ILR vs CLR mode dispatch, and the
shape contracts that downstream models depend on (see plan point 4 + the
ILR-on/off ablation in plan point 5).
"""

from __future__ import annotations

import math

import torch

from aitchinson_flow.config import Config
from aitchinson_flow.data.feature_dim import feature_dim
from aitchinson_flow.data.transforms.discrete import (
    token_ids_to_clr_x,
    token_ids_to_features,
    token_ids_to_ilr_x,
)


class TestFeatureDim:
    def test_ilr_mode_returns_k_minus_one(self) -> None:
        cfg = Config()
        cfg.dataset.K = 27
        cfg.hf_dataset.transform_mode = "ilr"
        assert feature_dim(cfg) == 26

    def test_clr_mode_returns_k(self) -> None:
        cfg = Config()
        cfg.dataset.K = 27
        cfg.hf_dataset.transform_mode = "clr"
        assert feature_dim(cfg) == 27

    def test_unknown_mode_raises(self) -> None:
        cfg = Config()
        cfg.hf_dataset.transform_mode = "bogus"
        try:
            feature_dim(cfg)
        except ValueError as e:
            assert "transform_mode" in str(e)
        else:
            raise AssertionError("unknown transform_mode must raise ValueError")


class TestLabelSmoothing:
    def test_default_eps_path_matches_legacy(self) -> None:
        ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        K = 5
        out = token_ids_to_ilr_x(ids, K=K, eps=1e-8, label_smoothing=0.0)
        # Legacy reference: one_hot + eps + log + ilr
        from aitchinson_flow.geometry import ilr

        oh = torch.zeros(ids.shape[0], K)
        oh.scatter_(dim=-1, index=ids.unsqueeze(-1), value=1.0)
        ref = ilr((oh + 1e-8).log())
        assert torch.allclose(out, ref, atol=1e-6)

    def test_label_smoothing_row_sums_to_one_in_prob_space(self) -> None:
        ids = torch.tensor([0, 1, 2], dtype=torch.long)
        K = 5
        alpha = 0.1
        # Convert through CLR mode then exponentiate to recover the row.
        clr = token_ids_to_clr_x(ids, K=K, label_smoothing=alpha)
        # CLR -> log-prob (additive constant chosen so logsumexp = 0).
        log_p = clr - clr.logsumexp(dim=-1, keepdim=True)
        p = log_p.exp()
        assert torch.allclose(p.sum(dim=-1), torch.ones(ids.shape[0]), atol=1e-5)
        # The peak of each row must equal 1 - alpha + alpha/K
        peak = p.max(dim=-1).values
        expected_peak = (1 - alpha) + alpha / K
        assert torch.allclose(peak, torch.full_like(peak, expected_peak), atol=1e-5)

    def test_label_smoothing_ignores_eps(self) -> None:
        """When ``label_smoothing > 0`` the additive eps is intentionally not applied."""
        ids = torch.tensor([0, 1, 2], dtype=torch.long)
        K = 5
        alpha = 0.1
        a = token_ids_to_ilr_x(ids, K=K, eps=1e-8, label_smoothing=alpha)
        b = token_ids_to_ilr_x(ids, K=K, eps=1e-2, label_smoothing=alpha)
        assert torch.allclose(a, b, atol=1e-6), (
            "label_smoothing path must be independent of eps"
        )

    def test_label_smoothing_out_of_range_raises(self) -> None:
        ids = torch.tensor([0, 1], dtype=torch.long)
        for bad in (-0.1, 1.0, 1.5):
            try:
                token_ids_to_ilr_x(ids, K=5, label_smoothing=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"label_smoothing={bad} must raise ValueError")


class TestTransformModeDispatch:
    def test_ilr_shape_is_k_minus_one(self) -> None:
        ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        K = 6
        out = token_ids_to_features(ids, K=K, transform_mode="ilr")
        assert out.shape == (ids.shape[0], K - 1)

    def test_clr_shape_is_k_and_rows_sum_to_zero(self) -> None:
        ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        K = 6
        out = token_ids_to_features(ids, K=K, transform_mode="clr", label_smoothing=0.1)
        assert out.shape == (ids.shape[0], K)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-5)


class TestModelInputDimRespectsTransformMode:
    def test_transformer_backbone_input_dim_tracks_feature_dim(self) -> None:
        from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead

        cfg = Config()
        cfg.dataset.K = 7
        cfg.dataset.L = 4
        cfg.transformer.d_model = 16
        cfg.transformer.nhead = 2
        cfg.transformer.num_layers = 1
        cfg.training.device = torch.device("cpu")

        # ILR
        cfg.hf_dataset.transform_mode = "ilr"
        bb = TransformerBackbone(cfg)
        vh = VelocityHead(cfg)
        assert bb.input_proj.in_features == 6
        assert vh.proj.out_features == 6

        # CLR
        cfg.hf_dataset.transform_mode = "clr"
        bb2 = TransformerBackbone(cfg)
        vh2 = VelocityHead(cfg)
        assert bb2.input_proj.in_features == 7
        assert vh2.proj.out_features == 7


class TestTransformerDropout:
    """Regression test for M4 — dropout must be wired from the config."""

    def test_dropout_active_in_train_mode(self) -> None:
        from aitchinson_flow.transformer_backbone import TransformerBackbone

        cfg = Config()
        cfg.dataset.K = 7
        cfg.dataset.L = 4
        cfg.transformer.d_model = 16
        cfg.transformer.nhead = 2
        cfg.transformer.num_layers = 2
        cfg.transformer.dropout = 0.5
        cfg.training.device = torch.device("cpu")
        cfg.hf_dataset.transform_mode = "ilr"

        bb = TransformerBackbone(cfg).train()
        torch.manual_seed(0)
        x = torch.randn(2, cfg.dataset.L, 6)
        torch.manual_seed(1)
        out_a = bb(x)
        torch.manual_seed(2)
        out_b = bb(x)
        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            "TransformerBackbone must apply dropout in train() mode when "
            "cfg.transformer.dropout > 0; two forward passes on the same input "
            "should produce different outputs"
        )

    def test_dropout_inactive_in_eval_mode(self) -> None:
        from aitchinson_flow.transformer_backbone import TransformerBackbone

        cfg = Config()
        cfg.dataset.K = 7
        cfg.dataset.L = 4
        cfg.transformer.d_model = 16
        cfg.transformer.nhead = 2
        cfg.transformer.num_layers = 2
        cfg.transformer.dropout = 0.5
        cfg.training.device = torch.device("cpu")
        cfg.hf_dataset.transform_mode = "ilr"

        bb = TransformerBackbone(cfg).eval()
        x = torch.randn(2, cfg.dataset.L, 6)
        out_a = bb(x)
        out_b = bb(x)
        assert torch.allclose(out_a, out_b, atol=1e-6), (
            "TransformerBackbone must be deterministic in eval() mode"
        )
