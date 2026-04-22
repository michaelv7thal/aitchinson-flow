"""Path B smoke test: Stage 1 trains llm_projection + backbone jointly; Stage 2 freezes both."""

from __future__ import annotations

import torch

from aitchinson_flow.config import Config, LLMEmbeddingDatasetConfig
from aitchinson_flow.models import (
    BayesianAuditorStage1,
    BayesianAuditorStage2,
    compose_auditor_from_stages,
)
from aitchinson_flow.models.base import TRAINING_LOSS_KEY


def _path_b_cfg(*, d_embed: int = 12, K: int = 6, L: int = 6, B: int = 2) -> Config:
    cfg = Config()
    cfg.dataset.K = K
    cfg.dataset.L = L
    cfg.transformer.d_model = 16
    cfg.transformer.nhead = 2
    cfg.transformer.num_layers = 1
    cfg.transformer.d_latent = 8
    cfg.gp.num_inducing = 8
    cfg.training.B = B
    cfg.training.device = torch.device("cpu")
    cfg.training.velocity_loss = "soft_hilbert"
    cfg.training_data.source = "llm_topk"
    cfg.llm_embedding_dataset = LLMEmbeddingDatasetConfig(
        char_window_length=16,
        corrupt_rate=0.3,
        generation_seed=1,
        llm_embed_dim=d_embed,
    )
    return cfg


def _path_b_batch(cfg: Config, *, seed: int = 0) -> dict[str, torch.Tensor]:
    B, L = cfg.training.B, cfg.dataset.L
    d_embed = cfg.llm_embedding_dataset.llm_embed_dim
    assert d_embed is not None
    gen = torch.Generator().manual_seed(seed)
    return {
        "embeddings": torch.randn(B, L, d_embed, generator=gen),
        "embeddings_invalid": torch.randn(B, L, d_embed, generator=gen) * 1.3 + 0.1,
        "token_ids": torch.randint(0, 1000, (B, L), generator=gen),
        "token_ids_invalid": torch.randint(0, 1000, (B, L), generator=gen),
    }


class TestPathBStage1:
    def test_projection_and_backbone_receive_gradients(self) -> None:
        cfg = _path_b_cfg()
        model = BayesianAuditorStage1(cfg)
        assert model.llm_projection is not None

        batch = _path_b_batch(cfg)
        out = model.training_step(batch, step=0)
        loss = out[TRAINING_LOSS_KEY]
        loss.backward()

        # llm_projection.proj.weight must receive a non-trivial gradient.
        proj_grad = model.llm_projection.proj.weight.grad
        assert proj_grad is not None
        assert torch.isfinite(proj_grad).all()
        assert proj_grad.abs().sum() > 0.0

        # Backbone must also receive gradients (Stage 1 trains both).
        any_backbone_grad = False
        for p in model.backbone.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0.0:
                any_backbone_grad = True
                break
        assert any_backbone_grad

    def test_prepare_batch_populates_log_x(self) -> None:
        cfg = _path_b_cfg()
        model = BayesianAuditorStage1(cfg)
        batch = _path_b_batch(cfg)
        prepared = model.prepare_batch(batch)
        assert "log_x" in prepared
        assert "log_x_invalid" in prepared
        # ILR output dim = K-1 by default.
        assert prepared["log_x"].shape == (cfg.training.B, cfg.dataset.L, cfg.dataset.K - 1)

    def test_path_a_batch_is_unchanged(self) -> None:
        """Path A batches (plain log_x) pass through prepare_batch untouched."""
        cfg = _path_b_cfg()
        model = BayesianAuditorStage1(cfg)
        plain = {"log_x": torch.randn(2, cfg.dataset.L, cfg.dataset.K - 1)}
        prepared = model.prepare_batch(plain)
        assert torch.equal(prepared["log_x"], plain["log_x"])
        assert "embeddings" not in prepared


class TestPathBStage2:
    def test_projection_is_frozen(self) -> None:
        cfg = _path_b_cfg()
        model = BayesianAuditorStage2(cfg)
        assert model.llm_projection is not None
        for p in model.llm_projection.parameters():
            assert p.requires_grad is False
        # After .train(), the projection stays in eval mode alongside the backbone.
        model.train()
        assert model.llm_projection.training is False

    def test_training_step_does_not_update_projection(self) -> None:
        cfg = _path_b_cfg()
        model = BayesianAuditorStage2(cfg)
        batch = _path_b_batch(cfg)

        out = model.training_step(batch, step=0)
        loss = out[TRAINING_LOSS_KEY]
        loss.backward()

        for p in model.llm_projection.parameters():
            # No gradient was computed (requires_grad=False) — ``p.grad`` stays None.
            assert p.grad is None


class TestPathBFusedCompose:
    def test_stage1_projection_transfers_to_fused(self) -> None:
        cfg = _path_b_cfg()
        stage1 = BayesianAuditorStage1(cfg)
        stage2 = BayesianAuditorStage2(cfg)

        # Take a few steps on stage1 to push the projection off its init.
        for step in range(2):
            out = stage1.training_step(_path_b_batch(cfg, seed=step), step=step)
            out[TRAINING_LOSS_KEY].backward()

        fused = compose_auditor_from_stages(
            cfg,
            stage1_state=dict(stage1.state_dict()),
            stage2_state=dict(stage2.state_dict()),
            strict=False,
        )
        assert fused.llm_projection is not None
        assert torch.allclose(
            fused.llm_projection.proj.weight,
            stage1.llm_projection.proj.weight,
        )
        assert torch.allclose(
            fused.llm_projection.proj.bias,
            stage1.llm_projection.proj.bias,
        )
