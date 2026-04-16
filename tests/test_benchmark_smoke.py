"""Smoke test: end-to-end benchmark pipeline with a mock LM (no real GPT-2)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import torch
import pytest

from aitchinson_flow.config import Config
from aitchinson_flow.data.teachers.causal_lm import CausalLMTeacher, CausalLMTeacherDataModule


class FakeLM:
    """Deterministic mock matching CausalLMForInference protocol."""

    def __init__(self, vocab_size: int = 50, device: torch.device = torch.device("cpu")) -> None:
        self._vocab = vocab_size
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.randn(B, L, self._vocab, device=self._device)

    def encode_text(self, text: str, *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.randint(0, self._vocab, (1, max_length))
        mask = torch.ones_like(ids)
        return ids, mask

    def generate_ids(
        self,
        *,
        batch_size: int,
        max_new_tokens: int,
        prompt_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        L = max_new_tokens
        if prompt_ids is not None:
            L += prompt_ids.shape[1]
            batch_size = prompt_ids.shape[0]
        return torch.randint(0, self._vocab, (batch_size, L), device=self._device)


def _make_cfg(*, K: int = 27, L: int = 20, n_batches: int = 2, B: int = 4) -> Config:
    cfg = Config()
    cfg.dataset.K = K
    cfg.dataset.L = L
    cfg.training.B = B
    cfg.training.model_name = "bayesian_auditor"
    cfg.benchmark.n_batches = n_batches
    cfg.benchmark.use_text8_prompts = False
    cfg.benchmark.compute_spilled_energy = True
    cfg.benchmark.corrupt_rate = 0.3
    return cfg


class TestTeacherEmitsLogits:
    def test_sample_with_logits_keys(self) -> None:
        cfg = _make_cfg()
        lm = FakeLM(vocab_size=50)
        teacher = CausalLMTeacher(lm, cfg)
        out = teacher.sample_with_logits(batch_size=4)
        assert "log_x" in out
        assert "token_ids" in out
        assert "logits" in out
        assert out["log_x"].shape == (4, cfg.dataset.L, cfg.dataset.K - 1)
        assert out["token_ids"].shape == (4, cfg.dataset.L)
        assert out["logits"].shape[0] == 4
        assert out["logits"].shape[1] == cfg.dataset.L

    def test_datamodule_yields_logits(self) -> None:
        cfg = _make_cfg(n_batches=2, B=3)
        lm = FakeLM(vocab_size=50)
        teacher = CausalLMTeacher(lm, cfg)
        dm = CausalLMTeacherDataModule(teacher, cfg, emit_logits=True)
        loader = dm.train_dataloader()
        batches = list(loader)
        assert len(batches) == 2
        for b in batches:
            assert "logits" in b
            assert "token_ids" in b


class TestTextAuditTaskSmoke:
    def test_produces_spilled_and_auditor_metrics(self) -> None:
        import benchmarks.tasks  # noqa: F401 — register tasks

        from benchmarks.tasks.registry import build_task

        cfg = _make_cfg(n_batches=2, B=4)
        lm = FakeLM(vocab_size=50)
        teacher = CausalLMTeacher(lm, cfg)
        dm = CausalLMTeacherDataModule(teacher, cfg, emit_logits=True)
        task = build_task("text_audit")

        from aitchinson_flow.models.factory import build_model

        model = build_model(cfg)
        result = task.run(model, dm, cfg)

        assert isinstance(result, dict)
        # Spilled-energy baseline metrics
        assert "spilled_mean_valid" in result
        assert "spilled_mean_invalid" in result
        assert "spilled_anomaly_valid" in result
        assert "spilled_anomaly_invalid" in result
        assert "spilled_separation" in result
        # Bayesian auditor metrics (with log_x_invalid present → eval_step path)
        assert "flow_loss" in result
        assert "mean_loss" in result
        assert "var_loss" in result
        # AUROC benchmark: auditor vs spilled energy
        assert "auroc_auditor" in result
        assert "auroc_spilled" in result
