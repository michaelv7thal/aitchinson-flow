"""Tests for training data-source dispatch and CLI plumbing.

Covers:
    * ``build_training_datamodule`` dispatches correctly for both ``raw_text``
      and ``llm_generated`` sources.
    * The returned metadata is JSON-serialisable and captures enough context
      to reproduce the run (teacher model_id, generation seed, prompt knobs).
    * The shared ``scripts/_shared/cli`` helpers apply all training-data CLI
      overrides onto a ``Config`` without touching unrelated fields.
    * LLM generation is seeded per-batch so a second build produces identical
      ``log_x`` tensors.
    * ``TrainingDataConfig`` validates its invariants at construction time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

# Make scripts/_shared importable for CLI tests (matches two_stage_train pattern).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _shared.cli import (  # noqa: E402
    add_training_data_args,
    apply_training_data_args,
    positive_int,
)

from aitchinson_flow.config import Config, TrainingDataConfig  # noqa: E402
from aitchinson_flow.llms import registry as lm_registry  # noqa: E402
from aitchinson_flow.training.data_sources import build_training_datamodule  # noqa: E402


class _StubLM:
    """Minimal ``CausalLMForInference`` that generates deterministic ids.

    Generation honours ``torch.initial_seed()`` so ``CausalLMTeacherDataset``'s
    per-batch seeding path yields reproducible token ids across independent
    builds.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, 100)

    def encode_text(
        self, text: str, *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, max_length, dtype=torch.long),
            torch.ones(1, max_length, dtype=torch.long),
        )

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
        gen = torch.Generator().manual_seed(torch.initial_seed() % (2**31))
        return torch.randint(0, 27, (batch_size, max_new_tokens), generator=gen)


@pytest.fixture(autouse=True)
def _register_stub_lm() -> None:
    """Register ``stub_lm`` once so build_lm(stub_lm, ...) works."""
    if "stub_lm" not in lm_registry.registered_lm_keys():
        lm_registry.register("stub_lm")(lambda _cfg: _StubLM())


def _base_smoke_config() -> Config:
    cfg = Config()
    cfg.training.device = torch.device("cpu")
    cfg.training.B = 2
    cfg.dataset.L = 8
    cfg.dataset.K = 27
    return cfg


class TestTrainingDataConfigValidation:
    def test_rejects_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="source"):
            TrainingDataConfig(source="bogus")

    def test_rejects_unknown_raw_dataset(self) -> None:
        with pytest.raises(ValueError, match="raw_dataset"):
            TrainingDataConfig(raw_dataset="bogus")

    def test_rejects_zero_top_p(self) -> None:
        with pytest.raises(ValueError, match="generation_top_p"):
            TrainingDataConfig(generation_top_p=0.0)

    def test_rejects_negative_temperature(self) -> None:
        with pytest.raises(ValueError, match="generation_temperature"):
            TrainingDataConfig(generation_temperature=-0.1)

    def test_rejects_non_positive_n_batches(self) -> None:
        with pytest.raises(ValueError, match="n_batches"):
            TrainingDataConfig(n_batches=0)

    def test_rejects_unknown_llm_feature_mode(self) -> None:
        with pytest.raises(ValueError, match="llm_feature_mode"):
            TrainingDataConfig(llm_feature_mode="bogus")


class TestLLMGeneratedDispatch:
    def test_produces_valid_and_invalid_batches(self) -> None:
        cfg = _base_smoke_config()
        cfg.training_data.source = "llm_generated"
        cfg.training_data.lm_key = "stub_lm"
        cfg.training_data.n_batches = 2
        cfg.training_data.generation_seed = 11

        dm, meta = build_training_datamodule(cfg)
        batches = list(dm.train_dataloader())

        assert len(batches) == 2
        assert {"log_x", "log_x_invalid"}.issubset(batches[0].keys())
        assert batches[0]["log_x"].shape == (cfg.training.B, cfg.dataset.L, cfg.dataset.K - 1)
        assert batches[0]["log_x_invalid"].shape == batches[0]["log_x"].shape

        assert meta["source"] == "llm_generated"
        assert meta["lm_key"] == "stub_lm"
        assert meta["generation_seed"] == 11
        assert meta["batch_size"] == cfg.training.B
        assert meta["seq_length"] == cfg.dataset.L
        assert "teacher" in meta and meta["teacher"]["model_id"] == cfg.teacher.model_id
        # Must round-trip through JSON for manifest persistence.
        json.dumps(meta, default=str)

    def test_seeded_generation_is_reproducible(self) -> None:
        cfg = _base_smoke_config()
        cfg.training_data.source = "llm_generated"
        cfg.training_data.lm_key = "stub_lm"
        cfg.training_data.n_batches = 1
        cfg.training_data.generation_seed = 2026

        dm_a, _ = build_training_datamodule(cfg)
        dm_b, _ = build_training_datamodule(cfg)
        batch_a = next(iter(dm_a.train_dataloader()))
        batch_b = next(iter(dm_b.train_dataloader()))
        assert torch.equal(batch_a["log_x"], batch_b["log_x"])
        assert torch.equal(batch_a["log_x_invalid"], batch_b["log_x_invalid"])

    def test_unknown_source_raises(self) -> None:
        cfg = _base_smoke_config()
        # Bypass __post_init__ validation to construct an invalid config in-place.
        cfg.training_data.source = "does-not-exist"
        with pytest.raises(ValueError, match="source"):
            build_training_datamodule(cfg)

    def test_unknown_raw_dataset_raises(self) -> None:
        cfg = _base_smoke_config()
        cfg.training_data.source = "raw_text"
        cfg.training_data.raw_dataset = "does-not-exist"
        with pytest.raises(ValueError, match="raw_dataset"):
            build_training_datamodule(cfg)

    def test_token_probs_feature_mode_uses_transform_dim(self) -> None:
        cfg = _base_smoke_config()
        cfg.training_data.source = "llm_generated"
        cfg.training_data.lm_key = "stub_lm"
        cfg.training_data.llm_feature_mode = "token_probs"
        cfg.hf_dataset.transform_mode = "clr"
        cfg.training_data.n_batches = 1
        dm, meta = build_training_datamodule(cfg)
        batch = next(iter(dm.train_dataloader()))
        assert batch["log_x"].shape == (cfg.training.B, cfg.dataset.L, cfg.dataset.K)
        assert meta["llm_feature_mode"] == "token_probs"


class TestCLIApply:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        add_training_data_args(parser)
        return parser.parse_args(argv)

    def test_defaults_preserve_config(self) -> None:
        args = self._parse([])
        cfg = Config()
        before = (
            cfg.training_data.source,
            cfg.training_data.raw_dataset,
            cfg.raw_text_dataset.provider,
            cfg.raw_text_dataset.source_ref,
            cfg.raw_text_dataset.dataset_name,
            cfg.raw_text_dataset.split_train,
            cfg.raw_text_dataset.split_val,
            cfg.raw_text_dataset.split_test,
            cfg.raw_text_dataset.text_column,
            cfg.training_data.lm_key,
            cfg.training_data.n_batches,
            cfg.training_data.generation_seed,
            cfg.training_data.generation_temperature,
            cfg.training_data.generation_top_p,
            cfg.training_data.llm_feature_mode,
            cfg.training_data.use_text8_prompts,
            cfg.training_data.text8_prompt_length,
        )
        apply_training_data_args(cfg, args)
        after = (
            cfg.training_data.source,
            cfg.training_data.raw_dataset,
            cfg.raw_text_dataset.provider,
            cfg.raw_text_dataset.source_ref,
            cfg.raw_text_dataset.dataset_name,
            cfg.raw_text_dataset.split_train,
            cfg.raw_text_dataset.split_val,
            cfg.raw_text_dataset.split_test,
            cfg.raw_text_dataset.text_column,
            cfg.training_data.lm_key,
            cfg.training_data.n_batches,
            cfg.training_data.generation_seed,
            cfg.training_data.generation_temperature,
            cfg.training_data.generation_top_p,
            cfg.training_data.llm_feature_mode,
            cfg.training_data.use_text8_prompts,
            cfg.training_data.text8_prompt_length,
        )
        assert before == after

    def test_overrides_applied(self) -> None:
        args = self._parse(
            [
                "--training-data-source",
                "llm_generated",
                "--raw-provider",
                "manual",
                "--raw-source-ref",
                "./data/local_text8",
                "--raw-dataset-name",
                "subset-a",
                "--raw-split-train",
                "train",
                "--raw-split-val",
                "val",
                "--raw-split-test",
                "test",
                "--raw-text-column",
                "content",
                "--training-lm-key",
                "stub_lm",
                "--generated-batches",
                "7",
                "--generation-seed",
                "123",
                "--generation-temperature",
                "0.5",
                "--generation-top-p",
                "0.8",
                "--llm-feature-mode",
                "token_probs",
                "--use-text8-prompts",
                "--text8-prompt-length",
                "16",
            ]
        )
        cfg = Config()
        apply_training_data_args(cfg, args)
        assert cfg.training_data.source == "llm_generated"
        assert cfg.raw_text_dataset.provider == "manual"
        assert cfg.raw_text_dataset.source_ref == "./data/local_text8"
        assert cfg.raw_text_dataset.dataset_name == "subset-a"
        assert cfg.raw_text_dataset.split_train == "train"
        assert cfg.raw_text_dataset.split_val == "val"
        assert cfg.raw_text_dataset.split_test == "test"
        assert cfg.raw_text_dataset.text_column == "content"
        assert cfg.training_data.lm_key == "stub_lm"
        assert cfg.training_data.n_batches == 7
        assert cfg.training_data.generation_seed == 123
        assert cfg.training_data.generation_temperature == pytest.approx(0.5)
        assert cfg.training_data.generation_top_p == pytest.approx(0.8)
        assert cfg.training_data.llm_feature_mode == "token_probs"
        assert cfg.training_data.use_text8_prompts is True
        assert cfg.training_data.text8_prompt_length == 16

    def test_positive_int_rejects_zero(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_positive_int_rejects_non_int(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("abc")
