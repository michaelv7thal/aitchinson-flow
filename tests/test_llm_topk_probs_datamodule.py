"""Smoke tests for LLMTopKProbsDataModule (Component 2: top-K probability path)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from aitchinson_flow.config import Config, LLMTopKProbsConfig
from aitchinson_flow.llms import registry as lm_registry
from aitchinson_flow.training.data_sources import build_training_datamodule


class _StubLMForTopK:
    """Stub LM that returns deterministic logits so clean vs corrupt inputs differ."""

    def __init__(self, vocab_size: int = 64, embed_dim: int = 8) -> None:
        self._vocab = vocab_size
        self._embed_dim = embed_dim
        torch.manual_seed(42)
        # Each token id maps to a fixed logit vector so tokenization affects output.
        self._logit_table = torch.randn(vocab_size, vocab_size)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def vocab_size(self) -> int:
        return self._vocab

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(False):
            table = torch.randn(self._vocab, self._embed_dim)
            return torch.nn.functional.embedding(token_ids.long(), table).clone()

    @torch.inference_mode()
    def forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Logits depend on token id so clean vs corrupted inputs differ.
        B, L = input_ids.shape
        ids_clamped = input_ids.long().clamp(0, self._vocab - 1)
        logits = self._logit_table[ids_clamped.reshape(-1)].reshape(B, L, self._vocab)
        return logits.clone()

    def encode_text(
        self, text: str, *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = abs(hash(text)) % (2**30)
        gen = torch.Generator().manual_seed(h)
        ids = torch.randint(0, self._vocab, (1, max_length), generator=gen)
        mask = torch.ones(1, max_length, dtype=torch.long)
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
        return torch.zeros(batch_size, max_new_tokens, dtype=torch.long)

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        return [""] * ids.shape[0]


@pytest.fixture(autouse=True)
def _register_stub() -> None:
    if "stub_topk_lm" not in lm_registry.registered_lm_keys():
        lm_registry.register("stub_topk_lm")(lambda _cfg: _StubLMForTopK())


def _fake_text8_splits(
    cfg: Config, cache_dir: str | None, L: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _chunk(n_windows: int, seed: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(0, 27, (n_windows, L), generator=gen)

    return _chunk(32, 10), _chunk(8, 20), _chunk(8, 30)


def _make_cfg(K: int = 8, L: int = 6, B: int = 2) -> Config:
    cfg = Config()
    cfg.training.device = torch.device("cpu")
    cfg.training.B = B
    cfg.dataset.K = K
    cfg.dataset.L = L
    cfg.training_data.source = "llm_topk_probs"
    cfg.llm_topk_probs = LLMTopKProbsConfig(
        lm_key="stub_topk_lm",
        char_window_length=32,
        corrupt_rate=0.4,
        generation_seed=99,
    )
    cfg.text8_dataset.max_train_windows = 8
    cfg.text8_dataset.max_eval_windows = 4
    return cfg


class TestLLMTopKProbsDispatch:
    def test_build_emits_log_x(self) -> None:
        cfg = _make_cfg()
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, meta = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        assert "log_x" in batch, "batch must contain log_x"
        assert "log_x_invalid" in batch, "batch must contain log_x_invalid"

    def test_log_x_shape(self) -> None:
        K, L, B = 8, 6, 2
        cfg = _make_cfg(K=K, L=L, B=B)
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        # ILR reduces last dim from K to K-1
        assert batch["log_x"].shape == (B, L, K - 1), batch["log_x"].shape
        assert batch["log_x_invalid"].shape == (B, L, K - 1), batch["log_x_invalid"].shape

    def test_values_are_finite(self) -> None:
        cfg = _make_cfg()
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        assert torch.isfinite(batch["log_x"]).all(), "log_x contains non-finite values"
        assert torch.isfinite(batch["log_x_invalid"]).all(), "log_x_invalid contains non-finite values"

    def test_clean_and_corrupt_differ(self) -> None:
        cfg = _make_cfg()
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        assert not torch.allclose(batch["log_x"], batch["log_x_invalid"]), (
            "clean and corrupted top-K ILR features should differ"
        )

    def test_autograd_safe(self) -> None:
        cfg = _make_cfg()
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        assert not batch["log_x"].is_inference(), "log_x must not be an inference tensor"
        assert not batch["log_x_invalid"].is_inference()
        K = cfg.dataset.K
        net = torch.nn.Linear(K - 1, 2)
        loss = net(batch["log_x"]).sum() + net(batch["log_x_invalid"]).sum()
        loss.backward()
        assert net.weight.grad is not None
        assert torch.isfinite(net.weight.grad).all()

    def test_metadata(self) -> None:
        cfg = _make_cfg(K=10)
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            _, meta = build_training_datamodule(cfg)

        assert meta["source"] == "llm_topk_probs"
        assert meta["lm_key"] == "stub_topk_lm"
        assert meta["top_k"] == 10
        assert meta["renormalize"] is True
        assert meta["corrupt_rate"] == pytest.approx(0.4)

    def test_val_dataloader_shape(self) -> None:
        K, L, B = 8, 6, 2
        cfg = _make_cfg(K=K, L=L, B=B)
        with patch(
            "aitchinson_flow.data.llm_topk_probs_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            val_batch = next(iter(dm.val_dataloader()))

        assert val_batch["log_x"].shape[-1] == K - 1
        assert val_batch["log_x_invalid"].shape[-1] == K - 1


class TestLLMTopKProbsConfigValidation:
    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="raw_text_backend"):
            LLMTopKProbsConfig(raw_text_backend="bogus")

    def test_rejects_zero_window_length(self) -> None:
        with pytest.raises(ValueError, match="char_window_length"):
            LLMTopKProbsConfig(char_window_length=0)

    def test_rejects_corrupt_rate_above_one(self) -> None:
        with pytest.raises(ValueError, match="corrupt_rate"):
            LLMTopKProbsConfig(corrupt_rate=1.5)

    def test_rejects_unknown_source(self) -> None:
        from aitchinson_flow.config import TrainingDataConfig

        with pytest.raises(ValueError, match="llm_topk_probs"):
            TrainingDataConfig(source="bogus_source_xyz")

    def test_accepts_valid_config(self) -> None:
        cfg = LLMTopKProbsConfig(
            lm_key="hf_causal",
            renormalize=True,
            char_window_length=128,
            corrupt_rate=0.3,
            generation_seed=7,
        )
        assert cfg.lm_key == "hf_causal"
        assert cfg.renormalize is True
