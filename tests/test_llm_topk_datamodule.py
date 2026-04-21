"""Tests for Path B: LLMTopKDataModule + sorted_topk_logits_to_features helper."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from aitchinson_flow.config import Config, LLMTopKDatasetConfig
from aitchinson_flow.data.transforms.discrete import sorted_topk_logits_to_features
from aitchinson_flow.llms import registry as lm_registry
from aitchinson_flow.training.data_sources import build_training_datamodule


class _DeterministicStubLM:
    """Deterministic stub LM whose logits depend on input_ids so clean vs
    corrupted inputs produce different feature batches.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self._vocab = vocab_size

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L = input_ids.shape
        base = torch.arange(self._vocab, dtype=torch.float32).view(1, 1, -1).expand(B, L, -1)
        bump = input_ids.float().unsqueeze(-1) * 0.01
        return base + bump

    def encode_text(
        self, text: str, *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic char-sum seeding so different input strings produce
        # different "tokenizations".
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
        lm_registry.register("stub_topk_lm")(lambda _cfg: _DeterministicStubLM())


class TestSortedTopKLogitsToFeatures:
    def test_shape_ilr(self) -> None:
        logits = torch.randn(2, 5, 100)
        out = sorted_topk_logits_to_features(logits, K=8, transform_mode="ilr")
        assert out.shape == (2, 5, 7)
        assert torch.isfinite(out).all()

    def test_shape_clr(self) -> None:
        logits = torch.randn(3, 4, 50)
        out = sorted_topk_logits_to_features(logits, K=10, transform_mode="clr")
        assert out.shape == (3, 4, 10)
        assert torch.isfinite(out).all()

    def test_2d_input(self) -> None:
        logits = torch.randn(6, 40)
        out = sorted_topk_logits_to_features(logits, K=5, transform_mode="ilr")
        assert out.shape == (6, 4)

    def test_rejects_too_small_vocab(self) -> None:
        with pytest.raises(ValueError, match="last dim must be >= K"):
            sorted_topk_logits_to_features(torch.randn(2, 3, 4), K=8)

    def test_permuting_non_topk_logits_preserves_features(self) -> None:
        # Build logits where the top-3 are clearly separated from the rest.
        logits = torch.full((1, 2, 10), -5.0)
        logits[..., 0] = 3.0
        logits[..., 1] = 2.0
        logits[..., 2] = 1.0
        out_a = sorted_topk_logits_to_features(logits, K=3, transform_mode="ilr")

        # Permute the non-top-3 region; top-3 probabilities (after softmax +
        # renormalization) are identical, so ILR features should match.
        perm = logits.clone()
        tail = perm[..., 3:]
        perm[..., 3:] = tail.flip(dims=(-1,))
        out_b = sorted_topk_logits_to_features(perm, K=3, transform_mode="ilr")
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_sorting_invariance_of_topk_identity(self) -> None:
        # Putting the top-K in different vocab indices (but same probability
        # values) must yield the same features, because we sort by value.
        a = torch.tensor([[[5.0, 3.0, 1.0, -5.0, -5.0]]])
        b = torch.tensor([[[-5.0, -5.0, 1.0, 3.0, 5.0]]])
        out_a = sorted_topk_logits_to_features(a, K=3, transform_mode="ilr")
        out_b = sorted_topk_logits_to_features(b, K=3, transform_mode="ilr")
        assert torch.allclose(out_a, out_b, atol=1e-5)


def _fake_text8_splits(cfg: Config, cache_dir: str | None, L: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    def _chunk(n_windows: int, seed: int) -> torch.Tensor:
        gg = torch.Generator().manual_seed(seed)
        return torch.randint(0, 27, (n_windows, L), generator=gg)
    return _chunk(64, 1), _chunk(16, 2), _chunk(16, 3)


def _make_path_b_cfg() -> Config:
    cfg = Config()
    cfg.training.device = torch.device("cpu")
    cfg.training.B = 2
    cfg.dataset.L = 8
    cfg.dataset.K = 16
    cfg.training_data.source = "llm_topk"
    cfg.llm_topk_dataset = LLMTopKDatasetConfig(
        lm_key="stub_topk_lm",
        char_window_length=32,
        corrupt_rate=0.5,
        generation_seed=7,
    )
    cfg.text8_dataset.max_train_windows = 8
    cfg.text8_dataset.max_eval_windows = 8
    return cfg


class TestLLMTopKDispatch:
    def test_build_via_data_sources_and_shapes(self) -> None:
        cfg = _make_path_b_cfg()
        with patch(
            "aitchinson_flow.data.llm_topk_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, meta = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        assert {"log_x", "log_x_invalid", "token_ids", "token_ids_invalid"}.issubset(batch)
        assert batch["log_x"].shape == (cfg.training.B, cfg.dataset.L, cfg.dataset.K - 1)
        assert batch["log_x_invalid"].shape == batch["log_x"].shape
        assert torch.isfinite(batch["log_x"]).all()
        assert torch.isfinite(batch["log_x_invalid"]).all()

        assert meta["source"] == "llm_topk"
        assert meta["lm_key"] == "stub_topk_lm"
        assert meta["top_k"] == cfg.dataset.K
        assert meta["char_window_length"] == 32
        assert meta["corrupt_rate"] == pytest.approx(0.5)

    def test_clean_and_corrupt_features_differ(self) -> None:
        cfg = _make_path_b_cfg()
        with patch(
            "aitchinson_flow.data.llm_topk_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))
        assert not torch.allclose(batch["log_x"], batch["log_x_invalid"])

    def test_rejects_non_text8_backend(self) -> None:
        cfg = _make_path_b_cfg()
        cfg.llm_topk_dataset = LLMTopKDatasetConfig(
            lm_key="stub_topk_lm", raw_text_backend="hf"
        )
        with (
            patch(
                "aitchinson_flow.data.llm_topk_datamodule._load_text8_splits_cfg",
                side_effect=_fake_text8_splits,
            ),
            pytest.raises(NotImplementedError, match="raw_text_backend"),
        ):
            build_training_datamodule(cfg)


class TestLLMTopKDatasetConfigValidation:
    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="raw_text_backend"):
            LLMTopKDatasetConfig(raw_text_backend="bogus")

    def test_rejects_non_positive_window(self) -> None:
        with pytest.raises(ValueError, match="char_window_length"):
            LLMTopKDatasetConfig(char_window_length=0)

    def test_rejects_out_of_range_corrupt_rate(self) -> None:
        with pytest.raises(ValueError, match="corrupt_rate"):
            LLMTopKDatasetConfig(corrupt_rate=1.5)
