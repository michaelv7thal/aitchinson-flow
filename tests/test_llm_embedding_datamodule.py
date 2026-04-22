"""Tests for Path B: LLMEmbeddingDataModule + frozen-embedding → simplex pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from aitchinson_flow.config import Config, LLMEmbeddingDatasetConfig
from aitchinson_flow.llms import registry as lm_registry
from aitchinson_flow.training.data_sources import build_training_datamodule


class _DeterministicStubLM:
    """Deterministic stub LM whose input embeddings depend on token id so
    clean vs corrupted inputs produce different embedding batches.
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 12) -> None:
        self._vocab = vocab_size
        self._embed_dim = embed_dim
        torch.manual_seed(0)
        self._embed_table = torch.randn(vocab_size, embed_dim)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Match the real HF impl: return a normal (autograd-safe) tensor even
        # when the call originates from within ``torch.inference_mode``.
        with torch.inference_mode(False):
            return torch.nn.functional.embedding(token_ids.long(), self._embed_table).clone()

    @torch.inference_mode()
    def forward_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise AssertionError(
            "forward_logits must not be called by Path B — the datamodule should "
            "use embed_tokens instead."
        )

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
    if "stub_embedding_lm" not in lm_registry.registered_lm_keys():
        lm_registry.register("stub_embedding_lm")(lambda _cfg: _DeterministicStubLM())


def _fake_text8_splits(
    cfg: Config, cache_dir: str | None, L: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    cfg.llm_embedding_dataset = LLMEmbeddingDatasetConfig(
        lm_key="stub_embedding_lm",
        char_window_length=32,
        corrupt_rate=0.5,
        generation_seed=7,
    )
    cfg.text8_dataset.max_train_windows = 8
    cfg.text8_dataset.max_eval_windows = 8
    return cfg


class TestLLMEmbeddingDispatch:
    def test_build_via_data_sources_emits_embeddings(self) -> None:
        cfg = _make_path_b_cfg()
        with patch(
            "aitchinson_flow.data.llm_embedding_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, meta = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        expected_keys = {"embeddings", "embeddings_invalid", "token_ids", "token_ids_invalid"}
        assert expected_keys.issubset(batch)
        assert "log_x" not in batch, "Path B now emits embeddings, not log_x"

        d_embed = 12  # matches the stub
        assert batch["embeddings"].shape == (cfg.training.B, cfg.dataset.L, d_embed)
        assert batch["embeddings_invalid"].shape == batch["embeddings"].shape
        assert torch.isfinite(batch["embeddings"]).all()

        # llm_embed_dim is surfaced onto the config and exposed by the DM.
        assert cfg.llm_embedding_dataset.llm_embed_dim == d_embed
        assert dm.llm_embed_dim == d_embed

        assert meta["source"] == "llm_topk"
        assert meta["lm_key"] == "stub_embedding_lm"
        assert meta["simplex_dim"] == cfg.dataset.K
        assert meta["llm_embed_dim"] == d_embed
        assert meta["char_window_length"] == 32
        assert meta["corrupt_rate"] == pytest.approx(0.5)

    def test_clean_and_corrupt_embeddings_differ(self) -> None:
        cfg = _make_path_b_cfg()
        with patch(
            "aitchinson_flow.data.llm_embedding_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))
        assert not torch.allclose(batch["embeddings"], batch["embeddings_invalid"])

    def test_batch_tensors_support_autograd(self) -> None:
        """Regression: tensors emitted by Path B must not be inference tensors.

        Stage 1 backprops through the learned projection, which requires the
        incoming ``embeddings`` tensor to be autograd-safe.
        """
        cfg = _make_path_b_cfg()
        with patch(
            "aitchinson_flow.data.llm_embedding_datamodule._load_text8_splits_cfg",
            side_effect=_fake_text8_splits,
        ):
            dm, _ = build_training_datamodule(cfg)
            batch = next(iter(dm.train_dataloader()))

        assert not batch["embeddings"].is_inference()
        assert not batch["embeddings_invalid"].is_inference()

        d_embed = batch["embeddings"].shape[-1]
        net = torch.nn.Linear(d_embed, 3)
        loss = net(batch["embeddings"]).sum() + net(batch["embeddings_invalid"]).sum()
        loss.backward()
        assert net.weight.grad is not None
        assert torch.isfinite(net.weight.grad).all()

    def test_same_token_ids_produce_same_embedding(self) -> None:
        """Key property versus the old sorted-top-K path: determinism by token id.

        Feed a hand-crafted id tensor with known duplicate positions straight
        through the stub's embed_tokens and assert the duplicates map to the
        same vector. The batch collate is exercised by the other tests; this
        one pins the deterministic-by-id contract.
        """
        stub = _DeterministicStubLM()
        # Duplicate id at positions (0, 3) and (1, 4) within a single sequence.
        ids = torch.tensor([[7, 42, 3, 7, 42, 11]])
        emb = stub.embed_tokens(ids)
        assert torch.allclose(emb[0, 0], emb[0, 3])
        assert torch.allclose(emb[0, 1], emb[0, 4])
        # And different ids get different embeddings (cheap sanity check).
        assert not torch.allclose(emb[0, 0], emb[0, 1])

    def test_rejects_non_text8_backend(self) -> None:
        cfg = _make_path_b_cfg()
        cfg.llm_embedding_dataset = LLMEmbeddingDatasetConfig(
            lm_key="stub_embedding_lm", raw_text_backend="hf"
        )
        with (
            patch(
                "aitchinson_flow.data.llm_embedding_datamodule._load_text8_splits_cfg",
                side_effect=_fake_text8_splits,
            ),
            pytest.raises(NotImplementedError, match="raw_text_backend"),
        ):
            build_training_datamodule(cfg)


class TestLLMEmbeddingDatasetConfigValidation:
    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="raw_text_backend"):
            LLMEmbeddingDatasetConfig(raw_text_backend="bogus")

    def test_rejects_non_positive_window(self) -> None:
        with pytest.raises(ValueError, match="char_window_length"):
            LLMEmbeddingDatasetConfig(char_window_length=0)

    def test_rejects_out_of_range_corrupt_rate(self) -> None:
        with pytest.raises(ValueError, match="corrupt_rate"):
            LLMEmbeddingDatasetConfig(corrupt_rate=1.5)

    def test_rejects_invalid_llm_embed_dim(self) -> None:
        with pytest.raises(ValueError, match="llm_embed_dim"):
            LLMEmbeddingDatasetConfig(llm_embed_dim=0)
