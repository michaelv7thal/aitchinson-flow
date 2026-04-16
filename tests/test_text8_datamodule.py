"""Tests for Text8DataModule — split integrity, shapes, collation."""

from __future__ import annotations

from unittest.mock import patch

import torch

from aitchinson_flow.config import Config
from aitchinson_flow.data.text8_datamodule import (
    CHAR2ID,
    VOCAB_SIZE,
    Text8DataModule,
    chunk_text8_to_ids,
)


def _make_corpus(n_chars: int = 5000, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    alphabet = list(CHAR2ID.keys())
    idx = torch.randint(0, len(alphabet), (n_chars,), generator=g).tolist()
    return "".join(alphabet[i] for i in idx)


class TestChunk:
    def test_chunk_shape_and_range(self) -> None:
        text = _make_corpus(1000)
        out = chunk_text8_to_ids(text, L=20)
        assert out.ndim == 2
        assert out.shape[1] == 20
        assert out.dtype == torch.long
        assert (out >= 0).all() and (out < VOCAB_SIZE).all()

    def test_chunk_filters_invalid_chars(self) -> None:
        out = chunk_text8_to_ids("abc!!! def???", L=3)
        flat = out.flatten().tolist()
        for v in flat:
            assert 0 <= v < VOCAB_SIZE

def _make_cfg(L: int = 20, B: int = 8) -> Config:
    cfg = Config()
    cfg.dataset.K = VOCAB_SIZE
    cfg.dataset.L = L
    cfg.training.B = B
    cfg.training.device = torch.device("cpu")
    cfg.text8_dataset.train_corrupt_rate = 0.15
    cfg.text8_dataset.eval_corrupt_rate = 0.4
    cfg.text8_dataset.max_train_windows = 64
    cfg.text8_dataset.max_eval_windows = 16
    return cfg


class TestText8DataModule:
    def _patched(self) -> Text8DataModule:
        cfg = _make_cfg(L=16, B=4)
        # Build distinct corpora per split so boundaries are preserved.
        fake_train = _make_corpus(n_chars=20_000, seed=7)
        fake_val = _make_corpus(n_chars=10_000, seed=8)
        fake_test = _make_corpus(n_chars=10_000, seed=9)

        with patch(
            "aitchinson_flow.data.text8_datamodule._load_text8_chars",
            side_effect=lambda split, cache_dir=None: {
                "train": fake_train,
                "validation": fake_val,
                "test": fake_test,
            }[split],
        ):
            return Text8DataModule(cfg)

    def test_loaders_exist(self) -> None:
        dm = self._patched()
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

    def test_train_batch_keys_and_shapes(self) -> None:
        dm = self._patched()
        batch = next(iter(dm.train_dataloader()))
        for k in ("log_x", "token_ids", "logits", "log_x_invalid", "token_ids_invalid"):
            assert k in batch, f"missing {k!r}"
        B, L, D = batch["log_x"].shape
        assert L == 16
        assert D == VOCAB_SIZE - 1
        assert batch["token_ids"].shape == (B, L)
        assert batch["logits"].shape == (B, L, VOCAB_SIZE)
        assert batch["log_x_invalid"].shape == (B, L, VOCAB_SIZE - 1)

    def test_invalid_differs_from_valid(self) -> None:
        dm = self._patched()
        batch = next(iter(dm.val_dataloader()))
        assert not torch.equal(batch["token_ids"], batch["token_ids_invalid"])

    def test_splits_disjoint(self) -> None:
        dm = self._patched()
        tr = {tuple(r.tolist()) for r in dm.splits.train}
        va = {tuple(r.tolist()) for r in dm.splits.val}
        te = {tuple(r.tolist()) for r in dm.splits.test}
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)
        # Native splits: each split should be a subset of the corresponding
        # chunked corpus used to construct it.
        assert len(tr) > 0
        assert len(va) > 0
        assert len(te) > 0
