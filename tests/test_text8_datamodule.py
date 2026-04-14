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
    split_windows,
)


def _make_corpus(n_chars: int = 5000, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    alphabet = list(CHAR2ID.keys())
    idx = torch.randint(0, len(alphabet), (n_chars,), generator=g).tolist()
    return "".join(alphabet[i] for i in idx)


class TestChunkAndSplit:
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

    def test_split_deterministic(self) -> None:
        w = torch.arange(1000 * 20).view(1000, 20)
        tr1, va1, te1 = split_windows(w, (0.8, 0.1, 0.1), seed=42)
        tr2, va2, te2 = split_windows(w, (0.8, 0.1, 0.1), seed=42)
        torch.testing.assert_close(tr1, tr2)
        torch.testing.assert_close(va1, va2)
        torch.testing.assert_close(te1, te2)

    def test_split_sizes_and_disjoint(self) -> None:
        w = torch.arange(1000 * 20).view(1000, 20)
        tr, va, te = split_windows(w, (0.8, 0.1, 0.1), seed=0)
        assert tr.shape[0] == 800
        assert va.shape[0] == 100
        assert te.shape[0] == 100
        # Disjointness: compare first column as id (since windows are arange)
        tr_ids = {int(r[0]) for r in tr}
        va_ids = {int(r[0]) for r in va}
        te_ids = {int(r[0]) for r in te}
        assert tr_ids.isdisjoint(va_ids)
        assert tr_ids.isdisjoint(te_ids)
        assert va_ids.isdisjoint(te_ids)


def _make_cfg(L: int = 20, B: int = 8) -> Config:
    cfg = Config()
    cfg.dataset.K = VOCAB_SIZE
    cfg.dataset.L = L
    cfg.training.B = B
    cfg.training.device = torch.device("cpu")
    cfg.text8_dataset.split_ratios = (0.7, 0.15, 0.15)
    cfg.text8_dataset.split_seed = 1
    cfg.text8_dataset.train_corrupt_rate = 0.15
    cfg.text8_dataset.eval_corrupt_rate = 0.4
    cfg.text8_dataset.max_train_windows = 64
    cfg.text8_dataset.max_eval_windows = 16
    return cfg


class TestText8DataModule:
    def _patched(self) -> Text8DataModule:
        cfg = _make_cfg(L=16, B=4)
        fake_text = _make_corpus(n_chars=20_000, seed=7)
        with patch(
            "aitchinson_flow.data.text8_datamodule._load_text8_chars",
            return_value=fake_text,
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
        B, L, K = batch["log_x"].shape
        assert L == 16
        assert K == VOCAB_SIZE
        assert batch["token_ids"].shape == (B, L)
        assert batch["logits"].shape == (B, L, K)
        assert batch["log_x_invalid"].shape == (B, L, K)

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
