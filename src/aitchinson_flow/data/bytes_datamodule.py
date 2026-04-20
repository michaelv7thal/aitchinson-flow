"""Byte-level text datamodule for Path B Phase 1 backbone training.

Loads a raw-text corpus via :func:`load_raw_text_column`, encodes each
character's UTF-8 bytes (stripping reserved role markers), and splits the
stream into fixed-length ``L`` windows. Shares the corruption pipeline
with :class:`Text8DataModule` so Stage 1 / Stage 2 training code needs no
special-case branching.

Designed for Phase 1 of the Path B auditor: the Stage 1 backbone is
retrained on byte windows with ``cfg.dataset.K == 256``. The existing
char-level (K=27) text8 path is unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.byte_vocab import RESERVED, VOCAB_SIZE
from aitchinson_flow.data.hf_hub import load_raw_text_column
from aitchinson_flow.data.text8_datamodule import (
    Text8WindowDataset,
    _CorruptingCollate,
    _Text8Splits,
)
from aitchinson_flow.training.datamodule import DataModule


def _text_to_byte_ids_stream(text: str) -> list[int]:
    """UTF-8 encode ``text`` and strip reserved role bytes from the stream."""
    raw = text.encode("utf-8", errors="replace")
    return [b for b in raw if b not in RESERVED]


def chunk_bytes_to_ids(text: str, L: int) -> Tensor:
    ids = _text_to_byte_ids_stream(text)
    n = len(ids) // L
    if n == 0:
        raise ValueError(f"byte corpus too short for L={L} (got {len(ids)} bytes)")
    t = torch.tensor(ids[: n * L], dtype=torch.long)
    return t.view(n, L)


def _load_byte_splits(cfg: Config) -> tuple[Tensor, Tensor, Tensor]:
    raw_cfg = cfg.raw_text_dataset
    train_text = load_raw_text_column(raw_cfg, split=raw_cfg.split_train)
    val_text = (
        load_raw_text_column(raw_cfg, split=raw_cfg.split_val)
        if raw_cfg.split_val
        else ""
    )
    test_text = (
        load_raw_text_column(raw_cfg, split=raw_cfg.split_test)
        if raw_cfg.split_test
        else ""
    )
    L = cfg.dataset.L
    train = chunk_bytes_to_ids(train_text, L)
    val = chunk_bytes_to_ids(val_text, L) if val_text else train[: max(1, train.shape[0] // 10)]
    test = (
        chunk_bytes_to_ids(test_text, L)
        if test_text
        else train[: max(1, train.shape[0] // 10)]
    )
    return train, val, test


class BytesDataModule(DataModule):
    """Raw-text corpus → UTF-8 byte windows of length ``L`` at ``K=256``.

    Training emits ``{log_x, token_ids, log_x_invalid, token_ids_invalid}``
    via the shared corruption collate, keeping Stage 1 / Stage 2 trainer
    code path-agnostic.
    """

    def __init__(self, cfg: Config) -> None:
        if cfg.dataset.K != VOCAB_SIZE:
            raise ValueError(
                f"BytesDataModule expects cfg.dataset.K == {VOCAB_SIZE} (byte-level); "
                f"got K={cfg.dataset.K}"
            )
        self._cfg = cfg
        L = cfg.dataset.L
        K = cfg.dataset.K
        tcfg = cfg.text8_dataset

        train, val, test = _load_byte_splits(cfg)
        if tcfg.max_train_windows is not None:
            train = train[: tcfg.max_train_windows]
        if tcfg.max_eval_windows is not None:
            val = val[: tcfg.max_eval_windows]
            test = test[: tcfg.max_eval_windows]
        self._splits = _Text8Splits(train=train, val=val, test=test)

        eps = cfg.hf_dataset.log_simplex_eps
        ls = cfg.hf_dataset.label_smoothing
        tm = cfg.hf_dataset.transform_mode
        self._train_ds = Text8WindowDataset(train, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        self._val_ds = Text8WindowDataset(val, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        self._test_ds = Text8WindowDataset(test, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)

        self._train_collate = _CorruptingCollate(
            K=K, vocab_size=K,
            corrupt_rate=tcfg.train_corrupt_rate,
            order_mix_rate=tcfg.train_order_mix_rate,
            order_mix_prob=tcfg.order_mix_prob,
            eps=eps, seed=tcfg.corruption_seed,
            label_smoothing=ls, transform_mode=tm,
        )
        self._eval_collate = _CorruptingCollate(
            K=K, vocab_size=K,
            corrupt_rate=tcfg.eval_corrupt_rate,
            order_mix_rate=tcfg.eval_order_mix_rate,
            order_mix_prob=tcfg.order_mix_prob,
            eps=eps, seed=tcfg.corruption_seed + 10_000,
            label_smoothing=ls, transform_mode=tm,
        )

    @property
    def splits(self) -> _Text8Splits:
        return self._splits

    def _loader(self, ds: Dataset[Any], *, shuffle: bool, collate: Any) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=self._cfg.training.B,
            shuffle=shuffle,
            num_workers=self._cfg.training.num_workers,
            collate_fn=collate,
            pin_memory=self._cfg.training.device.type == "cuda",
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self._train_ds, shuffle=True, collate=self._train_collate)

    def val_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._val_ds, shuffle=False, collate=self._eval_collate)

    def test_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._test_ds, shuffle=False, collate=self._eval_collate)

    def num_train_samples(self) -> int | None:
        return len(self._train_ds)
