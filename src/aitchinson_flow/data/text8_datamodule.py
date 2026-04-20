"""Char-level text8 datamodule with train/val/test splits + on-the-fly corruption."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from aitchinson_flow.config import Config
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.training.datamodule import DataModule


_ALPHABET = "abcdefghijklmnopqrstuvwxyz "
CHAR2ID: dict[str, int] = {c: i for i, c in enumerate(_ALPHABET)}
VOCAB_SIZE: int = len(_ALPHABET)  # 27


def _load_text8_chars(split: str, cache_dir: str | None = None) -> str:
    import datasets  # noqa: PLC0415

    candidates = ("afmck/text8", "afm-intelligence/text8")
    last_err: Exception | None = None
    for name in candidates:
        try:
            ds = datasets.load_dataset(name, split=split, cache_dir=cache_dir)
            return " ".join(ds["text"])
        except Exception as e:  # pragma: no cover - network-dependent
            last_err = e
    raise RuntimeError(f"could not load text8 from {candidates!r}") from last_err


def chunk_text8_to_ids(text: str, L: int) -> Tensor:
    """Filter to the 27-char vocab, map to ids, reshape into `(N, L)` windows."""
    ids = [CHAR2ID[c] for c in text if c in CHAR2ID]
    n = len(ids) // L
    if n == 0:
        raise ValueError(f"text8 corpus too short for L={L}")
    t = torch.tensor(ids[: n * L], dtype=torch.long)
    return t.view(n, L)


def _load_text8_splits(cache_dir: str | None, L: int) -> tuple[Tensor, Tensor, Tensor]:
    """Load native train/validation/test splits and chunk into windows."""
    train_text = _load_text8_chars("train", cache_dir=cache_dir)
    val_text = _load_text8_chars("validation", cache_dir=cache_dir)
    test_text = _load_text8_chars("test", cache_dir=cache_dir)

    train_windows = chunk_text8_to_ids(train_text, L)
    val_windows = chunk_text8_to_ids(val_text, L)
    test_windows = chunk_text8_to_ids(test_text, L)
    return train_windows, val_windows, test_windows


@dataclass(frozen=True)
class _Text8Splits:
    train: Tensor
    val: Tensor
    test: Tensor


class Text8WindowDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset of length-L char windows → `{log_x, token_ids}`.

    Honors ``cfg.hf_dataset.label_smoothing`` and
    ``cfg.hf_dataset.transform_mode`` so the ILR/CLR + label-smoothing
    ablations flip with config alone.
    """

    def __init__(
        self,
        windows: Tensor,
        *,
        K: int,
        eps: float,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> None:
        self._windows = windows
        self._K = K
        self._eps = eps
        self._label_smoothing = label_smoothing
        self._transform_mode = transform_mode

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ids = self._windows[idx]
        log_x = token_ids_to_features(
            ids,
            K=self._K,
            eps=self._eps,
            label_smoothing=self._label_smoothing,
            transform_mode=self._transform_mode,
        )
        return {"log_x": log_x, "token_ids": ids}


class _CorruptingCollate:
    """Collate window dicts and add `log_x_invalid`, `token_ids_invalid`."""

    def __init__(
        self,
        *,
        K: int,
        vocab_size: int,
        corrupt_rate: float,
        order_mix_rate: float,
        order_mix_prob: float,
        eps: float,
        seed: int,
        label_smoothing: float = 0.0,
        transform_mode: str = "ilr",
    ) -> None:
        self._K = K
        self._vocab = vocab_size
        self._rate = corrupt_rate
        self._order_mix_rate = order_mix_rate
        self._order_mix_prob = order_mix_prob
        self._eps = eps
        self._seed = seed
        self._label_smoothing = label_smoothing
        self._transform_mode = transform_mode
        self._n_calls = 0

    def _token_ids_to_logits(self, token_ids: Tensor) -> Tensor:
        bsz, seq_len = token_ids.shape
        logits = torch.full((bsz, seq_len, self._vocab), math.log(self._eps))
        logits.scatter_(dim=-1, index=token_ids.unsqueeze(-1), value=0.0)
        return logits

    def __call__(self, samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        from aitchinson_flow.data.corruption import build_invalid_batch  # noqa: PLC0415

        log_x = torch.stack([s["log_x"] for s in samples], dim=0)
        token_ids = torch.stack([s["token_ids"] for s in samples], dim=0)
        batch: dict[str, Tensor] = {"log_x": log_x, "token_ids": token_ids}

        # Keep benchmark logits in vocab-space even though features are ILR.
        batch["logits"] = self._token_ids_to_logits(token_ids)

        seed = self._seed + self._n_calls
        self._n_calls += 1
        build_invalid_batch(
            batch,
            K=self._K,
            corrupt_rate=self._rate,
            order_mix_rate=self._order_mix_rate,
            order_mix_prob=self._order_mix_prob,
            eps=self._eps,
            label_smoothing=self._label_smoothing,
            transform_mode=self._transform_mode,
            seed=seed,
        )
        return batch


class Text8DataModule(DataModule):
    """Char-level text8 with real train/val/test loaders and per-split corruption."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        tcfg = cfg.text8_dataset
        L = cfg.dataset.L
        K = cfg.dataset.K
        if K != VOCAB_SIZE:
            raise ValueError(f"Text8DataModule expects cfg.dataset.K == {VOCAB_SIZE}, got K={K}")

        train, val, test = _load_text8_splits(tcfg.cache_dir, L)

        if tcfg.max_train_windows is not None:
            train = train[: tcfg.max_train_windows]
        if tcfg.max_eval_windows is not None:
            val = val[: tcfg.max_eval_windows]
            test = test[: tcfg.max_eval_windows]

        self._splits = _Text8Splits(train=train, val=val, test=test)

        eps = cfg.hf_dataset.log_simplex_eps
        ls = cfg.hf_dataset.label_smoothing
        tm = cfg.hf_dataset.transform_mode
        self._train_ds = Text8WindowDataset(
            train, K=K, eps=eps, label_smoothing=ls, transform_mode=tm
        )
        self._val_ds = Text8WindowDataset(
            val, K=K, eps=eps, label_smoothing=ls, transform_mode=tm
        )
        self._test_ds = Text8WindowDataset(
            test, K=K, eps=eps, label_smoothing=ls, transform_mode=tm
        )

        self._train_collate = _CorruptingCollate(
            K=K,
            vocab_size=K,
            corrupt_rate=tcfg.train_corrupt_rate,
            order_mix_rate=tcfg.train_order_mix_rate,
            order_mix_prob=tcfg.order_mix_prob,
            eps=eps,
            seed=tcfg.corruption_seed,
            label_smoothing=ls,
            transform_mode=tm,
        )

        self._eval_collate = _CorruptingCollate(
            K=K,
            vocab_size=K,
            corrupt_rate=tcfg.eval_corrupt_rate,
            order_mix_rate=tcfg.eval_order_mix_rate,
            order_mix_prob=tcfg.order_mix_prob,
            eps=eps,
            seed=tcfg.corruption_seed + 10_000,
            label_smoothing=ls,
            transform_mode=tm,
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
