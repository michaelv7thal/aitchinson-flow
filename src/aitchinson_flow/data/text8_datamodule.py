from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from aitchinson_flow.config import Text8DataConfig, TransformationConfig, LoaderSettings
from aitchinson_flow.data.char_window_dataset import CharWindowDataset, VOCAB_SIZE
from aitchinson_flow.data.corrupting_collate import CorruptingCollate
from aitchinson_flow.data.hf_text_loader import load_splits
from aitchinson_flow.training import DataModule


def _windows_cache_path(dataset_cfg: Text8DataConfig) -> Path | None:
    if dataset_cfg.cache_dir is None:
        return None
    key_data = {
        "source_ref": dataset_cfg.source_ref,
        "dataset_name": dataset_cfg.dataset_name,
        "L": dataset_cfg.L,
        "K": dataset_cfg.K,
        "max_train": dataset_cfg.max_train_windows,
        "max_eval": dataset_cfg.max_eval_windows,
        "split_train": dataset_cfg.split_train,
        "split_val": dataset_cfg.split_val,
        "split_test": dataset_cfg.split_test,
    }
    key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    return Path(dataset_cfg.cache_dir) / "windows" / f"{key}.pt"


def _load_or_compute_splits(
    dataset_cfg: Text8DataConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_path = _windows_cache_path(dataset_cfg)
    if cache_path is not None and cache_path.exists():
        saved = torch.load(cache_path, weights_only=True)
        return saved["train"], saved["val"], saved["test"]

    train, val, test = load_splits(dataset_cfg, dataset_cfg.L)

    if dataset_cfg.max_train_windows is not None:
        train = train[: dataset_cfg.max_train_windows]
    if dataset_cfg.max_eval_windows is not None:
        val = val[: dataset_cfg.max_eval_windows]
        test = test[: dataset_cfg.max_eval_windows]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train": train, "val": val, "test": test}, cache_path)

    return train, val, test


@dataclass(frozen=True)
class _Text8Splits:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


class Text8DataModule(DataModule):
    """Char-level text8 datamodule — thin orchestrator."""

    def __init__(
        self,
        dataset_cfg: Text8DataConfig,
        transform_cfg: TransformationConfig,
        loader_settings: LoaderSettings,
    ) -> None:
        K = dataset_cfg.K
        alphabet = getattr(dataset_cfg, "alphabet", "full")

        if alphabet == "full":
            if K != VOCAB_SIZE:
                raise ValueError(
                    f"Text8DataModule expects K == {VOCAB_SIZE} when alphabet='full', got K={K}"
                )
        elif alphabet == "binary":
            if K != 2:
                raise ValueError(
                    f"Text8DataModule expects K == 2 when alphabet='binary', got K={K}"
                )
        else:
            raise ValueError(f"Unknown alphabet={alphabet!r}; expected 'full' or 'binary'")

        # Always load the full K=27 windows from cache; remap to K=2 after
        # the load if alphabet=='binary'. This way the expensive splits
        # cache (one HF download + window slicing) is shared across both
        # alphabets — only the post-mapping changes.
        from dataclasses import replace as _replace
        load_cfg = _replace(dataset_cfg, K=VOCAB_SIZE) if alphabet == "binary" else dataset_cfg
        train, val, test = _load_or_compute_splits(load_cfg)
        if alphabet == "binary":
            from aitchinson_flow.data.text8_binary import windows_to_binary
            train = windows_to_binary(train)
            val = windows_to_binary(val)
            test = windows_to_binary(test)

        self._splits = _Text8Splits(train=train, val=val, test=test)
        self._loader_settings = loader_settings

        ls = transform_cfg.label_smoothing

        self._train_ds = CharWindowDataset(train, K=K, label_smoothing=ls)
        self._val_ds = CharWindowDataset(val, K=K, label_smoothing=ls)
        self._test_ds = CharWindowDataset(test, K=K, label_smoothing=ls)

        self._train_collate = CorruptingCollate(
            K=K,
            corrupt_rate=dataset_cfg.train_corrupt_rate,
            order_mix_rate=dataset_cfg.train_order_mix_rate,
            order_mix_prob=dataset_cfg.order_mix_prob,
            seed=dataset_cfg.corruption_seed,
            label_smoothing=ls,
        )
        self._eval_collate = CorruptingCollate(
            K=K,
            corrupt_rate=dataset_cfg.eval_corrupt_rate,
            order_mix_rate=dataset_cfg.eval_order_mix_rate,
            order_mix_prob=dataset_cfg.order_mix_prob,
            seed=dataset_cfg.corruption_seed + 10_000,
            label_smoothing=ls,
        )

    @property
    def splits(self) -> _Text8Splits:
        return self._splits

    def _loader(self, ds: Dataset[Any], *, shuffle: bool, collate: Any) -> DataLoader:
        s = self._loader_settings
        return DataLoader(
            ds,
            batch_size=s.batch_size,
            shuffle=shuffle,
            num_workers=s.num_workers,
            collate_fn=collate,
            pin_memory=(s.device_type.type == "cuda"),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self._train_ds, shuffle=True, collate=self._train_collate)

    def val_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._val_ds, shuffle=False, collate=self._eval_collate)

    def test_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._test_ds, shuffle=False, collate=self._eval_collate)
