"""DataModule that serves a pre-cached wiki auditor dataset.

Used only when ``cfg.auditor.enabled = True`` (Phase F). Deliberately
minimal: a single deterministic train/val split of the cache file, no
text8-style augmentation, no probe-able ``splits`` attribute (so the
runner's ``_unigram_kl_probe`` correctly skips this datamodule).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from aitchinson_flow.config import AuditorConfig
from aitchinson_flow.data.wiki import WikiAuditorDataset, load_wiki_cache


class WikiAuditorDataModule:
    def __init__(self, auditor_cfg: AuditorConfig, *, num_workers: int = 0) -> None:
        cache = load_wiki_cache(auditor_cfg.cache_path)
        full = WikiAuditorDataset(cache)
        n = len(full)
        n_train = max(1, int(auditor_cfg.train_frac * n))
        idx = list(range(n))
        self._train_ds = Subset(full, idx[:n_train])
        self._val_ds = Subset(full, idx[n_train:]) if n_train < n else None
        self._batch_size = auditor_cfg.batch_size
        self._num_workers = num_workers
        self._meta = {
            "lm": cache.get("lm"),
            "L": cache.get("L"),
            "K": cache.get("K"),
            "V": cache.get("V"),
            "H": cache.get("H"),
            "n": cache.get("n"),
            "corrupt_rate": cache.get("corrupt_rate"),
        }

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader[Any] | None:
        if self._val_ds is None:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader[Any] | None:
        return None
