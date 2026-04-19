"""Wrap HF map-style or iterable splits; apply row transform (Layer 2)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class HFEMapRowsDataset(Dataset[dict[str, torch.Tensor]]):
    """HF map-style dataset with row transform (Layer 2)."""

    def __init__(
        self, hf_split: Any, transform: Callable[[dict[str, Any]], dict[str, torch.Tensor]]
    ) -> None:
        self._hf = hf_split
        self._transform = transform

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._hf[idx]
        return self._transform(row)


class HFIterableRowsDataset(IterableDataset[dict[str, torch.Tensor]]):
    """For ``streaming=True`` HF datasets.

    With ``num_workers > 0`` each worker processes a disjoint shard of the
    underlying HF iterable (via ``skip``/``step`` over global worker index) so
    that every sample is yielded exactly once per epoch.
    """

    def __init__(
        self, hf_split: Any, transform: Callable[[dict[str, Any]], dict[str, torch.Tensor]]
    ) -> None:
        self._hf = hf_split
        self._transform = transform

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is None:
            # Single-process: yield everything.
            it = self._hf
        else:
            # Multi-worker: each worker takes every nth row starting at its id.
            it = (row for i, row in enumerate(self._hf) if i % worker.num_workers == worker.id)
        for row in it:
            yield self._transform(row)
