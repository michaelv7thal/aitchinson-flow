from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch.utils.data import DataLoader


@runtime_checkable
class DataModule(Protocol):
    """Anything that supplies PyTorch loaders with your batch type (`Any`)."""

    def train_dataloader(self) -> DataLoader: ...

    def val_dataloader(self) -> DataLoader | None: ...

    def test_dataloader(self) -> DataLoader | None: ...
