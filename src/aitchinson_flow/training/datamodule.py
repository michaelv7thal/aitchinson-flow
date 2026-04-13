from __future__ import annotations

from typing import Any, Protocol

from torch.utils.data import DataLoader


class DataModule(Protocol):
    """Anything that supplies PyTorch loaders with your batch type (`Any`)."""

    def train_dataloader(self) -> DataLoader[Any]: ...

    def val_dataloader(self) -> DataLoader[Any] | None: ...

    def test_dataloader(self) -> DataLoader[Any] | None: ...
