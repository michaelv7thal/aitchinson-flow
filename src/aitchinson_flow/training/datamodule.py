from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch.utils.data import DataLoader


@runtime_checkable
class DataModule(Protocol):
    """Anything that supplies PyTorch loaders with your batch type (`Any`).

    Optional ``num_train_samples()`` reports the size of the training set so
    the trainer can normalize the KL term of variational models by ``N``
    rather than the batch size. Implementations that cannot report it
    (streaming datasets, iterable datasets with unknown length) may return
    ``None``; the trainer falls back to ``len(train_loader.dataset)``.
    """

    def train_dataloader(self) -> DataLoader[Any]: ...

    def val_dataloader(self) -> DataLoader[Any] | None: ...

    def test_dataloader(self) -> DataLoader[Any] | None: ...

    def num_train_samples(self) -> int | None:  # pragma: no cover - default impl
        """Size of the training set, or ``None`` if unknown / streaming."""
        return None
