"""``DataModule`` wiring Layer 1 loads + Layer 2 transforms + ``DataLoader``."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from aitchinson_flow.config import Config
from aitchinson_flow.data.hf_hub import load_hf_splits
from aitchinson_flow.data.transforms.collate import collate_tensor_dict
from aitchinson_flow.data.transforms.hf_rows_dataset import HFIterableRowsDataset, HFEMapRowsDataset
from aitchinson_flow.data.transforms.discrete import make_discrete_row_transform
from aitchinson_flow.training.datamodule import DataModule


def _wrap_split(hf_split: Any, *, streaming: bool, transform: Any) -> Any:
    if hf_split is None:
        return None
    if streaming:
        return HFIterableRowsDataset(hf_split, transform)
    return HFEMapRowsDataset(hf_split, transform)


class HFDataModule(DataModule):
    def __init__(self, cfg: Config) -> None:
        if not cfg.hf_dataset.enabled:
            raise ValueError("HFDataModule requires cfg.hf_dataset.enabled")

        train_hf, val_hf, test_hf = load_hf_splits(cfg.hf_dataset)
        transform = make_discrete_row_transform(cfg)
        st = cfg.hf_dataset.streaming

        self._train = _wrap_split(train_hf, streaming=st, transform=transform)
        self._val = _wrap_split(val_hf, streaming=st, transform=transform)
        self._test = _wrap_split(test_hf, streaming=st, transform=transform)

        self._cfg = cfg

    def _loader(self, ds: Any, *, train: bool = False) -> DataLoader[Any]:
        if ds is None:
            raise ValueError("cannot build DataLoader for None dataset")
        stream = self._cfg.hf_dataset.streaming
        shuffle = train and not stream
        return DataLoader(
            ds,
            batch_size=self._cfg.training.B,
            shuffle=shuffle,
            num_workers=self._cfg.training.num_workers,
            collate_fn=collate_tensor_dict,
            pin_memory=self._cfg.training.device.type == "cuda",
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self._train, train=True)

    def val_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._val) if self._val is not None else None

    def test_dataloader(self) -> DataLoader[Any] | None:
        return self._loader(self._test) if self._test is not None else None
