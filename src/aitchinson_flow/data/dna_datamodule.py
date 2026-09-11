from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from aitchinson_flow.config import Text8DataConfig, TransformationConfig, LoaderSettings
from aitchinson_flow.data.char_window_dataset import CharWindowDataset
from aitchinson_flow.data.corrupting_collate import CorruptingCollate
from aitchinson_flow.training import DataModule

# DNA alphabet — must match scripts/prep_dna_windows.py (id<->base order).
DNA_ALPHABET = "ACGT"
DNA_K = len(DNA_ALPHABET)


@dataclass(frozen=True)
class _DNASplits:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


class DNADataModule(DataModule):
    """K=4 DNA datamodule — loads a pre-tokenized windows ``.pt`` and reuses the
    exact text8 simplex machinery (``CharWindowDataset`` + ``CorruptingCollate``)
    with ``K=4``.

    The only thing that differs from ``Text8DataModule`` is the source of the
    ``(N, L)`` integer windows tensor (a cached ``.pt`` produced by
    ``scripts/prep_dna_windows.py`` rather than an HF text download) and the
    absence of the text8 27/2-only alphabet validation. Everything downstream
    — CLR/label-smoothing/Dirichlet features, corruption, dataloaders — is
    byte-for-byte the same code path, so the EqM_OneHot comparison isolates K.
    """

    def __init__(
        self,
        dataset_cfg: Text8DataConfig,
        transform_cfg: TransformationConfig,
        loader_settings: LoaderSettings,
    ) -> None:
        if dataset_cfg.windows_path is None:
            raise ValueError("DNADataModule requires dataset_cfg.windows_path")
        path = Path(dataset_cfg.windows_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DNA windows cache not found: {path}. "
                f"Run: uv run python scripts/prep_dna_windows.py --out {path}"
            )
        saved = torch.load(path, weights_only=False)
        train, val, test = saved["train"], saved["val"], saved["test"]

        K = dataset_cfg.K
        meta_K = int(saved.get("meta", {}).get("K", K))
        if K != meta_K:
            raise ValueError(
                f"cfg K={K} disagrees with cached DNA windows K={meta_K} ({path})"
            )

        # Honour the same train/eval caps as text8 so split sizes match the
        # baseline (the cache is already capped, but a tighter cfg cap wins).
        if dataset_cfg.max_train_windows is not None:
            train = train[: dataset_cfg.max_train_windows]
        if dataset_cfg.max_eval_windows is not None:
            val = val[: dataset_cfg.max_eval_windows]
            test = test[: dataset_cfg.max_eval_windows]

        self._splits = _DNASplits(train=train, val=val, test=test)
        self._loader_settings = loader_settings

        ls = transform_cfg.label_smoothing
        lazy = bool(getattr(dataset_cfg, "lazy_features", False))
        self._train_ds = CharWindowDataset(train, K=K, label_smoothing=ls, lazy=lazy)
        self._val_ds = CharWindowDataset(val, K=K, label_smoothing=ls, lazy=lazy)
        self._test_ds = CharWindowDataset(test, K=K, label_smoothing=ls, lazy=lazy)

        dirichlet = bool(getattr(transform_cfg, "dirichlet_sampling", False))
        alpha_peak = float(getattr(transform_cfg, "dirichlet_alpha_peak", 50.0))
        alpha_base = float(getattr(transform_cfg, "dirichlet_alpha_base", 0.1))

        self._train_collate = CorruptingCollate(
            K=K,
            corrupt_rate=dataset_cfg.train_corrupt_rate,
            order_mix_rate=dataset_cfg.train_order_mix_rate,
            order_mix_prob=dataset_cfg.order_mix_prob,
            seed=dataset_cfg.corruption_seed,
            label_smoothing=ls,
            dirichlet_sampling=dirichlet,
            dirichlet_alpha_peak=alpha_peak,
            dirichlet_alpha_base=alpha_base,
        )
        self._eval_collate = CorruptingCollate(
            K=K,
            corrupt_rate=dataset_cfg.eval_corrupt_rate,
            order_mix_rate=dataset_cfg.eval_order_mix_rate,
            order_mix_prob=dataset_cfg.order_mix_prob,
            seed=dataset_cfg.corruption_seed + 10_000,
            label_smoothing=ls,
            dirichlet_sampling=dirichlet,
            dirichlet_alpha_peak=alpha_peak,
            dirichlet_alpha_base=alpha_base,
        )

    @property
    def splits(self) -> _DNASplits:
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
