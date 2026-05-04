from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import datasets
import torch

from aitchinson_flow.data.char_window_dataset import text_to_windows


@runtime_checkable
class HFTextDatasetConfig(Protocol):
    source_ref: str
    dataset_name: str | None
    streaming: bool
    trust_remote_code: bool
    revision: str | None
    cache_dir: str | None
    text_column: str
    split_train: str
    split_val: str | None
    split_test: str | None


def build_hf_load_kwargs(
    dataset_cfg: HFTextDatasetConfig,
    *,
    split: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": dataset_cfg.source_ref,
        "name": dataset_cfg.dataset_name,
        "streaming": dataset_cfg.streaming,
        "trust_remote_code": dataset_cfg.trust_remote_code,
    }
    if dataset_cfg.revision is not None:
        kwargs["revision"] = dataset_cfg.revision
    if split is not None:
        kwargs["split"] = split
    resolved_cache_dir = cache_dir if cache_dir is not None else dataset_cfg.cache_dir
    if resolved_cache_dir is not None:
        kwargs["cache_dir"] = resolved_cache_dir
    return kwargs


def load_text_column(
    dataset_cfg: HFTextDatasetConfig,
    *,
    split: str,
    cache_dir: str | None = None,
) -> str:
    try:
        ds = datasets.load_dataset(
            **build_hf_load_kwargs(dataset_cfg, split=split, cache_dir=cache_dir)
        )
        if dataset_cfg.text_column not in ds.column_names:
            raise KeyError(
                f"text column {dataset_cfg.text_column!r} missing in dataset columns "
                f"{list(ds.column_names)!r}"
            )
        return " ".join(ds[dataset_cfg.text_column])
    except Exception as e:
        raise RuntimeError(
            f"Unable to load text split={split!r} from source_ref={dataset_cfg.source_ref!r}"
        ) from e


def load_splits(
    dataset_cfg: HFTextDatasetConfig,
    L: int,
    *,
    cache_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_train = dataset_cfg.split_train
    split_val = dataset_cfg.split_val or "validation"
    split_test = dataset_cfg.split_test or "test"

    train = text_to_windows(load_text_column(dataset_cfg, split=split_train, cache_dir=cache_dir), L)
    val   = text_to_windows(load_text_column(dataset_cfg, split=split_val,   cache_dir=cache_dir), L)
    test  = text_to_windows(load_text_column(dataset_cfg, split=split_test,  cache_dir=cache_dir), L)

    return train, val, test
