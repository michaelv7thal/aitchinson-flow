"""Layer 1: load Hugging Face datasets and optional capped splits (no tokenization / log_x)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aitchinson_flow.config import HFDatasetConfig, RawTextDatasetConfig

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset


TEXT8_DATASET_CANDIDATES: tuple[str, ...] = ("afmck/text8", "afm-intelligence/text8")


def _require_datasets() -> Any:
    try:
        import datasets  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Install the `datasets` package (e.g. pip install datasets) for hf_hub."
        ) from e
    return datasets


def load_hf_dataset_dict(cfg: HFDatasetConfig) -> Dataset | DatasetDict | IterableDataset:
    """Return a Dataset (or DatasetDict or IterableDataset) from path/name."""
    if not cfg.enabled:
        raise ValueError("HFDatasetConfig.enabled is False; nothing to load.")

    if not cfg.path:
        raise ValueError("HFDatasetConfig.path must be set when enabled.")

    datasets = _require_datasets()

    kwargs: dict[str, Any] = {
        "path": cfg.path,
        "name": cfg.name,
        "streaming": cfg.streaming,
        "trust_remote_code": cfg.trust_remote_code,
    }

    if cfg.revision is not None:
        kwargs["revision"] = cfg.revision

    raw = datasets.load_dataset(**kwargs)

    return raw


def infer_provider_from_source_ref(source_ref: str) -> str:
    """Best-effort source provider inference for legacy aliasing."""
    p = Path(source_ref)
    if source_ref.startswith("/") or source_ref.startswith("./") or source_ref.startswith("../"):
        return "manual"
    if p.exists():
        return "manual"
    return "huggingface"


def build_raw_text_load_kwargs(
    cfg: RawTextDatasetConfig,
    *,
    split: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": cfg.source_ref,
        "name": cfg.dataset_name,
        "streaming": cfg.streaming,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.revision is not None:
        kwargs["revision"] = cfg.revision
    if split is not None:
        kwargs["split"] = split
    resolved_cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir
    if resolved_cache_dir is not None:
        kwargs["cache_dir"] = resolved_cache_dir
    return kwargs


def load_raw_text_column(
    cfg: RawTextDatasetConfig,
    *,
    split: str,
    cache_dir: str | None = None,
    fallback_paths: tuple[str, ...] = (),
) -> str:
    """Load a split and concatenate a text column into a single corpus string."""
    datasets = _require_datasets()
    candidates = (cfg.source_ref, *fallback_paths)
    last_err: Exception | None = None
    for source_ref in candidates:
        try:
            local_cfg = RawTextDatasetConfig(
                provider=cfg.provider,
                source_ref=source_ref,
                dataset_name=cfg.dataset_name,
                revision=cfg.revision,
                split_train=cfg.split_train,
                split_val=cfg.split_val,
                split_test=cfg.split_test,
                text_column=cfg.text_column,
                cache_dir=cfg.cache_dir,
                trust_remote_code=cfg.trust_remote_code,
                streaming=cfg.streaming,
            )
            ds = datasets.load_dataset(
                **build_raw_text_load_kwargs(local_cfg, split=split, cache_dir=cache_dir)
            )
            if local_cfg.text_column not in ds.column_names:
                raise KeyError(
                    f"text column {local_cfg.text_column!r} missing in dataset columns "
                    f"{list(ds.column_names)!r}"
                )
            return " ".join(ds[local_cfg.text_column])
        except Exception as e:  # pragma: no cover - network/local IO dependent
            last_err = e
    raise RuntimeError(
        f"Unable to load text split={split!r} from source_ref={cfg.source_ref!r}"
    ) from last_err


def _get_split(raw: Any, split: str) -> Any:
    if isinstance(raw, dict) or hasattr(raw, "keys"):
        if split not in raw:
            available = list(raw.keys()) if hasattr(raw, "keys") else []
            raise KeyError(f"Split {split!r} not found. Available: {available}")
        return raw[split]

    # Single split was loaded directly
    return raw


def _try_get_split(raw: Any, split: str | None) -> Any | None:
    if split is None:
        return None
    try:
        return _get_split(raw, split)
    except KeyError:
        return None


def maybe_subsample(
    ds: Any,
    max_samples: int | None,
    *,
    seed: int,
    streaming: bool,
    shuffle_buffer_size: int,
) -> Any:
    if max_samples is None:
        return ds

    if max_samples <= 0:
        raise ValueError("max_samples must be positive when set.")

    if streaming:
        return ds.shuffle(seed=seed, buffer_size=max(shuffle_buffer_size, max_samples)).take(
            max_samples
        )

    n = min(max_samples, len(ds))

    return ds.shuffle(seed=seed).select(range(n))


def load_hf_splits(cfg: HFDatasetConfig) -> tuple[Any, Any | None, Any | None]:
    """Train, validation, and test (each optional by config or Hub availability).
    Subsampling: separate seeds per split (e.g. seed, seed+1, seed+2) so shuffles differ.
    """
    raw = load_hf_dataset_dict(cfg)
    train = _get_split(raw, cfg.split_train)
    train = maybe_subsample(
        train,
        cfg.max_samples_train,
        seed=cfg.shuffle_seed,
        streaming=cfg.streaming,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
    )
    val = _try_get_split(raw, cfg.split_val)
    if val is not None:
        val = maybe_subsample(
            val,
            cfg.max_samples_val,
            seed=cfg.shuffle_seed + 1,
            streaming=cfg.streaming,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
        )
    test = _try_get_split(raw, cfg.split_test)
    if test is not None:
        test = maybe_subsample(
            test,
            cfg.max_samples_test,
            seed=cfg.shuffle_seed + 2,
            streaming=cfg.streaming,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
        )
    return train, val, test
