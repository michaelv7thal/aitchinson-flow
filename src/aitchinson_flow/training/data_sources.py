from __future__ import annotations

from typing import Any

from aitchinson_flow.config import Config
from aitchinson_flow.training import DataModule


def build_training_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    if getattr(cfg, "auditor", None) is not None and cfg.auditor.enabled:
        return _build_wiki_auditor_datamodule(cfg)
    return _build_raw_text_datamodule(cfg)


def _build_wiki_auditor_datamodule(
    cfg: Config,
) -> tuple[DataModule, dict[str, Any]]:
    from aitchinson_flow.data.wiki_auditor_datamodule import WikiAuditorDataModule

    dm = WikiAuditorDataModule(
        auditor_cfg=cfg.auditor, num_workers=cfg.loader_settings.num_workers
    )
    meta = {"source": "wiki_auditor", **dm.meta}
    return dm, meta


def _build_raw_text_datamodule(cfg: Config) -> tuple[DataModule, dict[str, Any]]:
    from aitchinson_flow.data import Text8DataModule  # deferred to break import cycle

    dm = Text8DataModule(
        dataset_cfg=cfg.text8_dataset,
        transform_cfg=cfg.transformation,
        loader_settings=cfg.loader_settings,
    )
    return dm, {
        "source": "raw_text",
        "raw_dataset": "text8",
        "raw_provider": cfg.text8_dataset.provider,
        "raw_source_ref": cfg.text8_dataset.source_ref,
        "raw_dataset_name": cfg.text8_dataset.dataset_name,
        "raw_split_train": cfg.text8_dataset.split_train,
        "raw_split_val": cfg.text8_dataset.split_val,
        "raw_split_test": cfg.text8_dataset.split_test,
        "seq_length": cfg.text8_dataset.L,
        "vocab_size": cfg.text8_dataset.K,
        "corrupt_rate": cfg.text8_dataset.train_corrupt_rate,
        "order_mix_rate": cfg.text8_dataset.train_order_mix_rate,
        "order_mix_prob": cfg.text8_dataset.order_mix_prob,
        "corruption_seed": cfg.text8_dataset.corruption_seed,
    }
