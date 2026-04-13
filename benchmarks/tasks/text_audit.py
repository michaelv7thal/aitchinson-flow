from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.metrics.spilled_energy import (
    compute_spilled_energy_batch,
    sequence_anomaly_score,
)
from aitchinson_flow.models.base import AuditorModel
from aitchinson_flow.training.datamodule import DataModule
from aitchinson_flow.training.metrics import finalize_averages, running_average
from benchmarks.corruption import build_invalid_batch
from benchmarks.tasks.registry import register


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def _spilled_metrics(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    logits_invalid: torch.Tensor,
    token_ids_invalid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute spilled-energy metrics for valid and invalid sequences."""
    sp_valid = compute_spilled_energy_batch(logits, token_ids)  # (B, L-1)
    sp_invalid = compute_spilled_energy_batch(logits_invalid, token_ids_invalid)  # (B, L-1)

    valid_mean = sp_valid.mean()
    invalid_mean = sp_invalid.mean()
    anomaly_valid = -valid_mean
    anomaly_invalid = -invalid_mean

    return {
        "spilled_mean_valid": valid_mean,
        "spilled_mean_invalid": invalid_mean,
        "spilled_anomaly_valid": anomaly_valid,
        "spilled_anomaly_invalid": anomaly_invalid,
        "spilled_separation": anomaly_invalid - anomaly_valid,
        "spilled_token_mean": sp_valid.mean(dim=0).mean(),
        "spilled_token_std": sp_valid.std(dim=0).mean(),
    }


@register("text_audit")
class TextAuditTask:
    """Average ``audit(batch)`` metrics + spilled-energy baseline over the datamodule."""

    def run(self, model: nn.Module, datamodule: DataModule, cfg: Config) -> dict[str, float]:
        device = cfg.training.device
        model = model.to(device)
        model.eval()
        loader = datamodule.train_dataloader()
        auditor = cast(AuditorModel, model)
        bcfg = cfg.benchmark

        agg: dict[str, float] = {}
        counts: dict[str, int] = {}

        for batch_idx, batch in enumerate(loader):
            has_logits = "logits" in batch and "token_ids" in batch

            if has_logits and "log_x_invalid" not in batch:
                build_invalid_batch(
                    batch,
                    K=cfg.dataset.K,
                    corrupt_rate=bcfg.corrupt_rate,
                    eps=cfg.hf_dataset.log_simplex_eps,
                    seed=bcfg.corruption_seed + batch_idx,
                )

            batch_dev = _to_device(batch, device)
            out = auditor.audit(batch_dev)
            running_average(agg, counts, out)

            if has_logits and bcfg.compute_spilled_energy:
                sp = _spilled_metrics(
                    logits=batch["logits"],
                    token_ids=batch["token_ids"],
                    logits_invalid=batch["logits_invalid"],
                    token_ids_invalid=batch["token_ids_invalid"],
                )
                running_average(agg, counts, sp)

        return finalize_averages(agg, counts)
