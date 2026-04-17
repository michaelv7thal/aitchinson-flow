from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.metrics.spilled_energy import compute_spilled_energy_batch
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
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    sp_valid = compute_spilled_energy_batch(logits, token_ids)
    sp_invalid = compute_spilled_energy_batch(logits_invalid, token_ids_invalid)

    valid_mean = sp_valid.mean()
    invalid_mean = sp_invalid.mean()
    anomaly_valid = -valid_mean
    anomaly_invalid = -invalid_mean

    metrics = {
        "spilled_mean_valid": valid_mean,
        "spilled_mean_invalid": invalid_mean,
        "spilled_anomaly_valid": anomaly_valid,
        "spilled_anomaly_invalid": anomaly_invalid,
        "spilled_separation": anomaly_invalid - anomaly_valid,
        "spilled_token_mean": sp_valid.mean(dim=0).mean(),
        "spilled_token_std": sp_valid.std(dim=0).mean(),
    }
    return metrics, sp_valid, sp_invalid


def _safe_auroc(valid: np.ndarray, invalid: np.ndarray) -> float:
    """AUROC with valid=0, invalid=1; returns NaN if a class is empty."""
    if valid.size == 0 or invalid.size == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    labels = np.concatenate([np.zeros(valid.size), np.ones(invalid.size)])
    scores = np.concatenate([valid, invalid])
    if not np.isfinite(scores).all():
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return float(roc_auc_score(labels, scores))


def _auditor_score(
    model: nn.Module, log_x: torch.Tensor, *, ctx: torch.Tensor | None = None
) -> torch.Tensor | None:
    fn = getattr(model, "score_per_sample", None)
    if fn is None:
        return None
    try:
        out = fn(log_x, ctx=ctx) if ctx is not None else fn(log_x)
    except TypeError:
        out = fn(log_x)
    return out.detach().cpu()


def _auditor_sequence_uq(
    model: nn.Module, log_x: torch.Tensor, *, ctx: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    fn = getattr(model, "per_token_uq", None)
    if fn is None:
        return None

    if ctx is None:
        out = fn(log_x)
    else:
        out = fn(log_x, ctx=ctx)
    if not isinstance(out, tuple) or len(out) < 2:
        return None
    energy, variance = out[0], out[1]
    if not (torch.is_tensor(energy) and torch.is_tensor(variance)):
        return None
    return energy.detach().cpu(), variance.detach().cpu()


@register("text_audit")
class TextAuditTask:
    """Audit metrics + spilled-energy baseline + AUROC over a dataloader."""

    def _choose_loader(self, datamodule: DataModule) -> Any:
        test = datamodule.test_dataloader()
        if test is not None:
            return test
        val = datamodule.val_dataloader()
        if val is not None:
            return val
        return datamodule.train_dataloader()

    def run(self, model: nn.Module, datamodule: DataModule, cfg: Config) -> dict[str, Any]:
        device = cfg.training.device
        model = model.to(device)
        model.eval()
        loader = self._choose_loader(datamodule)
        auditor = cast(AuditorModel, model)
        bcfg = cfg.benchmark

        agg: dict[str, float] = {}
        counts: dict[str, int] = {}

        audit_valid: list[torch.Tensor] = []
        audit_invalid: list[torch.Tensor] = []
        spilled_valid: list[torch.Tensor] = []
        spilled_invalid: list[torch.Tensor] = []
        sp_valid_seqs: list[torch.Tensor] = []
        sp_invalid_seqs: list[torch.Tensor] = []
        audit_energy_valid_seqs: list[torch.Tensor] = []
        audit_energy_invalid_seqs: list[torch.Tensor] = []
        audit_var_valid_seqs: list[torch.Tensor] = []
        audit_var_invalid_seqs: list[torch.Tensor] = []

        for batch_idx, batch in enumerate(loader):
            has_logits = "logits" in batch and "token_ids" in batch

            if has_logits and "log_x_invalid" not in batch:
                build_invalid_batch(
                    batch,
                    K=cfg.dataset.K,
                    corrupt_rate=bcfg.corrupt_rate,
                    order_mix_rate=bcfg.order_mix_rate,
                    order_mix_prob=bcfg.order_mix_prob,
                    eps=cfg.hf_dataset.log_simplex_eps,
                    seed=bcfg.corruption_seed + batch_idx,
                )

            batch_dev = _to_device(batch, device)
            out = auditor.audit(batch_dev)
            running_average(agg, counts, out)

            v_score = _auditor_score(model, batch_dev["log_x"], ctx=batch_dev.get("ctx_1"))
            i_score = (
                _auditor_score(
                    model, batch_dev["log_x_invalid"], ctx=batch_dev.get("ctx_1_invalid")
                )
                if "log_x_invalid" in batch_dev
                else None
            )
            if v_score is not None:
                audit_valid.append(v_score)
            if i_score is not None:
                audit_invalid.append(i_score)

            v_seq = _auditor_sequence_uq(model, batch_dev["log_x"], ctx=batch_dev.get("ctx_1"))
            i_seq = (
                _auditor_sequence_uq(
                    model, batch_dev["log_x_invalid"], ctx=batch_dev.get("ctx_1_invalid")
                )
                if "log_x_invalid" in batch_dev
                else None
            )
            if v_seq is not None:
                e_v, var_v = v_seq
                audit_energy_valid_seqs.append(e_v)
                audit_var_valid_seqs.append(var_v)
            if i_seq is not None:
                e_i, var_i = i_seq
                audit_energy_invalid_seqs.append(e_i)
                audit_var_invalid_seqs.append(var_i)

            if has_logits and bcfg.compute_spilled_energy:
                sp_metrics, sp_v_seq, sp_i_seq = _spilled_metrics(
                    logits=batch["logits"],
                    token_ids=batch["token_ids"],
                    logits_invalid=batch["logits_invalid"],
                    token_ids_invalid=batch["token_ids_invalid"],
                )
                running_average(agg, counts, sp_metrics)
                spilled_valid.append(-sp_v_seq.mean(dim=1).detach().cpu())
                spilled_invalid.append(-sp_i_seq.mean(dim=1).detach().cpu())
                sp_valid_seqs.append(sp_v_seq.detach().cpu())
                sp_invalid_seqs.append(sp_i_seq.detach().cpu())

        result: dict[str, Any] = finalize_averages(agg, counts)

        def _cat(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        v_audit = _cat(audit_valid)
        i_audit = _cat(audit_invalid)
        v_spill = _cat(spilled_valid)
        i_spill = _cat(spilled_invalid)

        result["auroc_auditor"] = _safe_auroc(v_audit, i_audit)
        result["auroc_spilled"] = _safe_auroc(v_spill, i_spill)

        # Pearson r between GP variance and spilled energy scores
        if v_spill.size and v_audit.size and len(v_spill) == len(v_audit):
            result["pearson_r_valid"] = float(np.corrcoef(v_audit, v_spill)[0, 1])
        if i_spill.size and i_audit.size and len(i_spill) == len(i_audit):
            result["pearson_r_invalid"] = float(np.corrcoef(i_audit, i_spill)[0, 1])

        result["_scores"] = {
            "auditor_valid": v_audit,
            "auditor_invalid": i_audit,
            "spilled_valid": v_spill,
            "spilled_invalid": i_spill,
            "spilled_seq_valid": (
                torch.cat(sp_valid_seqs, dim=0).numpy() if sp_valid_seqs else None
            ),
            "spilled_seq_invalid": (
                torch.cat(sp_invalid_seqs, dim=0).numpy() if sp_invalid_seqs else None
            ),
            "auditor_energy_seq_valid": (
                torch.cat(audit_energy_valid_seqs, dim=0).numpy() if audit_energy_valid_seqs else None
            ),
            "auditor_energy_seq_invalid": (
                torch.cat(audit_energy_invalid_seqs, dim=0).numpy()
                if audit_energy_invalid_seqs
                else None
            ),
            "auditor_var_seq_valid": (
                torch.cat(audit_var_valid_seqs, dim=0).numpy() if audit_var_valid_seqs else None
            ),
            "auditor_var_seq_invalid": (
                torch.cat(audit_var_invalid_seqs, dim=0).numpy() if audit_var_invalid_seqs else None
            ),
        }

        # Corrupt-rate sweep (Experiment 3): re-run at multiple corruption levels
        sweep = cfg.benchmark.corrupt_rate_sweep
        if sweep:
            result["corrupt_rate_sweep"] = _run_corrupt_sweep(
                model=model,  # type: ignore[arg-type]
                datamodule=datamodule,
                cfg=cfg,
                rates=sweep,
            )

        return result


def _run_corrupt_sweep(
    model: nn.Module,
    datamodule: Any,
    cfg: Config,
    rates: list[float],
) -> dict[str, Any]:
    """Re-run the audit at multiple corruption rates; returns per-rate AUROC + Pearson r."""
    from copy import deepcopy  # noqa: PLC0415
    from dataclasses import replace  # noqa: PLC0415

    sweep_results: dict[str, Any] = {}

    loader_fn = (
        lambda: datamodule.test_dataloader()
        or datamodule.val_dataloader()
        or datamodule.train_dataloader()
    )

    device = cfg.training.device

    for rate in rates:
        rate_agg: dict[str, float] = {}
        rate_counts: dict[str, int] = {}
        av: list[torch.Tensor] = []
        ai: list[torch.Tensor] = []
        sv: list[torch.Tensor] = []
        si: list[torch.Tensor] = []

        sweep_bcfg = replace(cfg.benchmark, corrupt_rate=rate)
        sweep_cfg = deepcopy(cfg)
        sweep_cfg.benchmark = sweep_bcfg

        loader = loader_fn()
        for batch_idx, batch in enumerate(loader):
            has_logits = "logits" in batch and "token_ids" in batch
            if has_logits and "log_x_invalid" not in batch:
                build_invalid_batch(
                    batch,
                    K=cfg.dataset.K,
                    corrupt_rate=rate,
                    order_mix_rate=sweep_bcfg.order_mix_rate,
                    order_mix_prob=sweep_bcfg.order_mix_prob,
                    eps=cfg.hf_dataset.log_simplex_eps,
                    seed=sweep_bcfg.corruption_seed + batch_idx,
                )
            batch_dev = _to_device(batch, device)
            if "log_x_invalid" not in batch_dev:
                continue

            v_score = _auditor_score(model, batch_dev["log_x"])
            i_score = _auditor_score(model, batch_dev["log_x_invalid"])
            if v_score is not None:
                av.append(v_score)
            if i_score is not None:
                ai.append(i_score)

            if has_logits and sweep_bcfg.compute_spilled_energy:
                _, sp_v, sp_i = _spilled_metrics(
                    batch["logits"], batch["token_ids"],
                    batch["logits_invalid"], batch["token_ids_invalid"],
                )
                sv.append(-sp_v.mean(dim=1).detach().cpu())
                si.append(-sp_i.mean(dim=1).detach().cpu())

        def _cat(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        v_a = _cat(av)
        i_a = _cat(ai)
        v_s = _cat(sv)
        i_s = _cat(si)

        entry: dict[str, Any] = {
            "auroc_auditor": _safe_auroc(v_a, i_a),
            "auroc_spilled": _safe_auroc(v_s, i_s),
        }
        if v_s.size and v_a.size and len(v_s) == len(v_a):
            entry["pearson_r_valid"] = float(np.corrcoef(v_a, v_s)[0, 1])
        if i_s.size and i_a.size and len(i_s) == len(i_a):
            entry["pearson_r_invalid"] = float(np.corrcoef(i_a, i_s)[0, 1])

        sweep_results[str(rate)] = entry

    return sweep_results
