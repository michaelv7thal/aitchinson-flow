from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.data.corruption import build_invalid_batch
from aitchinson_flow.data.feature_dim import feature_dim as _feature_dim_for
from aitchinson_flow.metrics.auroc import safe_auroc
from aitchinson_flow.metrics.spilled_energy import compute_spilled_energy_batch
from aitchinson_flow.models.base import AuditorModel
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.training.datamodule import DataModule
from aitchinson_flow.training.metrics import finalize_averages, running_average
from benchmarks.tasks.registry import register


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


def _auditor_residual_score(model: nn.Module, log_x: torch.Tensor) -> torch.Tensor | None:
    """Return the flow-residual UQ score (`residual_score`) if the model exposes one.

    Available on `BayesianAuditorStage1` (direct velocity head) and on composed
    `BayesianAuditor` (GP-gradient velocity norm). Used to compare flow-based
    UQ against GP-variance UQ side-by-side in the benchmark.
    """
    fn = getattr(model, "residual_score", None)
    if fn is None:
        return None
    try:
        out = fn(log_x)
    except TypeError:
        return None
    return out.detach().cpu()


def _auditor_energy_score(model: nn.Module, log_x: torch.Tensor) -> torch.Tensor | None:
    """Return the Stage-1-style geometric energy score if the model exposes one.

    Computed from the explicit `energy_score(log_x)` API as
    ``g(x) = -d_H(f(x), x)`` (per-sequence, averaged over tokens).
    Sign-flipped to anomaly convention (higher = more OOD) for direct AUROC
    comparison against the GP-variance score, mirroring Plan point 5
    (Stage 1 geometric energy vs Stage 2 GP variance).

    Returns ``None`` if the model has no `energy_score` (e.g. Stage 2 only or
    plain `BayesianAuditor` without a Stage 1 backbone surface).
    """
    fn = getattr(model, "energy_score", None)
    if fn is None:
        return None
    try:
        out = fn(log_x)
    except TypeError:
        return None
    # Anomaly convention: AUROC labels invalid=1, so higher score = more OOD.
    return (-out).detach().cpu()


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


def _auditor_token_latents(model: nn.Module, log_x: torch.Tensor) -> torch.Tensor | None:
    """Per-token latent representation ``(B, L, d)`` when the model exposes one.

    Used by the stage-aware density plot. Stage 1 returns backbone hidden
    states, Stage 2 / fused return GP-input latents.
    """
    fn = getattr(model, "token_latents", None)
    if fn is None:
        return None
    try:
        out = fn(log_x)
    except TypeError:
        return None
    if not torch.is_tensor(out):
        return None
    return out.detach().cpu()


def _auditor_inducing_points(model: nn.Module) -> torch.Tensor | None:
    """GP inducing locations ``(M, d)`` when available (Stage 2 / fused)."""
    fn = getattr(model, "inducing_points", None)
    if fn is None:
        return None
    try:
        out = fn()
    except TypeError:
        return None
    if out is None or not torch.is_tensor(out):
        return None
    return out.detach().cpu()


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
        residual_valid: list[torch.Tensor] = []
        residual_invalid: list[torch.Tensor] = []
        energy_valid: list[torch.Tensor] = []
        energy_invalid: list[torch.Tensor] = []
        spilled_valid: list[torch.Tensor] = []
        spilled_invalid: list[torch.Tensor] = []
        sp_valid_seqs: list[torch.Tensor] = []
        sp_invalid_seqs: list[torch.Tensor] = []
        audit_energy_valid_seqs: list[torch.Tensor] = []
        audit_energy_invalid_seqs: list[torch.Tensor] = []
        audit_var_valid_seqs: list[torch.Tensor] = []
        audit_var_invalid_seqs: list[torch.Tensor] = []
        latent_valid_seqs: list[torch.Tensor] = []
        latent_invalid_seqs: list[torch.Tensor] = []
        token_ids_valid_seqs: list[torch.Tensor] = []
        token_ids_invalid_seqs: list[torch.Tensor] = []
        # Latents can be large; cap the number of sequences retained for plots.
        latent_cap = int(getattr(bcfg, "plot_latent_cap", 256))
        inducing_points: torch.Tensor | None = None

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
                    label_smoothing=cfg.hf_dataset.label_smoothing,
                    transform_mode=cfg.hf_dataset.transform_mode,
                    seed=bcfg.corruption_seed + batch_idx,
                )

            batch_dev = to_device(batch, device)
            out = auditor.audit(batch_dev)
            running_average(agg, counts, out)

            if "token_ids" in batch and "token_ids_invalid" in batch:
                token_ids_valid_seqs.append(batch["token_ids"].detach().cpu())
                token_ids_invalid_seqs.append(batch["token_ids_invalid"].detach().cpu())

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

            v_res = _auditor_residual_score(model, batch_dev["log_x"])
            i_res = (
                _auditor_residual_score(model, batch_dev["log_x_invalid"])
                if "log_x_invalid" in batch_dev
                else None
            )
            if v_res is not None:
                residual_valid.append(v_res)
            if i_res is not None:
                residual_invalid.append(i_res)

            v_eng = _auditor_energy_score(model, batch_dev["log_x"])
            i_eng = (
                _auditor_energy_score(model, batch_dev["log_x_invalid"])
                if "log_x_invalid" in batch_dev
                else None
            )
            if v_eng is not None:
                energy_valid.append(v_eng)
            if i_eng is not None:
                energy_invalid.append(i_eng)

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

            current_latent_count = sum(t.shape[0] for t in latent_valid_seqs)
            if current_latent_count < latent_cap:
                take_v = _auditor_token_latents(model, batch_dev["log_x"])
                if take_v is not None:
                    room = max(0, latent_cap - current_latent_count)
                    latent_valid_seqs.append(take_v[:room])
                if "log_x_invalid" in batch_dev:
                    take_i = _auditor_token_latents(model, batch_dev["log_x_invalid"])
                    if take_i is not None:
                        current_inv = sum(t.shape[0] for t in latent_invalid_seqs)
                        room = max(0, latent_cap - current_inv)
                        latent_invalid_seqs.append(take_i[:room])

            if inducing_points is None:
                inducing_points = _auditor_inducing_points(model)

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
        v_resid = _cat(residual_valid)
        i_resid = _cat(residual_invalid)
        v_eng = _cat(energy_valid)
        i_eng = _cat(energy_invalid)
        v_spill = _cat(spilled_valid)
        i_spill = _cat(spilled_invalid)

        result["auroc_auditor"] = safe_auroc(v_audit, i_audit)
        result["auroc_spilled"] = safe_auroc(v_spill, i_spill)
        if v_resid.size and i_resid.size:
            result["auroc_residual"] = safe_auroc(v_resid, i_resid)
        if v_eng.size and i_eng.size:
            # Stage 1 geometric energy AUROC (Plan point 5: energy vs variance).
            result["auroc_energy"] = safe_auroc(v_eng, i_eng)

        # Pearson r between GP variance and spilled energy scores
        if v_spill.size and v_audit.size and len(v_spill) == len(v_audit):
            result["pearson_r_valid"] = float(np.corrcoef(v_audit, v_spill)[0, 1])
        if i_spill.size and i_audit.size and len(i_spill) == len(i_audit):
            result["pearson_r_invalid"] = float(np.corrcoef(i_audit, i_spill)[0, 1])

        # Ablation tags: surface the configuration switches that produced this
        # row so a sweep CSV/JSON can be sliced cleanly along the
        # Hilbert-vs-MSE / ILR-vs-CLR / model / scale axes (Plan point 5).
        result["ablation_tags"] = {
            "model_name": cfg.training.model_name,
            "velocity_loss": cfg.training.velocity_loss,
            "transform_mode": cfg.hf_dataset.transform_mode,
            "label_smoothing": float(cfg.hf_dataset.label_smoothing),
            "feature_dim": int(_feature_dim_for(cfg)),
            "K": int(cfg.dataset.K),
            "L": int(cfg.dataset.L),
            "d_model": int(cfg.transformer.d_model),
            "num_layers": int(cfg.transformer.num_layers),
            "nhead": int(cfg.transformer.nhead),
            "corrupt_rate": float(bcfg.corrupt_rate),
        }

        audit_energy_seq_valid_np = (
            torch.cat(audit_energy_valid_seqs, dim=0).numpy() if audit_energy_valid_seqs else None
        )
        audit_energy_seq_invalid_np = (
            torch.cat(audit_energy_invalid_seqs, dim=0).numpy()
            if audit_energy_invalid_seqs
            else None
        )
        audit_var_seq_valid_np = (
            torch.cat(audit_var_valid_seqs, dim=0).numpy() if audit_var_valid_seqs else None
        )
        audit_var_seq_invalid_np = (
            torch.cat(audit_var_invalid_seqs, dim=0).numpy() if audit_var_invalid_seqs else None
        )
        spilled_token_valid_np = (
            torch.cat(sp_valid_seqs, dim=0).numpy() if sp_valid_seqs else None
        )
        spilled_token_invalid_np = (
            torch.cat(sp_invalid_seqs, dim=0).numpy() if sp_invalid_seqs else None
        )

        # Corruption ground truth: derived from (token_ids_invalid != token_ids).
        # Captures random-replace and partial-shuffle effects alike, at the cost
        # of missing the rare case where a shuffle leaves a token in place.
        corrupt_mask_invalid_np: np.ndarray | None = None
        if token_ids_valid_seqs and token_ids_invalid_seqs:
            tids_v = torch.cat(token_ids_valid_seqs, dim=0)
            tids_i = torch.cat(token_ids_invalid_seqs, dim=0)
            if tids_v.shape == tids_i.shape:
                corrupt_mask_invalid_np = (tids_i != tids_v).to(torch.int8).numpy()

        def _seq_mean(x: np.ndarray | None) -> np.ndarray | None:
            return None if x is None else x.mean(axis=1)

        result["_scores"] = {
            "auditor_valid": v_audit,
            "auditor_invalid": i_audit,
            "residual_valid": v_resid,
            "residual_invalid": i_resid,
            "energy_valid": v_eng,
            "energy_invalid": i_eng,
            "spilled_valid": v_spill,
            "spilled_invalid": i_spill,
            "spilled_token_valid": spilled_token_valid_np,
            "spilled_token_invalid": spilled_token_invalid_np,
            "auditor_energy_seq_valid": audit_energy_seq_valid_np,
            "auditor_energy_seq_invalid": audit_energy_seq_invalid_np,
            "auditor_var_seq_valid": audit_var_seq_valid_np,
            "auditor_var_seq_invalid": audit_var_seq_invalid_np,
            "auditor_energy_scalar_valid": _seq_mean(audit_energy_seq_valid_np),
            "auditor_energy_scalar_invalid": _seq_mean(audit_energy_seq_invalid_np),
            "auditor_variance_scalar_valid": _seq_mean(audit_var_seq_valid_np),
            "auditor_variance_scalar_invalid": _seq_mean(audit_var_seq_invalid_np),
            "corrupt_mask_invalid": corrupt_mask_invalid_np,
            "latent_tokens_valid": (
                torch.cat(latent_valid_seqs, dim=0).numpy() if latent_valid_seqs else None
            ),
            "latent_tokens_invalid": (
                torch.cat(latent_invalid_seqs, dim=0).numpy() if latent_invalid_seqs else None
            ),
            "inducing_points": (
                inducing_points.numpy() if inducing_points is not None else None
            ),
        }

        # Token-level AUROC surfaced in the orchestration manifest.
        if audit_energy_seq_valid_np is not None and audit_energy_seq_invalid_np is not None:
            result["auroc_auditor_token_all"] = safe_auroc(
                audit_energy_seq_valid_np.ravel(),
                audit_energy_seq_invalid_np.ravel(),
            )
            if corrupt_mask_invalid_np is not None:
                mask_bool = corrupt_mask_invalid_np.astype(bool)
                if mask_bool.shape == audit_energy_seq_invalid_np.shape:
                    result["auroc_auditor_token_corrupted"] = safe_auroc(
                        audit_energy_seq_valid_np.ravel(),
                        audit_energy_seq_invalid_np[mask_bool],
                    )
        if spilled_token_valid_np is not None and spilled_token_invalid_np is not None:
            result["auroc_spilled_token_all"] = safe_auroc(
                spilled_token_valid_np.ravel(),
                spilled_token_invalid_np.ravel(),
            )

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
                    label_smoothing=cfg.hf_dataset.label_smoothing,
                    transform_mode=cfg.hf_dataset.transform_mode,
                    seed=sweep_bcfg.corruption_seed + batch_idx,
                )
            batch_dev = to_device(batch, device)
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
            "auroc_auditor": safe_auroc(v_a, i_a),
            "auroc_spilled": safe_auroc(v_s, i_s),
        }
        if v_s.size and v_a.size and len(v_s) == len(v_a):
            entry["pearson_r_valid"] = float(np.corrcoef(v_a, v_s)[0, 1])
        if i_s.size and i_a.size and len(i_s) == len(i_a):
            entry["pearson_r_invalid"] = float(np.corrcoef(i_a, i_s)[0, 1])

        sweep_results[str(rate)] = entry

    return sweep_results
