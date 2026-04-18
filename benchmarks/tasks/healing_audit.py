"""Healing audit task — measures variance reduction and AUROC improvement after EqM healing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.models.factory import build_model
from aitchinson_flow.training.datamodule import DataModule
from benchmarks.corruption import build_invalid_batch
from benchmarks.healing import HealingPipeline
from benchmarks.tasks.registry import register
from benchmarks.tasks.text_audit import _safe_auroc, _to_device


def _load_healer(cfg: Config) -> nn.Module:
    """Load the EquilibriumAuditor from ``cfg.benchmark.healer_ckpt``."""
    import torch  # noqa: PLC0415

    healer = build_model(cfg)
    ckpt_path = cfg.benchmark.healer_ckpt
    if not ckpt_path:
        raise ValueError(
            "HealingAuditTask requires cfg.benchmark.healer_ckpt to point to a trained "
            "EquilibriumAuditor checkpoint."
        )
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        healer.load_state_dict(state["model_state_dict"])
    else:
        healer.load_state_dict(state)
    return healer


@register("healing_audit")
class HealingAuditTask:
    """Benchmark the self-healing pipeline.

    For each eval batch:
      1. Build corrupted (OOD) sequences via ``build_invalid_batch()``.
      2. Run ``HealingPipeline.heal_batch()`` on the corrupted sequences.
      3. Re-score both original (valid) and healed sequences.

    Metrics:
      - ``pre_auroc``:  AUROC(auditor) before healing (valid vs corrupted).
      - ``post_auroc``: AUROC(auditor) after healing (valid vs healed-corrupted).
      - ``mean_var_reduction``: mean drop in GP variance for healed sequences.
      - ``healing_success_rate``: fraction of flagged sequences that cross the threshold.
    """

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

        healer = _load_healer(cfg).to(device)
        healer.eval()

        pipeline = HealingPipeline(
            auditor=model,
            healer=healer,
            var_threshold=cfg.benchmark.healing_var_threshold,
        )

        bcfg = cfg.benchmark
        loader = self._choose_loader(datamodule)

        pre_valid: list[torch.Tensor] = []
        pre_invalid: list[torch.Tensor] = []
        post_valid: list[torch.Tensor] = []
        post_invalid: list[torch.Tensor] = []
        var_reductions: list[float] = []
        success_rates: list[float] = []

        ood_score_fn = getattr(model, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("model must implement ood_score(log_x)")

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
            elif "log_x_invalid" not in batch:
                # text8 path: corrupt manually
                from benchmarks.corruption import corrupt_token_ids, token_ids_to_ilr_x_batch  # noqa: PLC0415
                batch = _corrupt_text8_batch(batch, cfg, bcfg, batch_idx)

            batch_dev = _to_device(batch, device)
            log_x = batch_dev["log_x"]
            log_x_invalid = batch_dev["log_x_invalid"]

            # Pre-healing scores
            pre_v = ood_score_fn(log_x).detach().cpu()
            pre_i = ood_score_fn(log_x_invalid).detach().cpu()
            pre_valid.append(pre_v)
            pre_invalid.append(pre_i)

            # Heal corrupted sequences
            result = pipeline.heal_batch(log_x_invalid, steps=bcfg.healing_steps)
            post_v = ood_score_fn(log_x).detach().cpu()
            post_i = result.post_var.detach().cpu()
            post_valid.append(post_v)
            post_invalid.append(post_i)

            # Variance reduction only for flagged sequences
            if result.healed_mask.any():
                pre_flagged = result.pre_var[result.healed_mask].cpu()
                post_flagged = result.post_var[result.healed_mask].cpu()
                reduction = float((pre_flagged - post_flagged).mean().item())
                var_reductions.append(reduction)

            success_rates.append(result.success_rate)

        def _cat(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        pv = _cat(pre_valid)
        pi = _cat(pre_invalid)
        qv = _cat(post_valid)
        qi = _cat(post_invalid)

        pre_auroc = _safe_auroc(pv, pi)
        post_auroc = _safe_auroc(qv, qi)
        mean_var_reduction = float(np.mean(var_reductions)) if var_reductions else float("nan")
        healing_success_rate = float(np.mean(success_rates)) if success_rates else 0.0

        return {
            "pre_auroc": pre_auroc,
            "post_auroc": post_auroc,
            "auroc_improvement": post_auroc - pre_auroc,
            "mean_var_reduction": mean_var_reduction,
            "healing_success_rate": healing_success_rate,
            "_scores": {
                "auditor_valid": qv,
                "auditor_invalid": qi,
                "pre_invalid": pi,
                "post_invalid": qi,
            },
        }


def _corrupt_text8_batch(
    batch: dict, cfg: Config, bcfg: Any, batch_idx: int
) -> dict:
    """Corrupt a text8 batch to produce log_x_invalid in-place.

    Honors ``cfg.hf_dataset.label_smoothing`` and
    ``cfg.hf_dataset.transform_mode`` so the healing path matches whatever
    discrete→continuous pipeline the model was trained with.
    """
    from benchmarks.corruption import corrupt_token_ids  # noqa: PLC0415
    from aitchinson_flow.data.transforms.discrete import token_ids_to_features  # noqa: PLC0415

    token_ids = batch["token_ids"]  # (B, L)
    seed = bcfg.corruption_seed + batch_idx
    corrupted = corrupt_token_ids(
        token_ids,
        vocab_size=cfg.dataset.K,
        corrupt_rate=bcfg.corrupt_rate,
        seed=seed,
    )
    log_x_invalid = torch.stack([
        token_ids_to_features(
            corrupted[i],
            cfg.dataset.K,
            eps=cfg.hf_dataset.log_simplex_eps,
            label_smoothing=cfg.hf_dataset.label_smoothing,
            transform_mode=cfg.hf_dataset.transform_mode,
        )
        for i in range(corrupted.shape[0])
    ])
    batch = dict(batch)
    batch["log_x_invalid"] = log_x_invalid
    batch["token_ids_invalid"] = corrupted
    return batch
