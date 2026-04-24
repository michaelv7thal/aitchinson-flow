"""Healing audit task — evaluate all Phase 4 healing strategies.

Dispatches to one of four healing strategies based on ``cfg.healing.strategy``:

* ``"targeted_resample"`` — :class:`TargetedResampler`: mask high-energy token
  positions, project via EqM (or LLM when token_ids available), iterate.
* ``"simplex_project"`` — :class:`SimplexProjectHealer`: project high-energy
  sequences to valid manifold via EqM + nearest-token snap via ILR inverse.
* ``"beam_rerank"`` — :class:`BeamRerankScorer`: score corrupted vs healed
  candidates by ``log_p − λ · energy``; pick best.
* ``"eqm"`` (legacy / default) — whole-sequence EqM integration via the
  existing :class:`HealingPipeline`.

Metrics reported (all strategies):
  ``pre_auroc``              AUROC(auditor) before healing (valid vs corrupted).
  ``post_auroc``             AUROC(auditor) after healing  (valid vs healed).
  ``auroc_improvement``      post − pre.
  ``mean_energy_reduction``  Mean drop in OOD score for healed sequences.
  ``mean_var_reduction``     Mean drop in GP variance (when auditor has ``per_token_uq``).
  ``healing_success_rate``   Fraction of flagged seqs where energy dropped.
  ``token_accuracy``         Fraction of healed positions matching ground-truth
                             token IDs (only when ``batch["token_ids"]`` available).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.data.corruption import build_invalid_batch
from aitchinson_flow.healing.targeted_resample import TargetedResampler
from aitchinson_flow.healing.simplex_project import SimplexProjectHealer
from aitchinson_flow.healing.beam_rerank import BeamRerankScorer
from aitchinson_flow.metrics.auroc import safe_auroc
from aitchinson_flow.models.factory import build_model
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.training.datamodule import DataModule
from benchmarks.healing import HealingPipeline
from benchmarks.tasks.registry import register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_healer(cfg: Config) -> nn.Module:
    """Load the EquilibriumAuditor from ``cfg.benchmark.healer_ckpt``."""
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


def _corrupt_text8_batch(batch: dict, cfg: Config, bcfg: Any, batch_idx: int) -> dict:
    """Corrupt a text8 batch to produce log_x_invalid in-place."""
    from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: PLC0415
    from aitchinson_flow.data.transforms.discrete import token_ids_to_features  # noqa: PLC0415

    token_ids = batch["token_ids"]
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


def _cat(xs: list[torch.Tensor]) -> np.ndarray:
    return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)


def _token_accuracy(
    healed_token_ids: torch.Tensor | None,
    gt_token_ids: torch.Tensor | None,
    changed_mask: torch.Tensor | None,
) -> float | None:
    """Fraction of changed positions where healed token matches ground-truth.

    Returns ``None`` if any required tensor is missing.
    """
    if healed_token_ids is None or gt_token_ids is None or changed_mask is None:
        return None
    if not changed_mask.any():
        return float("nan")
    correct = (healed_token_ids[changed_mask] == gt_token_ids[changed_mask]).float()
    return float(correct.mean().item())


# ---------------------------------------------------------------------------
# Registered task
# ---------------------------------------------------------------------------

@register("healing_audit")
class HealingAuditTask:
    """Benchmark the Phase 4 healing pipeline.

    Selects the healing strategy from ``cfg.healing.strategy``; falls back to
    ``"eqm"`` (legacy whole-sequence EqM integration) if ``cfg.healing`` is not
    populated or if no strategy is configured.
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
        strategy = cfg.healing.strategy

        if strategy == "targeted_resample":
            return self._run_targeted_resample(model, datamodule, cfg)
        elif strategy == "simplex_project":
            return self._run_simplex_project(model, datamodule, cfg)
        elif strategy == "beam_rerank":
            return self._run_beam_rerank(model, datamodule, cfg)
        else:
            # "eqm" or any unrecognised value: legacy path
            return self._run_eqm(model, datamodule, cfg)

    # ------------------------------------------------------------------
    # Strategy: targeted resample
    # ------------------------------------------------------------------

    def _run_targeted_resample(
        self, model: nn.Module, datamodule: DataModule, cfg: Config
    ) -> dict[str, Any]:
        device = cfg.training.device
        model = model.to(device).eval()
        healer = _load_healer(cfg).to(device).eval()
        hcfg = cfg.healing

        resampler = TargetedResampler(
            auditor=model,
            healer=healer,
            threshold=hcfg.threshold,
            max_iter=hcfg.max_iter,
            steps=hcfg.heal_steps,
            dt=hcfg.heal_dt,
        )

        bcfg = cfg.benchmark
        loader = self._choose_loader(datamodule)
        ood_score_fn = getattr(model, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("model must implement ood_score(log_x)")

        pre_valid: list[torch.Tensor] = []
        pre_invalid: list[torch.Tensor] = []
        post_valid: list[torch.Tensor] = []
        post_invalid: list[torch.Tensor] = []
        success_rates: list[float] = []
        token_accuracies: list[float] = []

        for batch_idx, batch in enumerate(loader):
            batch = self._ensure_invalid(batch, cfg, bcfg, batch_idx)
            batch_dev = to_device(batch, device)
            log_x = batch_dev["log_x"]
            log_x_invalid = batch_dev["log_x_invalid"]
            token_ids = batch_dev.get("token_ids")
            token_ids_invalid = batch_dev.get("token_ids_invalid")

            pre_valid.append(ood_score_fn(log_x).detach().cpu())
            pre_invalid.append(ood_score_fn(log_x_invalid).detach().cpu())

            result = resampler.heal(
                log_x_invalid,
                token_ids=token_ids_invalid,
                K=cfg.dataset.K,
                eps=hcfg.eps,
                label_smoothing=hcfg.label_smoothing,
                transform_mode=hcfg.transform_mode,
            )

            post_valid.append(ood_score_fn(log_x).detach().cpu())
            post_invalid.append(result.post_energy.cpu())
            success_rates.append(result.success_rate)

            # Token accuracy (when ground-truth token_ids available)
            if token_ids is not None and result.token_mask.any():
                # Recover healed token IDs via argmax on ilr_inv
                from aitchinson_flow.geometry import ilr_inv  # noqa: PLC0415
                healed_log_probs = ilr_inv(
                    result.healed_log_x.cpu(), cfg.dataset.K
                )  # (B, L, K)
                healed_tok_ids = healed_log_probs.argmax(-1).cpu()  # (B, L)
                acc = _token_accuracy(healed_tok_ids, token_ids.cpu(), result.token_mask.cpu())
                if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
                    token_accuracies.append(acc)

        return _build_result_dict(
            pre_valid, pre_invalid, post_valid, post_invalid,
            success_rates, token_accuracies, strategy="targeted_resample"
        )

    # ------------------------------------------------------------------
    # Strategy: simplex project
    # ------------------------------------------------------------------

    def _run_simplex_project(
        self, model: nn.Module, datamodule: DataModule, cfg: Config
    ) -> dict[str, Any]:
        device = cfg.training.device
        model = model.to(device).eval()
        healer = _load_healer(cfg).to(device).eval()
        hcfg = cfg.healing

        projector = SimplexProjectHealer(
            auditor=model,
            healer=healer,
            energy_threshold=hcfg.threshold,
            steps=hcfg.heal_steps,
            dt=hcfg.heal_dt,
            re_encode=hcfg.re_encode,
            eps=hcfg.eps,
            label_smoothing=hcfg.label_smoothing,
            transform_mode=hcfg.transform_mode,
        )

        bcfg = cfg.benchmark
        loader = self._choose_loader(datamodule)
        ood_score_fn = getattr(model, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("model must implement ood_score(log_x)")

        pre_valid: list[torch.Tensor] = []
        pre_invalid: list[torch.Tensor] = []
        post_valid: list[torch.Tensor] = []
        post_invalid: list[torch.Tensor] = []
        success_rates: list[float] = []
        token_accuracies: list[float] = []

        for batch_idx, batch in enumerate(loader):
            batch = self._ensure_invalid(batch, cfg, bcfg, batch_idx)
            batch_dev = to_device(batch, device)
            log_x = batch_dev["log_x"]
            log_x_invalid = batch_dev["log_x_invalid"]
            token_ids = batch_dev.get("token_ids")

            pre_valid.append(ood_score_fn(log_x).detach().cpu())
            pre_invalid.append(ood_score_fn(log_x_invalid).detach().cpu())

            result = projector.heal(log_x_invalid, K=cfg.dataset.K)

            post_valid.append(ood_score_fn(log_x).detach().cpu())
            post_invalid.append(result.post_energy.cpu())
            success_rates.append(result.success_rate)

            # Token accuracy: compare nearest_token_ids to ground-truth
            if token_ids is not None and result.seq_mask.any():
                gt = token_ids.cpu()
                healed = result.nearest_token_ids.cpu()
                # Mask: changed positions are those in healed sequences
                seq_mask_expanded = result.seq_mask.cpu().unsqueeze(1).expand_as(gt)
                acc = _token_accuracy(healed, gt, seq_mask_expanded)
                if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
                    token_accuracies.append(acc)

        return _build_result_dict(
            pre_valid, pre_invalid, post_valid, post_invalid,
            success_rates, token_accuracies, strategy="simplex_project"
        )

    # ------------------------------------------------------------------
    # Strategy: beam rerank
    # ------------------------------------------------------------------

    def _run_beam_rerank(
        self, model: nn.Module, datamodule: DataModule, cfg: Config
    ) -> dict[str, Any]:
        """Beam reranking: generate healed and original as two 'beams', pick best."""
        device = cfg.training.device
        model = model.to(device).eval()
        healer = _load_healer(cfg).to(device).eval()
        hcfg = cfg.healing

        scorer = BeamRerankScorer(auditor=model, lambda_energy=hcfg.lambda_energy)

        integrate_fn = getattr(healer, "integrate", None)
        if integrate_fn is None:
            raise AttributeError("healer must implement integrate(log_x, ...)")

        integrate_kw: dict[str, Any] = {}
        if hcfg.heal_steps is not None:
            integrate_kw["steps"] = hcfg.heal_steps
        if hcfg.heal_dt is not None:
            integrate_kw["dt"] = hcfg.heal_dt

        bcfg = cfg.benchmark
        loader = self._choose_loader(datamodule)
        ood_score_fn = getattr(model, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("model must implement ood_score(log_x)")

        pre_valid: list[torch.Tensor] = []
        pre_invalid: list[torch.Tensor] = []
        post_valid: list[torch.Tensor] = []
        post_invalid: list[torch.Tensor] = []
        success_rates: list[float] = []

        for batch_idx, batch in enumerate(loader):
            batch = self._ensure_invalid(batch, cfg, bcfg, batch_idx)
            batch_dev = to_device(batch, device)
            log_x = batch_dev["log_x"]
            log_x_invalid = batch_dev["log_x_invalid"]

            pre_valid.append(ood_score_fn(log_x).detach().cpu())
            pre_invalid_energy = ood_score_fn(log_x_invalid).detach().cpu()
            pre_invalid.append(pre_invalid_energy)

            # Build two beams: [original_invalid, eqm_healed]
            # log_p is approximated by negative energy (higher energy → lower prob)
            healed_x = integrate_fn(log_x_invalid, **integrate_kw)
            healed_energy = ood_score_fn(healed_x).detach().cpu()

            B = log_x_invalid.shape[0]
            # Stack beams: (B, 2, L, D)
            log_x_beams = torch.stack(
                [log_x_invalid.cpu(), healed_x.cpu()], dim=1
            )
            # Use negative OOD score as proxy log-probability
            log_p_beams = torch.stack(
                [-pre_invalid_energy, -healed_energy], dim=1
            )  # (B, 2)

            rerank_result = scorer.rerank_batch(log_x_beams, log_p_beams)
            best_log_x = rerank_result.best_log_x.to(device)
            post_energy = ood_score_fn(best_log_x).detach().cpu()

            post_valid.append(ood_score_fn(log_x).detach().cpu())
            post_invalid.append(post_energy)

            improved = (post_energy < pre_invalid_energy).float()
            success_rates.append(float(improved.mean().item()))

        return _build_result_dict(
            pre_valid, pre_invalid, post_valid, post_invalid,
            success_rates, [], strategy="beam_rerank"
        )

    # ------------------------------------------------------------------
    # Legacy EqM strategy
    # ------------------------------------------------------------------

    def _run_eqm(
        self, model: nn.Module, datamodule: DataModule, cfg: Config
    ) -> dict[str, Any]:
        device = cfg.training.device
        model = model.to(device).eval()
        healer = _load_healer(cfg).to(device).eval()
        hcfg = cfg.healing
        bcfg = cfg.benchmark

        pipeline = HealingPipeline(
            auditor=model,
            healer=healer,
            var_threshold=hcfg.threshold,
        )
        loader = self._choose_loader(datamodule)
        ood_score_fn = getattr(model, "ood_score", None)
        if ood_score_fn is None:
            raise AttributeError("model must implement ood_score(log_x)")

        pre_valid: list[torch.Tensor] = []
        pre_invalid: list[torch.Tensor] = []
        post_valid: list[torch.Tensor] = []
        post_invalid: list[torch.Tensor] = []
        var_reductions: list[float] = []
        success_rates: list[float] = []

        for batch_idx, batch in enumerate(loader):
            batch = self._ensure_invalid(batch, cfg, bcfg, batch_idx)
            batch_dev = to_device(batch, device)
            log_x = batch_dev["log_x"]
            log_x_invalid = batch_dev["log_x_invalid"]

            pre_valid.append(ood_score_fn(log_x).detach().cpu())
            pre_invalid.append(ood_score_fn(log_x_invalid).detach().cpu())

            heal_steps = hcfg.heal_steps if hcfg.heal_steps is not None else bcfg.healing_steps
            result = pipeline.heal_batch(log_x_invalid, steps=heal_steps)
            post_valid.append(ood_score_fn(log_x).detach().cpu())
            post_invalid.append(result.post_var.detach().cpu())
            success_rates.append(result.success_rate)

            if result.healed_mask.any():
                pre_f = result.pre_var[result.healed_mask].cpu()
                post_f = result.post_var[result.healed_mask].cpu()
                var_reductions.append(float((pre_f - post_f).mean().item()))

        pv = _cat(pre_valid)
        pi = _cat(pre_invalid)
        qv = _cat(post_valid)
        qi = _cat(post_invalid)
        pre_auroc = safe_auroc(pv, pi)
        post_auroc = safe_auroc(qv, qi)
        mean_var_red = float(np.mean(var_reductions)) if var_reductions else float("nan")
        suc = float(np.mean(success_rates)) if success_rates else 0.0
        return {
            "strategy": "eqm",
            "pre_auroc": pre_auroc,
            "post_auroc": post_auroc,
            "auroc_improvement": post_auroc - pre_auroc,
            "mean_var_reduction": mean_var_red,
            "mean_energy_reduction": mean_var_red,
            "healing_success_rate": suc,
            "_scores": {
                "auditor_valid": qv,
                "auditor_invalid": qi,
                "pre_invalid": pi,
                "post_invalid": qi,
            },
        }

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_invalid(batch: dict, cfg: Config, bcfg: Any, batch_idx: int) -> dict:
        """Guarantee ``log_x_invalid`` is present in ``batch``."""
        if "log_x_invalid" in batch:
            return batch
        has_logits = "logits" in batch and "token_ids" in batch
        if has_logits:
            build_invalid_batch(
                batch,
                K=cfg.dataset.K,
                corrupt_rate=bcfg.corrupt_rate,
                order_mix_rate=bcfg.order_mix_rate,
                order_mix_prob=bcfg.order_mix_prob,
                eps=cfg.hf_dataset.log_simplex_eps,
                seed=bcfg.corruption_seed + batch_idx,
            )
        else:
            batch = _corrupt_text8_batch(batch, cfg, bcfg, batch_idx)
        return batch


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_result_dict(
    pre_valid: list[torch.Tensor],
    pre_invalid: list[torch.Tensor],
    post_valid: list[torch.Tensor],
    post_invalid: list[torch.Tensor],
    success_rates: list[float],
    token_accuracies: list[float],
    strategy: str,
) -> dict[str, Any]:
    pv = _cat(pre_valid)
    pi = _cat(pre_invalid)
    qv = _cat(post_valid)
    qi = _cat(post_invalid)

    pre_auroc = safe_auroc(pv, pi)
    post_auroc = safe_auroc(qv, qi)

    mean_energy_reduction = float(np.mean(pi - qi)) if pi.size > 0 else float("nan")
    suc = float(np.mean(success_rates)) if success_rates else 0.0
    tok_acc = float(np.mean(token_accuracies)) if token_accuracies else float("nan")

    return {
        "strategy": strategy,
        "pre_auroc": pre_auroc,
        "post_auroc": post_auroc,
        "auroc_improvement": post_auroc - pre_auroc,
        "mean_energy_reduction": mean_energy_reduction,
        "healing_success_rate": suc,
        "token_accuracy": tok_acc,
        "_scores": {
            "auditor_valid": qv,
            "auditor_invalid": qi,
            "pre_invalid": pi,
            "post_invalid": qi,
        },
    }
