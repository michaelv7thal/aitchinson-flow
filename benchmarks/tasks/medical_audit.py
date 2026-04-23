"""Medical text structural + contextual UQ audit task.

Evaluates Components 1 and 2 (structural and contextual UQ) on clinical text
sequences. Runs the standard audit loop for the overall AUROC and adds three
targeted passes with different corruption modes as the invalid class:

| Metric | Invalid class |
|--------|--------------|
| ``auroc_medical_overall`` | Mixed (datamodule default — chars swap + unit + random) |
| ``auroc_medical_char_swap`` | Adjacent character swaps (typos / misspellings) |
| ``auroc_medical_unit_corrupt`` | Dosage unit replacement (mg → mcg etc.) |
| ``auroc_medical_random`` | Random character replacement |

Expected behavior:
- Valid clinical notes → low structural energy, low variance
- Drug-name misspellings (char swap) → elevated energy at corrupted characters
- Wrong dosage units → high contextual energy on the unit span
- Random replacement → high energy and variance globally
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.data.medical_datamodule import (
    VOCAB_SIZE as MEDICAL_VOCAB_SIZE,
    corrupt_medical_char_swap,
    corrupt_medical_random,
    corrupt_medical_unit,
)
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.metrics.auroc import safe_auroc
from aitchinson_flow.models.base import AuditorModel
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.training.datamodule import DataModule
from benchmarks.tasks.registry import register


def _score_invalid_medical(
    model: nn.Module,
    valid_ids: torch.Tensor,
    corrupt_fn: Any,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a batch of valid ids against a specific corruption of them.

    Returns ``(scores_valid, scores_invalid)`` as numpy arrays, shape (B,).
    """
    auditor = cast(AuditorModel, model)
    K = cfg.dataset.K
    eps = cfg.hf_dataset.log_simplex_eps
    ls = cfg.hf_dataset.label_smoothing
    tm = cfg.hf_dataset.transform_mode

    bad_ids = corrupt_fn(valid_ids).cpu()

    rows_valid = [
        token_ids_to_features(row, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        for row in valid_ids.cpu()
    ]
    rows_invalid = [
        token_ids_to_features(row, K=K, eps=eps, label_smoothing=ls, transform_mode=tm)
        for row in bad_ids
    ]
    log_x_v = torch.stack(rows_valid, dim=0).to(device)
    log_x_i = torch.stack(rows_invalid, dim=0).to(device)

    with torch.no_grad():
        fn = getattr(auditor, "score_per_sample", None)
        if fn is None:
            fn = getattr(auditor, "ood_score", None)
        if fn is None:
            return np.empty(0), np.empty(0)
        sv = fn(log_x_v).detach().cpu().numpy()
        si = fn(log_x_i).detach().cpu().numpy()

    return sv, si


@register("medical_audit")
class MedicalAuditTask:
    """Structural + contextual UQ audit for clinical text with per-corruption-type AUROC.

    Runs the standard audit loop (using the datamodule's built-in corruption)
    for the overall AUROC, then re-runs three additional passes with targeted
    corruption modes applied directly to the batch token ids.
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
        auditor = cast(AuditorModel, model)
        bcfg = cfg.benchmark

        loader = self._choose_loader(datamodule)

        overall_valid: list[torch.Tensor] = []
        overall_invalid: list[torch.Tensor] = []

        swap_valid: list[np.ndarray] = []
        swap_invalid: list[np.ndarray] = []
        unit_valid: list[np.ndarray] = []
        unit_invalid: list[np.ndarray] = []
        rnd_valid: list[np.ndarray] = []
        rnd_invalid: list[np.ndarray] = []

        for batch in loader:
            batch_dev = to_device(batch, device)

            prepare = getattr(auditor, "prepare_batch", None)
            if prepare is not None:
                with torch.no_grad():
                    batch_dev = prepare(batch_dev)

            score_fn = getattr(auditor, "score_per_sample", None)
            if score_fn is None:
                score_fn = getattr(auditor, "ood_score", None)

            if score_fn is not None and "log_x" in batch_dev:
                with torch.no_grad():
                    sv = score_fn(batch_dev["log_x"]).detach().cpu()
                overall_valid.append(sv)
                if "log_x_invalid" in batch_dev:
                    with torch.no_grad():
                        si = score_fn(batch_dev["log_x_invalid"]).detach().cpu()
                    overall_invalid.append(si)

            if "token_ids" in batch:
                tids = batch["token_ids"]
                rate = bcfg.corrupt_rate

                def _swap(ids: torch.Tensor) -> torch.Tensor:
                    return corrupt_medical_char_swap(ids, corrupt_rate=rate)

                def _unit(ids: torch.Tensor) -> torch.Tensor:
                    return corrupt_medical_unit(ids, corrupt_rate=rate)

                def _rnd(ids: torch.Tensor) -> torch.Tensor:
                    return corrupt_medical_random(ids, corrupt_rate=rate)

                with torch.no_grad():
                    sv_sw, si_sw = _score_invalid_medical(model, tids, _swap, cfg, device)
                    sv_u, si_u = _score_invalid_medical(model, tids, _unit, cfg, device)
                    sv_r, si_r = _score_invalid_medical(model, tids, _rnd, cfg, device)

                if sv_sw.size and si_sw.size:
                    swap_valid.append(sv_sw)
                    swap_invalid.append(si_sw)
                if sv_u.size and si_u.size:
                    unit_valid.append(sv_u)
                    unit_invalid.append(si_u)
                if sv_r.size and si_r.size:
                    rnd_valid.append(sv_r)
                    rnd_invalid.append(si_r)

        def _cat_t(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        def _cat_n(xs: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(xs, axis=0) if xs else np.empty(0, dtype=np.float32)

        v_all = _cat_t(overall_valid)
        i_all = _cat_t(overall_invalid)
        v_sw = _cat_n(swap_valid)
        i_sw = _cat_n(swap_invalid)
        v_u = _cat_n(unit_valid)
        i_u = _cat_n(unit_invalid)
        v_r = _cat_n(rnd_valid)
        i_r = _cat_n(rnd_invalid)

        result: dict[str, Any] = {
            "auroc_medical_overall": safe_auroc(v_all, i_all),
            "auroc_medical_char_swap": safe_auroc(v_sw, i_sw),
            "auroc_medical_unit_corrupt": safe_auroc(v_u, i_u),
            "auroc_medical_random": safe_auroc(v_r, i_r),
        }

        for tag, v, i in [
            ("overall", v_all, i_all),
            ("char_swap", v_sw, i_sw),
            ("unit_corrupt", v_u, i_u),
            ("random", v_r, i_r),
        ]:
            if v.size and i.size:
                result[f"energy_gap_{tag}"] = float(i.mean() - v.mean())

        result["_scores"] = {
            "auditor_valid": v_all,
            "auditor_invalid": i_all,
            "char_swap_valid": v_sw,
            "char_swap_invalid": i_sw,
            "unit_corrupt_valid": v_u,
            "unit_corrupt_invalid": i_u,
            "random_valid": v_r,
            "random_invalid": i_r,
        }
        result["ablation_tags"] = {
            "model_name": cfg.training.model_name,
            "K": MEDICAL_VOCAB_SIZE,
            "L": int(cfg.dataset.L),
            "d_model": int(cfg.transformer.d_model),
            "corrupt_rate": float(bcfg.corrupt_rate),
        }

        return result
