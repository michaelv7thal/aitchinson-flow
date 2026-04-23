"""DNA structural UQ audit task — AUROC per mutation type.

Evaluates Component 1 (structural geometry UQ) on nucleotide sequences.
In addition to the overall AUROC from the standard audit loop, this task
runs three targeted passes over the test loader, each applying a different
corruption mode as the invalid class:

| Metric | Invalid class |
|--------|--------------|
| ``auroc_dna_point_mutation`` | Point mutations only |
| ``auroc_dna_frameshift`` | Frameshift insertions only |
| ``auroc_dna_random`` | Random base replacement |
| ``auroc_dna_overall`` | Mixed (datamodule default) |

Expected behavior (Component 1 acceptance criteria):
- Valid DNA → low energy, low variance
- Point mutations → elevated energy at mutated position(s)
- Frameshift → high energy, high variance throughout
- Random shuffles → high energy, high variance globally
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.data.dna_datamodule import (
    corrupt_dna_frameshift,
    corrupt_dna_point_mutation,
)
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.metrics.auroc import safe_auroc
from aitchinson_flow.models.base import AuditorModel
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.training.datamodule import DataModule
from benchmarks.tasks.registry import register


def _score_invalid(
    model: nn.Module,
    valid_ids: torch.Tensor,
    corrupt_fn: Any,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a batch of valid token_ids against a corruption of them.

    Returns ``(scores_valid, scores_invalid)`` as numpy arrays, both shape (B,).
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


@register("dna_audit")
class DnaAuditTask:
    """Structural UQ audit for DNA sequences with per-mutation-type AUROC.

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
        K = cfg.dataset.K
        bcfg = cfg.benchmark

        loader = self._choose_loader(datamodule)

        # Accumulators for overall + per-mutation AUROCs
        overall_valid: list[torch.Tensor] = []
        overall_invalid: list[torch.Tensor] = []

        pm_valid: list[np.ndarray] = []
        pm_invalid: list[np.ndarray] = []
        fs_valid: list[np.ndarray] = []
        fs_invalid: list[np.ndarray] = []
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

            # Per-mutation-type passes on the batch token_ids
            if "token_ids" in batch:
                tids = batch["token_ids"]

                def _pm(ids: torch.Tensor) -> torch.Tensor:
                    return corrupt_dna_point_mutation(
                        ids, vocab_size=K, corrupt_rate=bcfg.corrupt_rate
                    )

                def _fs(ids: torch.Tensor) -> torch.Tensor:
                    return corrupt_dna_frameshift(
                        ids, vocab_size=K, corrupt_rate=bcfg.corrupt_rate
                    )

                def _rnd(ids: torch.Tensor) -> torch.Tensor:
                    from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: PLC0415
                    return corrupt_token_ids(ids, vocab_size=K, corrupt_rate=bcfg.corrupt_rate)

                with torch.no_grad():
                    sv_pm, si_pm = _score_invalid(model, tids, _pm, cfg, device)
                    sv_fs, si_fs = _score_invalid(model, tids, _fs, cfg, device)
                    sv_rnd, si_rnd = _score_invalid(model, tids, _rnd, cfg, device)

                if sv_pm.size and si_pm.size:
                    pm_valid.append(sv_pm)
                    pm_invalid.append(si_pm)
                if sv_fs.size and si_fs.size:
                    fs_valid.append(sv_fs)
                    fs_invalid.append(si_fs)
                if sv_rnd.size and si_rnd.size:
                    rnd_valid.append(sv_rnd)
                    rnd_invalid.append(si_rnd)

        def _cat_t(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        def _cat_n(xs: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(xs, axis=0) if xs else np.empty(0, dtype=np.float32)

        v_all = _cat_t(overall_valid)
        i_all = _cat_t(overall_invalid)
        v_pm = _cat_n(pm_valid)
        i_pm = _cat_n(pm_invalid)
        v_fs = _cat_n(fs_valid)
        i_fs = _cat_n(fs_invalid)
        v_rnd = _cat_n(rnd_valid)
        i_rnd = _cat_n(rnd_invalid)

        result: dict[str, Any] = {
            "auroc_dna_overall": safe_auroc(v_all, i_all),
            "auroc_dna_point_mutation": safe_auroc(v_pm, i_pm),
            "auroc_dna_frameshift": safe_auroc(v_fs, i_fs),
            "auroc_dna_random": safe_auroc(v_rnd, i_rnd),
        }

        # Summary stats
        for tag, v, i in [
            ("overall", v_all, i_all),
            ("point_mutation", v_pm, i_pm),
            ("frameshift", v_fs, i_fs),
            ("random", v_rnd, i_rnd),
        ]:
            if v.size and i.size:
                result[f"energy_gap_{tag}"] = float(i.mean() - v.mean())

        result["_scores"] = {
            "auditor_valid": v_all,
            "auditor_invalid": i_all,
            "point_mutation_valid": v_pm,
            "point_mutation_invalid": i_pm,
            "frameshift_valid": v_fs,
            "frameshift_invalid": i_fs,
            "random_valid": v_rnd,
            "random_invalid": i_rnd,
        }
        result["ablation_tags"] = {
            "model_name": cfg.training.model_name,
            "K": int(K),
            "L": int(cfg.dataset.L),
            "d_model": int(cfg.transformer.d_model),
            "corrupt_rate": float(bcfg.corrupt_rate),
        }

        return result
