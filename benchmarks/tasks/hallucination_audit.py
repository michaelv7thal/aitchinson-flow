"""Hallucination audit task — AUROC for byte-level Q+A hallucination detection.

Consumes val/test batches from
:class:`~aitchinson_flow.data.qa_datamodule.QAPairsDataModule`, where odd
indices carry LLM-generated answers (label=1) and even indices carry the
dataset's correct answer (label=0). The trained Path B Stage 2 auditor
scores each ``[Q][A]`` pair via its GP epistemic variance, restricted to
answer-span positions when ``cfg.gp.score_answer_tokens_only`` is set.

AUROC is reported as ``auroc_hallucination``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.metrics.auroc import safe_auroc
from aitchinson_flow.training.batch import to_device
from aitchinson_flow.training.datamodule import DataModule
from benchmarks.tasks.registry import register


def _score_with_mask(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    use_answer_mask: bool,
) -> torch.Tensor:
    """Return a (B,) OOD score for each sequence.

    When ``use_answer_mask`` is True, the per-token scores are averaged over
    the answer-span positions only (pushed by :func:`ood_score_tokenwise` if
    available; otherwise falls back to ``ood_score``).
    """
    log_x = batch["log_x"]
    mask = batch.get("answer_mask")

    if use_answer_mask and mask is not None and hasattr(model, "ood_score_tokenwise"):
        token_scores: torch.Tensor = model.ood_score_tokenwise(log_x)  # (B, L)
        mask_bool = mask.to(dtype=torch.bool, device=token_scores.device)
        safe = mask_bool.float()
        denom = safe.sum(dim=-1).clamp_min(1.0)
        return (token_scores * safe).sum(dim=-1) / denom

    return model.ood_score(log_x)


@register("hallucination_audit")
class HallucinationAuditTask:
    """AUROC for byte-level Q+A hallucination detection (Path B)."""

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

        use_answer_mask = bool(cfg.gp.score_answer_tokens_only)
        loader = self._choose_loader(datamodule)

        correct_scores: list[torch.Tensor] = []
        halluc_scores: list[torch.Tensor] = []

        for batch in loader:
            batch = to_device(batch, device)
            labels = batch.get(
                "label", torch.zeros(batch["log_x"].shape[0], dtype=torch.long, device=device)
            )
            with torch.no_grad():
                scores = _score_with_mask(model, batch, use_answer_mask=use_answer_mask).detach().cpu()
            labels_cpu = labels.cpu()
            correct_scores.append(scores[labels_cpu == 0])
            halluc_scores.append(scores[labels_cpu == 1])

        def _cat(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        v = _cat(correct_scores)
        i = _cat(halluc_scores)
        auroc = safe_auroc(v, i)
        return {
            "auroc_hallucination": auroc,
            "var_correct_mean": float(v.mean()) if v.size else float("nan"),
            "var_hallucinated_mean": float(i.mean()) if i.size else float("nan"),
            "var_separation": float(i.mean() - v.mean()) if (v.size and i.size) else float("nan"),
            "n_correct": int(v.size),
            "n_hallucinated": int(i.size),
            "_scores": {"auditor_valid": v, "auditor_invalid": i},
        }
