"""Trivia OOD audit task — one-class GP trained on correct answers detects incorrect ones."""

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


@register("trivia_audit")
class TriviaAuditTask:
    """AUROC for one-class GP OOD detection on trivia correct vs incorrect answers.

    Expects the datamodule's val/test loader to return batches with:
      - ``log_x``: (B, L, K-1) ILR-encoded answers
      - ``label``:  (B,) long, 0=correct, 1=incorrect

    The model must implement ``ood_score(log_x) → (B,)`` variance.
    For ``FrozenBackboneAuditor`` the model scores ``token_ids`` instead; this task
    checks for ``token_ids`` in the batch and falls back if available.
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

        loader = self._choose_loader(datamodule)
        use_token_ids = hasattr(model, "ood_score") and _takes_token_ids(model)

        valid_scores: list[torch.Tensor] = []
        invalid_scores: list[torch.Tensor] = []

        for batch in loader:
            batch = to_device(batch, device)
            labels: torch.Tensor = batch.get("label", torch.zeros(batch["log_x"].shape[0], dtype=torch.long))

            if use_token_ids and "token_ids" in batch:
                scores: torch.Tensor = model.ood_score(batch["token_ids"]).detach().cpu()
            else:
                scores = model.ood_score(batch["log_x"]).detach().cpu()

            labels_cpu = labels.cpu()
            valid_scores.append(scores[labels_cpu == 0])
            invalid_scores.append(scores[labels_cpu == 1])

        def _cat(xs: list[torch.Tensor]) -> np.ndarray:
            return torch.cat(xs, dim=0).numpy() if xs else np.empty(0, dtype=np.float32)

        v = _cat(valid_scores)
        i = _cat(invalid_scores)
        auroc = safe_auroc(v, i)

        return {
            "auroc_trivia": auroc,
            "var_correct_mean": float(v.mean()) if v.size else float("nan"),
            "var_incorrect_mean": float(i.mean()) if i.size else float("nan"),
            "var_separation": float(i.mean() - v.mean()) if (v.size and i.size) else float("nan"),
            "_scores": {"auditor_valid": v, "auditor_invalid": i},
        }


def _takes_token_ids(model: nn.Module) -> bool:
    """Heuristic: FrozenBackboneAuditor uses token_ids; other auditors use log_x."""
    return model.__class__.__name__ == "FrozenBackboneAuditor"
