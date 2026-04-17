"""Frozen pretrained LLM backbone + one-class GP auditor for the scaling experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.gp.gp import SparseGP
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.factory import register


class FrozenHFExtractor(nn.Module):
    """Wraps a pretrained HF model as a frozen feature extractor.

    Loads ``model_id``, freezes all parameters, mean-pools the last hidden state,
    and projects to ``d_latent`` via a trainable linear layer.
    """

    def __init__(self, model_id: str, d_latent: int) -> None:
        super().__init__()
        from transformers import AutoModel  # noqa: PLC0415

        hf_model = AutoModel.from_pretrained(model_id)
        hf_model.eval()
        for p in hf_model.parameters():
            p.requires_grad_(False)
        self.hf_model = hf_model
        hidden_size: int = hf_model.config.hidden_size
        self.proj = nn.Linear(hidden_size, d_latent)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``input_ids`` (B, L) → (B, d_latent)."""
        with torch.no_grad():
            out = self.hf_model(input_ids)
        h = out.last_hidden_state.mean(dim=1)  # (B, hidden_size)
        return self.proj(h)  # (B, d_latent)


def build_frozen_extractor(model_id: str, d_latent: int) -> FrozenHFExtractor:
    return FrozenHFExtractor(model_id, d_latent)


@dataclass
class _FrozenAuditResult:
    """Convenience holder returned by audit()."""
    ood_var_mean: torch.Tensor
    ood_var_std: torch.Tensor


class FrozenBackboneAuditor(nn.Module):
    """One-class GP auditor with a frozen pretrained LLM backbone.

    Trains with: anchor loss (keep valid GP mean near zero) + KL regularisation.
    No flow-matching loss — the backbone is frozen and non-differentiable w.r.t.
    the log-simplex representation, so velocity computation is not applicable here.

    Expected batch keys:
      - ``token_ids``: (B, L) long tensor of token IDs (LLM tokeniser vocabulary)
      - ``token_ids_invalid``: (B, L) — only used at eval for contrastive metrics
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        backbone_id = cfg.transformer.pretrained_backbone
        if not backbone_id:
            raise ValueError(
                "FrozenBackboneAuditor requires cfg.transformer.pretrained_backbone to be set."
            )
        self.extractor = build_frozen_extractor(backbone_id, cfg.transformer.d_latent)
        self.gp = SparseGP(cfg=cfg)

    def _extract(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``token_ids`` (B, L) → (B, d_latent) GP input."""
        return self.extractor(token_ids)

    def _one_class_loss(self, token_ids: torch.Tensor) -> LossDict:
        B = token_ids.shape[0]
        z = self._extract(token_ids)
        dist = self.gp(z)
        anchor = dist.mean.pow(2).mean()
        kl = self.gp.kl_divergence()
        total = anchor + self.cfg.gp.lambda_kl * kl / B
        return {
            TRAINING_LOSS_KEY: total,
            "anchor": anchor.detach(),
            "kl": kl.detach(),
        }

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._one_class_loss(batch["token_ids"])

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        return self._one_class_loss(batch["token_ids"])

    @torch.no_grad()
    def ood_score(self, token_ids: torch.Tensor) -> torch.Tensor:
        """GP epistemic variance as OOD score, shape (B,)."""
        was_training = self.training
        self.eval()
        dist = self.gp(self._extract(token_ids))
        if was_training:
            self.train()
        return dist.variance

    @torch.no_grad()
    def score_per_sample(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.ood_score(token_ids)

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        scores = self.ood_score(batch["token_ids"])
        return {"ood_var_mean": scores.mean(), "ood_var_std": scores.std(unbiased=False)}


@register("frozen_backbone_auditor")
def build_frozen_backbone_auditor(cfg: Config) -> FrozenBackboneAuditor:
    return FrozenBackboneAuditor(cfg)
