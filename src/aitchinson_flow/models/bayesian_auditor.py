from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.bayesian_generator import BayesianGenerator, _uniform_log_x0
from aitchinson_flow.models.factory import register
from aitchinson_flow.config import Config


class BayesianAuditor(BayesianGenerator):
    """
    BayesianGenerator + contrastive energy/variance hinge losses.

    Expects batches with:
      - batch["log_x"]: valid samples, shape (B, L, D)
      - batch["log_x_invalid"]: invalid/OOD samples, shape (B, L, D)
    """

    def _auditor_loss(
        self,
        log_x1: torch.Tensor,
        log_x1_invalid: torch.Tensor,
        *,
        create_graph: bool,
    ) -> LossDict:
        B, L, K = log_x1.shape

        # Base flow-matching terms (aligned with BayesianGenerator: uniform log_x0 → log_x1)
        log_x0 = _uniform_log_x0(B, L, K, log_x1.device).to(dtype=log_x1.dtype)
        t = torch.rand(B, device=log_x1.device, dtype=log_x1.dtype)
        log_xt = (
            ((1.0 - t[:, None, None]) * log_x0 + t[:, None, None] * log_x1)
            .detach()
            .requires_grad_(True)
        )
        u_tgt = (log_x1 - log_x0).detach()
        with torch.enable_grad():
            v_pred = self._velocity(log_xt, create_graph=create_graph)
        noise_var = self.gp.noise_var

        if self.cfg.bayesian_generator.use_gp_variance_weighting:
            with torch.no_grad():
                z_t = self._extract(log_xt.detach())
                var_t = self.gp(z_t).variance

            total_var = var_t + noise_var
            w = 1.0 / total_var
            w = w / w.mean()
            weighted_mse = (w[:, None, None] * (v_pred - u_tgt).pow(2)).mean()

        else:
            weighted_mse = self._velocity_loss_fn(v_pred, u_tgt)

        flow_loss = 0.5 * weighted_mse / noise_var + 0.5 * noise_var.log()

        # Contrastive valid/invalid GP losses
        valid_dist = self.gp(self._extract(log_x1))
        invalid_dist = self.gp(self._extract(log_x1_invalid))

        energy_gap = self.cfg.gp.margin_E - (invalid_dist.mean - valid_dist.mean)
        mean_loss = F.relu(energy_gap).mean()

        # mean_loss = mean_loss + self.cfg.gp.lambda_anchor * valid_dist.mean.pow(2).mean()

        var_gap = self.cfg.gp.margin_V - (invalid_dist.variance - valid_dist.variance)
        var_loss = F.relu(var_gap).mean()

        kl = self.gp.kl_divergence()

        total = (
            flow_loss
            + mean_loss
            + self.cfg.gp.lambda_var * var_loss
            + self.cfg.gp.lambda_kl * kl / B
        )

        return {
            TRAINING_LOSS_KEY: total,
            "flow_loss": flow_loss,
            "mean_loss": mean_loss,
            "var_loss": var_loss,
            "kl": kl,
            "noise_var": noise_var.detach(),
        }

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        if "log_x_invalid" not in batch:
            raise KeyError("BayesianAuditor requires batch['log_x_invalid']")

        return self._auditor_loss(
            log_x1=batch["log_x"],
            log_x1_invalid=batch["log_x_invalid"],
            create_graph=True,
        )

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        if "log_x_invalid" not in batch:
            raise KeyError("BayesianAuditor requires batch['log_x_invalid']")

        return self._auditor_loss(
            log_x1=batch["log_x"],
            log_x1_invalid=batch["log_x_invalid"],
            create_graph=False,
        )

    @torch.no_grad()
    def ood_score(self, log_x: torch.Tensor) -> torch.Tensor:
        """Predictive variance as OOD score, shape (B,)."""
        was_training = self.training
        self.eval()
        dist = self.gp(self._extract(log_x))

        if was_training:
            self.train()

        return dist.variance

    @torch.no_grad()
    def score_per_sample(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-sample OOD score (shape ``(B,)``) — higher = more anomalous."""
        return self.ood_score(log_x)

    @torch.no_grad()
    def per_token_uq(self, log_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Per-token GP energy and epistemic variance, plus scalar aleatoric noise variance.

        Returns:
            energy: (B, L)
            variance: (B, L)
            noise_var: scalar float
        """
        was_training = self.training
        self.eval()
        b, seq_len, _k = log_x.shape
        h = self.backbone(log_x)
        z_tokens = self.latent_head.proj(h).reshape(b * seq_len, -1)
        dist = self.gp(z_tokens)
        if was_training:
            self.train()
        return (
            dist.mean.view(b, seq_len),
            dist.variance.view(b, seq_len),
            float(self.gp.noise_var.detach().cpu()),
        )

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        # If benchmark batch lacks invalid samples, still provide useful score.
        if "log_x_invalid" in batch:
            return self.eval_step(batch)

        ood = self.ood_score(batch["log_x"])

        return {"ood_var_mean": ood.mean(), "ood_var_std": ood.std(unbiased=False)}


@register("bayesian_auditor")
def build_bayesian_auditor(cfg: Config) -> BayesianAuditor:
    return BayesianAuditor(cfg)
