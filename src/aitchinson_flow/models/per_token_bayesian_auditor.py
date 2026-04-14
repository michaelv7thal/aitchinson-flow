"""Per-token Bayesian auditor: per-position latents + SVGP (optionally product-kernel with LM context)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


from aitchinson_flow.config import Config
from aitchinson_flow.geometry import volume_penalty
from aitchinson_flow.gp.gp import GPOutput, ProductSparseGP
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.bayesian_auditor import BayesianAuditor
from aitchinson_flow.models.bayesian_generator import _sdpa_math_ctx, _uniform_log_x0
from aitchinson_flow.models.factory import register


# Default last-hidden size for GPT-2; override via Config.per_token_auditor.ctx_hidden
GPT2_HIDDEN_DIM = 768


class PerTokenBayesianAuditor(BayesianAuditor):
    """
    BayesianAuditor with per-token (per-position) GP evaluation.

    Replaces global `LatentHead` with `pos_proj` on backbone hidden states.
    Optionally uses `ProductSparseGP` when `use_context=True` and batch provides
    `ctx_1` / `ctx_1_invalid` aligned with `log_x` / `log_x_invalid` (shape ``(B, L, H)``).
    """

    def __init__(
        self, cfg: Config, *, use_context: bool = False, ctx_hidden: int = GPT2_HIDDEN_DIM
    ) -> None:
        super().__init__(cfg)
        self.use_context = use_context

        d_model = cfg.transformer.d_model
        d_latent = cfg.transformer.d_latent
        m = cfg.gp.num_inducing

        # Parent constructed `latent_head`; we use per-token projection instead.
        del self.latent_head

        self.pos_proj = nn.Linear(d_model, d_latent)

        if use_context:
            self.gp = ProductSparseGP(d_latent, d_latent, M=m)
            self.ctx_proj = nn.Linear(ctx_hidden, d_latent)
        # else: keep SparseGP from BayesianGenerator / BayesianAuditor path (already `self.gp`)

    def _extract_per_token(self, log_x: torch.Tensor) -> torch.Tensor:
        """log_x (B, L, K) → z (B, L, d_latent) — GP inputs (before flatten)."""
        with _sdpa_math_ctx():
            h = self.backbone(log_x)
        return self.pos_proj(h)

    def _extract(self, log_x: torch.Tensor) -> torch.Tensor:
        """Global mean over positions — used by `generate` / `generate_with_uq` in parent."""
        return self._extract_per_token(log_x).mean(dim=1)

    def _gp_forward(self, z: torch.Tensor, h: torch.Tensor | None = None) -> GPOutput:
        """z: (N, d_latent); if product GP, h: (N, d_latent) aligned row-wise."""
        if self.use_context:
            if h is None:
                raise ValueError("PerTokenBayesianAuditor: context required when use_context=True")
            return self.gp(z, h)
        return self.gp(z)

    def _velocity(
        self,
        log_xt: torch.Tensor,
        *,
        create_graph: bool,
        ctx_1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """−∇ E(log_xt) with per-token energy summed over positions."""
        b, seq_len, _k = log_xt.shape
        z_pt = self._extract_per_token(log_xt)
        z_flat = z_pt.reshape(b * seq_len, -1)
        if self.use_context:
            if ctx_1 is None:
                raise ValueError("ctx_1 is required when use_context=True")
            # Teacher activations: no grad through HF backbone for energy (matches typical auditor).
            h_flat = self.ctx_proj(ctx_1.to(log_xt.device).float()).reshape(b * seq_len, -1).detach()
            dist = self._gp_forward(z_flat, h_flat)
        else:
            dist = self._gp_forward(z_flat, None)
        energy_j = dist.mean.view(b, seq_len)
        vol = volume_penalty(
            log_xt.view(b, -1),
            alpha=self.cfg.bayesian_generator.volume_penalty_alpha,
            diag_approx=True,
        )
        energy = (energy_j.sum(dim=1) - vol).sum()
        (g,) = torch.autograd.grad(energy, log_xt, create_graph=create_graph)
        return -g

    def _auditor_loss(
        self,
        log_x1: torch.Tensor,
        log_x1_invalid: torch.Tensor,
        *,
        create_graph: bool,
        ctx_1: torch.Tensor | None = None,
        ctx_1_invalid: torch.Tensor | None = None,
    ) -> LossDict:
        b, seq_len, k = log_x1.shape
        log_x0 = _uniform_log_x0(b, seq_len, k, log_x1.device).to(dtype=log_x1.dtype)
        t = torch.rand(b, device=log_x1.device, dtype=log_x1.dtype)
        log_xt = (
            ((1.0 - t[:, None, None]) * log_x0 + t[:, None, None] * log_x1)
            .detach()
            .requires_grad_(True)
        )
        u_tgt = (log_x1 - log_x0).detach()
        v_ctx = ctx_1 if self.use_context else None
        v_pred = self._velocity(log_xt, create_graph=create_graph, ctx_1=v_ctx)
        noise_var = self.gp.noise_var
        # GP variance weighting (coarse per-token aggregate → scalar weights over batch)
        if self.cfg.bayesian_generator.use_gp_variance_weighting:
            with torch.no_grad():
                z_t = self._extract_per_token(log_xt.detach()).reshape(b * seq_len, -1)
                if self.use_context and ctx_1 is not None:
                    h_t = self.ctx_proj(ctx_1.to(log_x1.device).float()).reshape(b * seq_len, -1).detach()
                    var_t = self._gp_forward(z_t, h_t).variance.reshape(b, seq_len)
                else:
                    var_t = self.gp(z_t).variance.reshape(b, seq_len)
                # Per-sequence scalar: mean epistemic variance over positions
                var_t = var_t.mean(dim=1)
            total_var = var_t + noise_var
            w = 1.0 / total_var
            w = w / w.mean()
            weighted_mse = (w[:, None, None] * (v_pred - u_tgt).pow(2)).mean()
        else:
            weighted_mse = self._velocity_loss_fn(v_pred, u_tgt)
        flow_loss = 0.5 * weighted_mse / noise_var + 0.5 * noise_var.log()
        z_v = self._extract_per_token(log_x1).reshape(b * seq_len, -1)
        z_i = self._extract_per_token(log_x1_invalid).reshape(b * seq_len, -1)
        if self.use_context:
            if ctx_1 is None or ctx_1_invalid is None:
                raise ValueError(
                    "use_context=True requires ctx_1 and ctx_1_invalid in batch / loss"
                )
            h_v = self.ctx_proj(ctx_1.to(log_x1.device).float()).reshape(b * seq_len, -1)
            h_i = self.ctx_proj(ctx_1_invalid.to(log_x1.device).float()).reshape(b * seq_len, -1)
            d_v = self._gp_forward(z_v, h_v)
            d_i = self._gp_forward(z_i, h_i)
        else:
            d_v = self.gp(z_v)
            d_i = self.gp(z_i)
        mean_loss = d_v.mean.pow(2).mean() + F.relu(self.cfg.gp.margin_E - d_i.mean).mean()
        var_loss = d_v.variance.mean() + F.relu(self.cfg.gp.margin_V - d_i.variance).mean()
        kl = self.gp.kl_divergence()
        total = (
            flow_loss
            + mean_loss
            + self.cfg.gp.lambda_var * var_loss
            + self.cfg.gp.lambda_kl * kl / b
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
            raise KeyError("PerTokenBayesianAuditor requires batch['log_x_invalid']")
        if self.use_context:
            if "ctx_1" not in batch or "ctx_1_invalid" not in batch:
                raise KeyError(
                    "PerTokenBayesianAuditor with use_context=True requires "
                    "batch['ctx_1'] and batch['ctx_1_invalid'], shape (B, L, H)"
                )
        return self._auditor_loss(
            log_x1=batch["log_x"],
            log_x1_invalid=batch["log_x_invalid"],
            create_graph=True,
            ctx_1=batch.get("ctx_1"),
            ctx_1_invalid=batch.get("ctx_1_invalid"),
        )

    def eval_step(self, batch: Any) -> LossDict:
        if "log_x_invalid" not in batch:
            raise KeyError("PerTokenBayesianAuditor requires batch['log_x_invalid']")
        if self.use_context and ("ctx_1" not in batch or "ctx_1_invalid" not in batch):
            raise KeyError(
                "PerTokenBayesianAuditor with use_context=True requires ctx_1 and ctx_1_invalid"
            )
        return self._auditor_loss(
            log_x1=batch["log_x"],
            log_x1_invalid=batch["log_x_invalid"],
            create_graph=False,
            ctx_1=batch.get("ctx_1"),
            ctx_1_invalid=batch.get("ctx_1_invalid"),
        )

    @torch.no_grad()
    def per_token_ood(
        self,
        log_x: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token GP mean and epistemic variance: each (B, L)."""
        was_training = self.training
        self.eval()
        b, seq_len, _k = log_x.shape
        z = self._extract_per_token(log_x).reshape(b * seq_len, -1)
        if self.use_context:
            if ctx is None:
                raise ValueError("ctx is required when use_context=True")
            h = self.ctx_proj(ctx.to(log_x.device).float()).reshape(b * seq_len, -1)
            dist = self._gp_forward(z, h)
        else:
            dist = self.gp(z)
        if was_training:
            self.train()
        return dist.mean.view(b, seq_len), dist.variance.view(b, seq_len)

    @torch.no_grad()
    def score_per_sample(
        self, log_x: torch.Tensor, ctx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-sample OOD score: mean epistemic variance over positions, shape ``(B,)``."""
        _, var = self.per_token_ood(log_x, ctx=ctx)
        return var.mean(dim=1)

    @torch.no_grad()
    def per_token_uq(
        self,
        log_x: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Energy (B, L), epistemic variance (B, L), scalar aleatoric σ²_noise."""
        epi_mean, epi_var = self.per_token_ood(log_x, ctx=ctx)
        return epi_mean, epi_var, float(self.gp.noise_var.detach().cpu())


@register("per_token_bayesian_auditor")
def build_per_token_bayesian_auditor(cfg: Config) -> PerTokenBayesianAuditor:
    p = getattr(cfg, "per_token_auditor", None)
    if p is None:
        return PerTokenBayesianAuditor(cfg)
    return PerTokenBayesianAuditor(
        cfg,
        use_context=p.use_context,
        ctx_hidden=p.ctx_hidden,
    )
