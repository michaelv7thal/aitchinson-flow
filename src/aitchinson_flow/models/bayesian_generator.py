"""Bayesian generator: transformer latent extractor + SVGP energy + flow matching."""

from __future__ import annotations


import math
from typing import Any
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from aitchinson_flow.config import Config
from aitchinson_flow.geometry import volume_penalty
from aitchinson_flow.gp.gp import SparseGP
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.factory import register
from aitchinson_flow.transformer_backbone import LatentHead, TransformerBackbone
from aitchinson_flow.loss import build_velocity_loss


def _uniform_log_x0(B: int, L: int, K: int, device: torch.device) -> torch.Tensor:
    return torch.full((B, L, K), math.log(1.0 / K), device=device, dtype=torch.float32)


def _sdpa_math_ctx():
    return sdpa_kernel(SDPBackend.MATH)


def _corrupt_log_x0(log_x1: torch.Tensor, mode: str) -> torch.Tensor:
    B, L, K = log_x1.shape
    if mode == "missing":
        return torch.full_like(log_x1, math.log(1.0 / K))
    idx = torch.randint(K, (B, L), device=log_x1.device)
    one_hot = torch.zeros(B, L, K, device=log_x1.device, dtype=log_x1.dtype)
    one_hot.scatter_(-1, idx.unsqueeze(-1), 1.0)
    oh = one_hot + 1e-8
    return oh.log() - oh.log().logsumexp(-1, keepdim=True)


class BayesianGenerator(nn.Module):
    """Transformer → latent z, SVGP energy on z, velocity = −∇(E − vol_penalty); CFM-style loss."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self._velocity_loss_fn = build_velocity_loss(
            cfg.training.velocity_loss, soft_hilbert_alpha=cfg.training.soft_hilbert_alpha
        )

        self.backbone = TransformerBackbone(
            cfg=cfg, time_conditioned=False, sdp_math_for_autograd=True
        )
        self.latent_head = LatentHead(cfg=cfg)
        self.gp = SparseGP(cfg=cfg)

    def _extract(self, log_x: torch.Tensor) -> torch.Tensor:
        with _sdpa_math_ctx():
            h = self.backbone(log_x)
        return self.latent_head(h)

    def _velocity(self, log_xt: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
        B = log_xt.shape[0]
        z = self._extract(log_xt)
        dist = self.gp(z)
        vol = volume_penalty(
            log_xt.view(B, -1),
            alpha=self.cfg.bayesian_generator.volume_penalty_alpha,
            diag_approx=True,
        )
        energy = (dist.mean - vol).sum()
        (g,) = torch.autograd.grad(energy, log_xt, create_graph=create_graph)
        return -g

    def forward(self, log_x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Inference velocity (detached); t unused for this backbone."""
        log_x_g = log_x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = self._velocity(log_x_g, create_graph=False)
        return v.detach()

    def _flow_matching_loss(
        self,
        log_x1: torch.Tensor,
        *,
        create_graph: bool,
    ) -> LossDict:
        B, L, K = log_x1.shape
        bg = self.cfg.bayesian_generator

        if bg.corrupt_source_prob > 0.0 and torch.rand(1) < bg.corrupt_source_prob:
            log_x0 = _corrupt_log_x0(log_x1, bg.corrupt_mode)
        else:
            log_x0 = _uniform_log_x0(B, L, K, log_x1.device).to(dtype=log_x1.dtype)

        t = torch.rand(B, device=log_x1.device, dtype=log_x1.dtype)
        log_xt = (
            ((1 - t[:, None, None]) * log_x0 + t[:, None, None] * log_x1)
            .detach()
            .requires_grad_(True)
        )
        u_tgt = (log_x1 - log_x0).detach()

        v_pred = self._velocity(log_xt, create_graph=create_graph)
        kl = self.gp.kl_divergence()
        noise_var = self.gp.noise_var

        if bg.use_gp_variance_weighting:
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
        total = flow_loss + self.cfg.gp.lambda_kl * kl / B

        return {
            TRAINING_LOSS_KEY: total,
            "flow_loss": flow_loss,
            "kl": kl,
            "noise_var": noise_var.detach(),
        }

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        log_x1 = batch["log_x"]
        return self._flow_matching_loss(log_x1, create_graph=True)

    def eval_step(self, batch: Any) -> LossDict:
        log_x1 = batch["log_x"]
        return self._flow_matching_loss(log_x1, create_graph=False)

    def generate(self, n: int, steps: int | None = None) -> torch.Tensor:
        steps = steps if steps is not None else self.cfg.bayesian_generator.ode_steps
        device = self.cfg.training.device
        L, K = self.cfg.dataset.L, self.cfg.dataset.K
        self.eval()
        x = _uniform_log_x0(n, L, K, device).to(dtype=torch.float32)
        x = (x + self.cfg.bayesian_generator.ode_init_noise * torch.randn_like(x)).requires_grad_(
            True
        )
        dt = 1.0 / steps

        with torch.enable_grad():
            for _ in range(steps):
                v = self._velocity(x, create_graph=False)
                x = (x + v * dt).detach().requires_grad_(True)
        self.train()
        return x.detach().softmax(dim=-1).argmax(dim=-1)

    @torch.no_grad()
    def generate_with_uq(
        self, n: int, steps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = steps if steps is not None else self.cfg.bayesian_generator.ode_steps
        device = self.cfg.training.device
        L, K = self.cfg.dataset.L, self.cfg.dataset.K
        self.eval()
        x = _uniform_log_x0(n, L, K, device)
        x = x + self.cfg.bayesian_generator.ode_init_noise * torch.randn_like(x)
        dt = 1.0 / steps
        energy_traj: list[torch.Tensor] = []
        var_traj: list[torch.Tensor] = []

        def snapshot(log_xt: torch.Tensor) -> None:
            z = self._extract(log_xt)
            dist = self.gp(z)
            energy_traj.append(dist.mean.cpu())
            var_traj.append(dist.variance.cpu())

        snapshot(x)
        for _ in range(steps):
            x_g = x.detach().requires_grad_(True)
            with torch.enable_grad(), _sdpa_math_ctx():
                z = self._extract(x_g)
                dist_v = self.gp(z)
                vol = volume_penalty(
                    x_g.view(n, -1),
                    alpha=self.cfg.bayesian_generator.volume_penalty_alpha,
                    diag_approx=True,
                )
                energy = (dist_v.mean - vol).sum()
                (v,) = torch.autograd.grad(energy, x_g)
            x = (x_g + v * dt).detach()
            snapshot(x)
        self.train()
        tokens = x.softmax(dim=-1).argmax(dim=-1)
        return tokens, torch.stack(energy_traj), torch.stack(var_traj)


@register("bayesian_generator")
def build_bayesian_generator(cfg: Config) -> BayesianGenerator:
    return BayesianGenerator(cfg)
