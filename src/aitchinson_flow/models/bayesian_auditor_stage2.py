"""Stage 2 of the two-stage Bayesian auditor: frozen-backbone token-level GP.

Purpose: on top of a backbone pretrained in Stage 1 (EqM + Hilbert), train a
token-level GP over the valid data manifold. The GP is fit by maximising the
token-level predictive likelihood at ``y = 0`` on valid sequences, with a
random-negative contrastive energy hinge and SVGP KL regularisation. The latent
projection (``latent_head``) is trained jointly
with the GP; only the ``backbone`` is frozen. Stage 2 will also run from a
random backbone for ablation purposes (the freeze rule is unchanged).

    Freeze policy:
    * ``backbone`` — ``requires_grad=False``, pinned to ``eval()`` by
      ``train()``, and forwarded under ``torch.no_grad`` so no autograd graph
      is built through it.
    * ``latent_head`` — trainable, follows ``train()``/``eval()`` mode.
    * ``gp`` — fully trainable, including the homoscedastic aleatoric noise
      (``gp.log_noise_var``), so the GP can learn the appropriate observation
      noise scale under the token-level likelihood.

Batch contract:
    * ``batch["log_x"]`` — valid sequences, shape ``(B, L, K-1)``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from aitchinson_flow.config import Config
from aitchinson_flow.gp.gp import GPOutput, SparseGP
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict, kl_normalizer
from aitchinson_flow.models.factory import register
from aitchinson_flow.transformer_backbone import TokenLatentHead, TransformerBackbone


def _gp_log_prob(dist: GPOutput, y: torch.Tensor) -> torch.Tensor:
    """Gaussian log-likelihood of ``y`` under the GP predictive distribution.

    The predictive is ``N(mean, variance + noise_var)`` with epistemic
    ``variance`` from the SVGP posterior and homoscedastic aleatoric
    ``noise_var``. Returns a tensor matching ``y``'s shape (elementwise).
    """
    noise_var = dist.noise_var if dist.noise_var is not None else y.new_zeros(())
    total_var = dist.variance + noise_var
    residual = y - dist.mean
    return -0.5 * (math.log(2.0 * math.pi) + total_var.log() + residual.pow(2) / total_var)


class BayesianAuditorStage2(nn.Module):
    """Frozen backbone + token-level GP likelihood head (Stage 2).

    Architecture matches `BayesianAuditor`'s representation/GP path exactly so
    that Stage 1 backbone weights and Stage 2 GP weights can be state-dict
    composed into a single `BayesianAuditor` at inference time.

    Training objective (token-level, valid vs random negatives):
        ``z       = latent_head.proj(backbone(log_x))``  shape ``(B, L, d_latent)``
        ``dist_v  = gp(z.reshape(B*L, d_latent))``               (valid tokens)
        ``nll     = -log p(y = 0 | x_valid)``                    (per-token Gaussian NLL)
        ``x_rand  = centered randn_like(log_x)``                 (random negatives)
        ``dist_r  = gp(z_rand.reshape(B*L, d_latent))``
        ``contrastive = relu(margin_E - (mean_r - mean_v)).mean()``
        ``kl      = gp.kl_divergence()``
        ``total   = nll.mean() + lambda_contrastive * contrastive + lambda_kl * kl / (N * L)``

    Flow matching, velocity backprop, Hilbert penalties, and volume penalties
    are intentionally absent in Stage 2. The frozen backbone receives no
    gradient updates; ``latent_head`` and all GP parameters (including the
    aleatoric noise) are updated jointly so the latent projection and GP are
    co-adapted to the valid-data likelihood with random-negative contrastive
    separation in GP energy.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = TransformerBackbone(
            cfg=cfg, time_conditioned=False, sdp_math_for_autograd=True
        )
        self.latent_head = TokenLatentHead(cfg=cfg)
        self.gp = SparseGP(cfg=cfg)

        self._freeze_representations()

    def _freeze_representations(self) -> None:
        """Disable gradients for the backbone only.

        ``latent_head`` and the full GP (including ``log_noise_var``) remain
        trainable so the token-level likelihood can shape both the projection
        and the observation-noise scale on top of fixed backbone features.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "BayesianAuditorStage2":
        """Override to keep the backbone in eval even when training the head."""
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Explicit list of parameters that should be handed to an optimizer.

        ``backbone.*`` is frozen and intentionally omitted. All ``latent_head``
        and GP parameters (including ``gp.log_noise_var``) are optimized.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def _extract_tokens(self, log_x: torch.Tensor) -> torch.Tensor:
        """Token-level representation: ``log_x`` (B, L, K-1) → ``z`` (B, L, d_latent).

        The backbone runs under ``torch.no_grad`` (it is frozen), but the
        ``TokenLatentHead`` projection is differentiable so its parameters
        get a gradient from the token-level likelihood.
        """
        B, L, _ = log_x.shape
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            h = self.backbone(log_x)
        z = self.latent_head(h)
        assert z.shape[:2] == (B, L), (
            f"TokenLatentHead must preserve (B, L) dims; got {tuple(z.shape)}"
        )
        return z

    def _token_likelihood_loss(self, log_x1: torch.Tensor) -> LossDict:
        B, L, _ = log_x1.shape

        # Valid tokens: maximise likelihood at y=0.
        z_valid = self._extract_tokens(log_x1).reshape(B * L, -1)
        dist_valid = self.gp(z_valid)
        nll = -_gp_log_prob(dist_valid, torch.zeros_like(dist_valid.mean))

        # Random negatives: centered in log-space to match simplex tangent space.
        with torch.no_grad():
            log_x_random = torch.randn_like(log_x1)
            log_x_random = log_x_random - log_x_random.mean(dim=-1, keepdim=True)
        z_random = self._extract_tokens(log_x_random).reshape(B * L, -1)
        dist_random = self.gp(z_random)

        # Energy margin: random tokens should have higher energy than valid.
        # Both sides receive gradients so the GP can simultaneously lower valid
        # energy and raise random-negative energy.
        energy_gap = self.cfg.gp.margin_E - (dist_random.mean - dist_valid.mean)
        contrastive = F.relu(energy_gap).mean()

        # Optional anchor term keeps valid energy near zero (prevents joint
        # drift now that dist_valid.mean is no longer detached). Disabled by
        # default (lambda_anchor=0.0) to preserve the pre-fix objective.
        anchor = dist_valid.mean.pow(2).mean()

        kl = self.gp.kl_divergence()
        noise_var = self.gp.noise_var

        # Per-token normalisation: nll is already averaged over B*L; divide KL
        # by the same B*L so both terms are on the same per-token ELBO scale.
        # lambda_kl then has a direct interpretation as regularisation weight
        # relative to the likelihood, independent of batch size or dataset size.
        n_norm = float(B * max(1, L))

        # n_norm = kl_normalizer(self, B) * max(1, L)

        total = (
            nll.mean()
            + self.cfg.gp.lambda_contrastive * contrastive
            + self.cfg.gp.lambda_anchor * anchor
            + self.cfg.gp.lambda_kl * kl / n_norm
        )
        return {
            TRAINING_LOSS_KEY: total,
            "nll": nll.mean().detach(),
            "contrastive": contrastive.detach(),
            "anchor": anchor.detach(),
            "kl": kl.detach(),
            "noise_var": noise_var.detach(),
        }

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        if "log_x" not in batch:
            raise KeyError(
                "BayesianAuditorStage2.training_step requires batch['log_x'] "
                "(the valid-sample log-coordinates, shape (B, L, K-1)); got "
                f"keys {sorted(batch.keys()) if hasattr(batch, 'keys') else type(batch).__name__}"
            )
        return self._token_likelihood_loss(batch["log_x"])

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        if "log_x" not in batch:
            raise KeyError(
                "BayesianAuditorStage2.eval_step requires batch['log_x'] "
                "(the valid-sample log-coordinates, shape (B, L, K-1)); got "
                f"keys {sorted(batch.keys()) if hasattr(batch, 'keys') else type(batch).__name__}"
            )
        return self._token_likelihood_loss(batch["log_x"])

    @torch.no_grad()
    def ood_score(self, log_x: torch.Tensor) -> torch.Tensor:
        """Sequence-level OOD score: mean token epistemic variance, shape ``(B,)``."""
        was_training = self.training
        self.eval()
        B, L, _ = log_x.shape
        z_tokens = self._extract_tokens(log_x).reshape(B * L, -1)
        dist = self.gp(z_tokens)
        if was_training:
            self.train()
        return dist.variance.view(B, L).mean(dim=1)

    @torch.no_grad()
    def score_per_sample(self, log_x: torch.Tensor) -> torch.Tensor:
        return self.ood_score(log_x)

    @torch.no_grad()
    def per_token_uq(self, log_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Per-token GP energy and epistemic variance, plus scalar aleatoric noise.

        The ``@torch.no_grad`` decorator is sufficient here — neither the
        frozen ``backbone`` nor the trainable ``latent_head`` build an autograd
        graph in this scoring path.
        """
        was_training = self.training
        self.eval()
        B, L, _ = log_x.shape
        z_tokens = self._extract_tokens(log_x).reshape(B * L, -1)
        dist = self.gp(z_tokens)
        if was_training:
            self.train()
        return (
            dist.mean.view(B, L),
            dist.variance.view(B, L),
            float(self.gp.noise_var.detach().cpu()),
        )

    @torch.no_grad()
    def token_latents(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-token GP input latents, shape ``(B, L, d_latent)``.

        Same projection used by the token-level likelihood and
        `per_token_uq`; returned detached on the input device for plotting.
        """
        was_training = self.training
        self.eval()
        B, L, _ = log_x.shape
        z_tokens = self._extract_tokens(log_x).reshape(B, L, -1)
        if was_training:
            self.train()
        return z_tokens.detach()

    @torch.no_grad()
    def inducing_points(self) -> torch.Tensor:
        """GP inducing locations, shape ``(M, d_latent)``."""
        return self.gp.Z.detach()

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        return self.eval_step(batch)


@register("bayesian_auditor_stage2")
def build_bayesian_auditor_stage2(cfg: Config) -> BayesianAuditorStage2:
    return BayesianAuditorStage2(cfg)
