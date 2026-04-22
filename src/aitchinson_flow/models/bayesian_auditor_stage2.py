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
    * ``batch["log_x_invalid"]`` *(optional)* — real semantic negatives,
      shape ``(B, L, K-1)``. When present, replaces the randn fallback in
      the contrastive hinge.
    * ``batch["answer_mask"]`` *(optional)* — boolean mask of shape
      ``(B, L)`` marking answer-span positions. When present *and*
      ``cfg.gp.score_answer_tokens_only`` is ``True``, the token-level NLL
      and contrastive loss are restricted to masked positions (Path B).
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
from aitchinson_flow.models.bayesian_auditor_stage1 import (
    _build_llm_projection,
    _prepare_batch_with_projection,
)
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
        # Path B: learned embedding→simplex projection, frozen along with the
        # backbone. Stage 2 never trains this — it comes from Stage 1 via
        # ``compose_auditor_from_stages`` / ``_load_backbone_into_stage2``.
        self.llm_projection = _build_llm_projection(cfg)

        self._freeze_representations()

    def _freeze_representations(self) -> None:
        """Disable gradients for the backbone (and the Path B projection).

        ``latent_head`` and the full GP (including ``log_noise_var``) remain
        trainable so the token-level likelihood can shape both the projection
        and the observation-noise scale on top of fixed backbone features.
        The Path B ``llm_projection`` is trained in Stage 1 and treated as
        part of the frozen representation from Stage 2 onward.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        if self.llm_projection is not None:
            for p in self.llm_projection.parameters():
                p.requires_grad_(False)
            self.llm_projection.eval()

    def train(self, mode: bool = True) -> "BayesianAuditorStage2":
        """Override to keep the backbone (and llm_projection) in eval while training the head."""
        super().train(mode)
        self.backbone.eval()
        if self.llm_projection is not None:
            self.llm_projection.eval()
        return self

    def prepare_batch(self, batch: Any) -> Any:
        """Path B shim: convert ``embeddings`` → ``log_x`` via the frozen projection."""
        return _prepare_batch_with_projection(batch, self.llm_projection)

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

    def _token_likelihood_loss(
        self,
        log_x1: torch.Tensor,
        log_x_invalid: torch.Tensor | None = None,
        answer_mask: torch.Tensor | None = None,
    ) -> LossDict:
        B, L, _ = log_x1.shape

        # Valid tokens: maximise likelihood at y=0.
        z_valid_full = self._extract_tokens(log_x1)  # (B, L, d_latent)

        # Negatives: prefer real semantic negatives from the batch; fall back
        # to centered random noise (preserves pre-Phase-2 Path A behavior).
        if log_x_invalid is not None:
            if log_x_invalid.shape != log_x1.shape:
                raise ValueError(
                    f"log_x_invalid shape {tuple(log_x_invalid.shape)} must match "
                    f"log_x {tuple(log_x1.shape)}"
                )
            z_random_full = self._extract_tokens(log_x_invalid)
        else:
            with torch.no_grad():
                log_x_random = torch.randn_like(log_x1)
                log_x_random = log_x_random - log_x_random.mean(dim=-1, keepdim=True)
            z_random_full = self._extract_tokens(log_x_random)

        # Optional answer-span restriction (Path B): only score answer tokens.
        use_mask = self.cfg.gp.score_answer_tokens_only and answer_mask is not None
        if use_mask:
            assert answer_mask is not None  # for type checkers
            if answer_mask.shape != (B, L):
                raise ValueError(
                    f"answer_mask shape {tuple(answer_mask.shape)} must equal (B, L)=({B}, {L})"
                )
            mask_bool = answer_mask.to(dtype=torch.bool)
            if not mask_bool.any():
                raise ValueError("answer_mask is all-False across the batch; nothing to score")
            z_valid = z_valid_full[mask_bool]  # (N_mask, d_latent)
            z_random = z_random_full[mask_bool]
        else:
            z_valid = z_valid_full.reshape(B * L, -1)
            z_random = z_random_full.reshape(B * L, -1)

        dist_valid = self.gp(z_valid)
        nll = -_gp_log_prob(dist_valid, torch.zeros_like(dist_valid.mean))

        dist_random = self.gp(z_random)

        # Energy margin: negative tokens should have higher energy than valid.
        # Both sides receive gradients so the GP can simultaneously lower valid
        # energy and raise negative energy.
        energy_gap = self.cfg.gp.margin_E - (dist_random.mean - dist_valid.mean)
        contrastive = F.relu(energy_gap).mean()

        # Optional anchor term keeps valid energy near zero (prevents joint
        # drift now that dist_valid.mean is no longer detached). Disabled by
        # default (lambda_anchor=0.0) to preserve the pre-fix objective.
        anchor = dist_valid.mean.pow(2).mean()

        kl = self.gp.kl_divergence()
        noise_var = self.gp.noise_var

        # 1. Get the sequence-level normalizer from your robust utility
        # Returns N_sequences (or B if in a unit test)
        n_seq_norm = kl_normalizer(self, B)

        # 2. Scale it to the Token Level based on the active path
        if use_mask:
            # PATH B (Trivia Task): We are only scoring the Answer tokens.
            # We must multiply by the expected number of answer tokens per sequence.
            # You can add `avg_answer_len` to your config, or default to a reasonable estimate.
            avg_answer_len = getattr(self.cfg.dataset, "avg_answer_length", 10.0)
            n_norm = n_seq_norm * float(avg_answer_len)
        else:
            # PATH A (Normal Text): We are scoring every token in the sequence.
            n_norm = n_seq_norm * float(L)

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
        batch = self.prepare_batch(batch)
        if "log_x" not in batch:
            raise KeyError(
                "BayesianAuditorStage2.training_step requires batch['log_x'] "
                "(the valid-sample log-coordinates, shape (B, L, K-1)); got "
                f"keys {sorted(batch.keys()) if hasattr(batch, 'keys') else type(batch).__name__}"
            )
        return self._token_likelihood_loss(
            batch["log_x"],
            log_x_invalid=batch.get("log_x_invalid"),
            answer_mask=batch.get("answer_mask"),
        )

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        batch = self.prepare_batch(batch)
        if "log_x" not in batch:
            raise KeyError(
                "BayesianAuditorStage2.eval_step requires batch['log_x'] "
                "(the valid-sample log-coordinates, shape (B, L, K-1)); got "
                f"keys {sorted(batch.keys()) if hasattr(batch, 'keys') else type(batch).__name__}"
            )
        return self._token_likelihood_loss(
            batch["log_x"],
            log_x_invalid=batch.get("log_x_invalid"),
            answer_mask=batch.get("answer_mask"),
        )

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
    def ood_score_tokenwise(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-token GP epistemic variance, shape ``(B, L)``.

        Used by the Path B hallucination-audit task to compute answer-span
        averaged scores when ``cfg.gp.score_answer_tokens_only`` is set.
        """
        was_training = self.training
        self.eval()
        B, L, _ = log_x.shape
        z_tokens = self._extract_tokens(log_x).reshape(B * L, -1)
        dist = self.gp(z_tokens)
        if was_training:
            self.train()
        return dist.variance.view(B, L)

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
