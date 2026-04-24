from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict, kl_normalizer
from aitchinson_flow.models.bayesian_generator import (
    BayesianGenerator,
    _sdpa_math_ctx,
    _uniform_log_x0,
)
from aitchinson_flow.models.bayesian_auditor_stage1 import (
    _build_llm_projection,
    _prepare_batch_with_projection,
)
from aitchinson_flow.models.factory import register
from aitchinson_flow.config import Config
from aitchinson_flow.transformer_backbone import TokenLatentHead


logger = logging.getLogger(__name__)


_STAGE1_TRANSFER_PREFIXES: tuple[str, ...] = ("backbone.", "llm_projection.")
"""State-dict key prefixes copied from a Stage 1 checkpoint into `BayesianAuditor`.

Stage 1 also owns a `velocity_head.*` that is not part of the combined model,
so those keys are intentionally dropped at composition. The optional Path B
``llm_projection.*`` (trained in Stage 1, frozen in Stage 2) is copied through
alongside the backbone so the fused auditor reproduces Stage 1's simplex
features.
"""

_STAGE2_TRANSFER_PREFIXES: tuple[str, ...] = ("latent_head.", "gp.")
"""State-dict key prefixes copied from a Stage 2 checkpoint into `BayesianAuditor`.

Stage 2's `backbone.*` is ignored when a Stage 1 state dict is supplied; if
Stage 2 is used alone (ablation), its backbone is still copied — see
`compose_auditor_from_stages`.
"""


class BayesianAuditor(BayesianGenerator):
    """
    BayesianGenerator + contrastive energy/variance hinge losses.

    Expects batches with:
      - batch["log_x"]: valid samples, shape (B, L, D)
      - batch["log_x_invalid"]: invalid/OOD samples, shape (B, L, D)

    Overrides the parent's ``PooledLatentHead`` with a ``TokenLatentHead`` so
    per-token queries (``per_token_uq``, ``token_latents``) can project each
    position independently. The sequence-level ``_extract`` path is preserved
    by explicitly averaging the per-token latents, which is numerically
    identical to projecting a pooled hidden state.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.latent_head = TokenLatentHead(cfg=cfg)
        # Path B: optional learned embedding→simplex projection. Same freeze
        # semantics as in Stage 2 — the projection comes from the Stage 1
        # checkpoint at compose time and is not further trained here.
        self.llm_projection = _build_llm_projection(cfg)
        if self.llm_projection is not None:
            for p in self.llm_projection.parameters():
                p.requires_grad_(False)
            self.llm_projection.eval()

    def prepare_batch(self, batch: Any) -> Any:
        """Path B shim: convert ``embeddings`` → ``log_x`` via the frozen projection."""
        return _prepare_batch_with_projection(batch, self.llm_projection)

    def _extract(self, log_x: torch.Tensor) -> torch.Tensor:
        """Sequence-level latent ``(B, d_latent)`` via per-token head + mean-pool."""
        with _sdpa_math_ctx():
            h = self.backbone(log_x)
        z_tokens = self.latent_head(h)
        return z_tokens.mean(dim=1)

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

        # 1. Anchor valid energy near zero — prevents joint drift
        anchor_loss = valid_dist.mean.pow(2).mean()

        # Contrastive energy margin
        energy_gap = self.cfg.gp.margin_E - (invalid_dist.mean - valid_dist.mean)
        mean_loss = F.relu(energy_gap).mean()

        kl = self.gp.kl_divergence()
        n_norm = kl_normalizer(self, B)
        kl_norm = kl / n_norm

        total = (
            flow_loss
            + mean_loss
            + self.cfg.gp.lambda_var * anchor_loss
            + self.cfg.gp.lambda_kl * kl_norm
        )

        return {
            TRAINING_LOSS_KEY: total,
            "flow_loss": flow_loss,
            "mean_loss": mean_loss,
            # "var_loss": var_loss,
            "kl": kl,
            "kl_norm": kl_norm,
            "noise_var": noise_var.detach(),
        }

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        batch = self.prepare_batch(batch)
        if "log_x_invalid" not in batch:
            raise KeyError("BayesianAuditor requires batch['log_x_invalid']")

        return self._auditor_loss(
            log_x1=batch["log_x"],
            log_x1_invalid=batch["log_x_invalid"],
            create_graph=True,
        )

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        batch = self.prepare_batch(batch)
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

    def residual_score(self, log_x: torch.Tensor) -> torch.Tensor:
        """Flow-based sequence-level UQ — norm of the GP-gradient velocity.

        This is the combined-model counterpart to `BayesianAuditorStage1.residual_score`.
        Where Stage 1 uses its direct velocity head (from EqM + Hilbert training),
        the composed auditor uses ``v(log_x) = -∇(E(log_x) - vol(log_x))`` (see
        `_velocity`) and returns ``||v||_2`` per sample. Exposing this signal here
        lets benchmarks compare flow-residual UQ against GP-variance UQ from a
        single unified auditor without carrying a separate Stage 1 instance.

        Args:
            log_x: (B, L, D) log-simplex sequences.

        Returns:
            (B,) tensor of L2-norms of the velocity flattened over (L, D).
        """
        was_training = self.training
        self.eval()
        v = self.forward(log_x)
        score = v.reshape(v.shape[0], -1).norm(dim=1).detach()
        if was_training:
            self.train()
        return score

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
        z_tokens = self.latent_head(h)
        assert z_tokens.shape[:2] == (b, seq_len), (
            f"TokenLatentHead must preserve (B, L) dims; got {tuple(z_tokens.shape)}"
        )
        z_tokens = z_tokens.reshape(b * seq_len, -1)
        dist = self.gp(z_tokens)
        if was_training:
            self.train()
        return (
            dist.mean.view(b, seq_len),
            dist.variance.view(b, seq_len),
            float(self.gp.noise_var.detach().cpu()),
        )

    @torch.no_grad()
    def token_latents(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-token GP input latents, shape ``(B, L, d_latent)``.

        Matches the projection used by `per_token_uq` so plots of the latent
        cloud are directly comparable to the GP mean/variance channels.
        """
        was_training = self.training
        self.eval()
        b, seq_len, _k = log_x.shape
        h = self.backbone(log_x)
        z_tokens = self.latent_head(h)
        assert z_tokens.shape[:2] == (b, seq_len), (
            f"TokenLatentHead must preserve (B, L) dims; got {tuple(z_tokens.shape)}"
        )
        if was_training:
            self.train()
        return z_tokens.detach()

    @torch.no_grad()
    def inducing_points(self) -> torch.Tensor:
        """GP inducing locations, shape ``(M, d_latent)``."""
        return self.gp.Z.detach()

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        batch = self.prepare_batch(batch)
        # If benchmark batch lacks invalid samples, still provide useful score.
        if "log_x_invalid" in batch:
            return self.eval_step(batch)

        ood = self.ood_score(batch["log_x"])

        return {"ood_var_mean": ood.mean(), "ood_var_std": ood.std(unbiased=False)}


@register("bayesian_auditor")
def build_bayesian_auditor(cfg: Config) -> BayesianAuditor:
    return BayesianAuditor(cfg)


def _filter_prefixes(
    state: Mapping[str, torch.Tensor], prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if any(k.startswith(p) for p in prefixes)}


def compose_auditor_from_stages(
    cfg: Config,
    *,
    stage1_state: Mapping[str, torch.Tensor] | None = None,
    stage2_state: Mapping[str, torch.Tensor] | None = None,
    strict: bool = False,
) -> BayesianAuditor:
    """Build a `BayesianAuditor` by fusing Stage 1 backbone + Stage 2 GP head weights.

    The auditor starts from a fresh random initialization, then keys matching
    known prefixes are overwritten from the provided state dicts:

    * Stage 1 → `backbone.*` (its `velocity_head.*` is intentionally dropped).
    * Stage 2 → `latent_head.*`, `gp.*` (its `backbone.*` is skipped when
      Stage 1 is provided, otherwise used as a random-backbone ablation).

    Args:
        cfg: Config used to construct the target `BayesianAuditor`.
        stage1_state: State dict produced by `BayesianAuditorStage1`, or None.
        stage2_state: State dict produced by `BayesianAuditorStage2`, or None.
        strict: If True, raises on unexpected/missing target keys after fusion.

    Returns:
        A `BayesianAuditor` ready for inference. Absent stages fall back to
        the fresh random init — useful for quick ablations.
    """
    auditor = BayesianAuditor(cfg)
    target = dict(auditor.state_dict())
    applied: dict[str, int] = {"stage1": 0, "stage2": 0, "stage2_backbone_fallback": 0}

    if stage1_state is not None:
        for k, v in _filter_prefixes(stage1_state, _STAGE1_TRANSFER_PREFIXES).items():
            if k in target:
                target[k] = v
                applied["stage1"] += 1

    if stage2_state is not None:
        s2_head = _filter_prefixes(stage2_state, _STAGE2_TRANSFER_PREFIXES)
        for k, v in s2_head.items():
            if k in target:
                target[k] = v
                applied["stage2"] += 1

        if stage1_state is None:
            s2_bb = _filter_prefixes(stage2_state, ("backbone.",))
            for k, v in s2_bb.items():
                if k in target:
                    target[k] = v
                    applied["stage2_backbone_fallback"] += 1

    missing_keys, unexpected_keys = auditor.load_state_dict(target, strict=False)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Strict composition failed: missing={list(missing_keys)}, "
            f"unexpected={list(unexpected_keys)}"
        )
    if missing_keys:
        logger.warning(
            "compose_auditor_from_stages: %d key(s) remained at random init: %s",
            len(missing_keys),
            sorted(missing_keys),
        )
    if unexpected_keys:
        logger.warning(
            "compose_auditor_from_stages: %d unexpected key(s) ignored: %s",
            len(unexpected_keys),
            sorted(unexpected_keys),
        )
    logger.info(
        "compose_auditor_from_stages: loaded stage1_backbone=%d, stage2_head=%d, "
        "stage2_backbone_fallback=%d keys",
        applied["stage1"],
        applied["stage2"],
        applied["stage2_backbone_fallback"],
    )
    return auditor


def load_auditor_from_stage_checkpoints(
    cfg: Config,
    *,
    stage1_ckpt: str | Path | None = None,
    stage2_ckpt: str | Path | None = None,
    map_location: str | torch.device | None = None,
    strict: bool = False,
) -> BayesianAuditor:
    """File-based convenience wrapper around `compose_auditor_from_stages`.

    Loads `torch.save` checkpoint files written by the training runner
    (`{"model_state_dict": ..., ...}`) and composes them into a single
    `BayesianAuditor`. Either path may be omitted, in which case the
    corresponding weight group is left at random init.
    """

    def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, Mapping):
            raise ValueError(f"Checkpoint at {path} did not contain a state-dict mapping.")
        return dict(state)

    s1 = _load_state(stage1_ckpt) if stage1_ckpt is not None else None
    s2 = _load_state(stage2_ckpt) if stage2_ckpt is not None else None
    return compose_auditor_from_stages(cfg, stage1_state=s1, stage2_state=s2, strict=strict)
