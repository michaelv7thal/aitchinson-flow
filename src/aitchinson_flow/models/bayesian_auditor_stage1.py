"""Stage 1 of the two-stage Bayesian auditor: EqM + Hilbert backbone training.

Purpose: train the backbone to learn the *valid manifold geometry* via
Equilibrium Matching with a Hilbert-family velocity loss. No GP, no
contrastive terms, no invalid sequences — just random sequences → valid
sequences (random-to-data transport in the log-simplex).

    The resulting backbone is then frozen and composed with a Stage 2 GP head
for combined inference (see `BayesianAuditor` and `compose_auditor_from_stages`).

Notes:
    * Mirrors the backbone architecture of `BayesianAuditor` so weights are
      state-dict-compatible at composition time.
    * Exposes `energy_score(log_x) = -d_H(f(x), x)` — the geometric
      energy implied by the EqM + Hilbert objective. The companion
      `ood_score` returns ``d_H`` (anomaly convention: higher = more
      OOD) and `score_per_sample` is wired to it so the benchmark
      ``auroc_auditor`` reflects the geometric energy, not a velocity norm.
    * `residual_score` (norm of the predicted velocity) is retained for
      backward compatibility and as a complementary flow-residual signal.
    * The Hilbert-family loss is enforced at construction time by default;
      set `cfg.training.velocity_loss` to one of {"soft_hilbert",
      "hard_hilbert", "clr_mse", "ilr_mse"}.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aitchinson_flow.config import Config
from aitchinson_flow.data.transforms.discrete import token_ids_to_features
from aitchinson_flow.data.feature_dim import feature_dim
from aitchinson_flow.geometry import nielsen_soft_hilbert_distance
from aitchinson_flow.loss import build_velocity_loss
from aitchinson_flow.models.base import TRAINING_LOSS_KEY, LossDict
from aitchinson_flow.models.factory import register
from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead


_HILBERT_FAMILY = {"soft_hilbert", "hard_hilbert", "clr_mse", "ilr_mse"}
_CGAMMA_STRATEGIES = {"legacy", "linear", "truncated", "piecewise"}


def _uniform_log_x0(
    B: int, L: int, D: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Uniform log-simplex source: zeros in log-space (constant after centering)."""
    return torch.zeros((B, L, D), device=device, dtype=dtype)


def _scrambled_log_x0(
    cfg: Config,
    *,
    bsz: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a stochastic source from random vocabulary tokens (gibberish ``x_0``).

    Source ids are sampled uniformly from ``[0, K)`` and converted with the same
    discrete→feature transform used by the data pipeline (label smoothing +
    ILR/CLR), so Stage 1 noise stays tied to the configured vocabulary.
    """
    k = cfg.dataset.K
    token_ids = torch.randint(0, k, (bsz, seq_len), device=device)
    rows = [
        token_ids_to_features(
            token_ids[i],
            k,
            eps=cfg.hf_dataset.log_simplex_eps,
            label_smoothing=cfg.hf_dataset.label_smoothing,
            transform_mode=cfg.hf_dataset.transform_mode,
        )
        for i in range(bsz)
    ]
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


class BayesianAuditorStage1(nn.Module):
    """EqM + Hilbert flow-matching backbone (Stage 1 of two-stage auditor).

    Architecture:
        ``backbone`` (shared with `BayesianAuditor`, time-independent) +
        ``velocity_head`` (direct velocity prediction, dropped at composition).

    Training objective:
        EqM target ``u_tgt = c(gamma) * (log_x0 - log_x1)`` on the linear
        log-space path ``log_x_gamma = (1 - gamma) log_x0 + gamma log_x1``,
        fit with the selected Hilbert-family velocity loss (default: soft
        Hilbert).

        Sign convention (data → noise): the target velocity points from data
        toward noise (``log_x0 - log_x1``), which is the negative of the path
        tangent ``d log_x_gamma / d gamma = log_x1 - log_x0``. Inference
        integrators therefore subtract the predicted velocity to flow from
        noise toward the data manifold — see ``integrate``/``generate`` which
        update ``x ← x - v(x) * dt``.

        This convention matches ``EquilibriumAuditor`` and
        ``BayesianGenerator`` for state-dict-compatible composition.

    Batch contract:
        ``batch["log_x"]`` — valid log-simplex targets, shape ``(B, L, K-1)``.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        eq = cfg.equilibrium
        if not (0.0 <= eq.eqm_interp < 1.0):
            raise ValueError(f"equilibrium.eqm_interp must be in [0, 1), got {eq.eqm_interp}")
        self._scale = 1.0 / (1.0 - eq.eqm_interp)
        self._validate_c_gamma_config()

        vname = cfg.training.velocity_loss
        if vname not in _HILBERT_FAMILY:
            raise ValueError(
                f"BayesianAuditorStage1 expects a Hilbert-family velocity loss "
                f"(one of {sorted(_HILBERT_FAMILY)}), got {vname!r}. "
                "Set cfg.training.velocity_loss to 'soft_hilbert' (default) for EqM+Hilbert."
            )

        # Aligned with Stage 2 / BayesianAuditor: same ``time_conditioned`` flag so
        # ``compose_auditor_from_stages`` loads ``backbone.*`` without shape/key skew.
        self.backbone = TransformerBackbone(
            cfg=cfg, time_conditioned=cfg.transformer.time_conditioned
        )
        self.velocity_head = VelocityHead(cfg=cfg)
        self._velocity_loss_fn = build_velocity_loss(
            vname, soft_hilbert_alpha=cfg.training.soft_hilbert_alpha
        )

    def forward(self, log_x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """``log_x`` (B, L, D) → velocity (B, L, D). ``t`` is ignored (time-independent field)."""
        del t
        return self.velocity_head(self.backbone(log_x))

    def _c_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        """EqM schedule c(gamma), configurable via ``cfg.equilibrium``.

        Supported strategies:
            - legacy:   min(eqm_start, scale * (1-gamma)) * 4
            - linear:   1 - gamma
            - truncated: 1 if gamma <= a, else (1-gamma)/(1-a)
            - piecewise: b - ((b-1)/a)*gamma if gamma <= a else (1-gamma)/(1-a)

        A global gradient multiplier is applied to any strategy:
            c <- c / eqm_gradient_lambda

        Returns:
            Tensor of shape (B, 1, 1) for broadcasting over (B, L, D).
        """
        eq = self.cfg.equilibrium
        strategy = eq.eqm_decay_strategy.lower()
        one_minus_gamma = 1.0 - gamma

        if strategy == "legacy":
            cap = torch.as_tensor(eq.eqm_start, device=gamma.device, dtype=gamma.dtype)
            c_gamma = torch.clamp(torch.minimum(cap, self._scale * one_minus_gamma), min=0.0) * 4.0
        elif strategy == "linear":
            c_gamma = one_minus_gamma
        elif strategy == "truncated":
            a = torch.as_tensor(eq.eqm_decay_a, device=gamma.device, dtype=gamma.dtype)
            c_gamma = torch.where(
                gamma <= a,
                torch.ones_like(gamma),
                one_minus_gamma / (1.0 - a),
            )
        elif strategy == "piecewise":
            a = torch.as_tensor(eq.eqm_decay_a, device=gamma.device, dtype=gamma.dtype)
            b = torch.as_tensor(eq.eqm_decay_b, device=gamma.device, dtype=gamma.dtype)
            left = b - ((b - 1.0) / a) * gamma
            right = one_minus_gamma / (1.0 - a)
            c_gamma = torch.where(gamma <= a, left, right)
        else:
            raise ValueError(
                f"Unknown equilibrium.eqm_decay_strategy={eq.eqm_decay_strategy!r}; "
                f"expected one of {sorted(_CGAMMA_STRATEGIES)}."
            )

        c_gamma = c_gamma / torch.as_tensor(
            eq.eqm_gradient_lambda, device=gamma.device, dtype=gamma.dtype
        )
        return c_gamma[:, None, None]

    def _validate_c_gamma_config(self) -> None:
        eq = self.cfg.equilibrium
        strategy = eq.eqm_decay_strategy.lower()
        if strategy not in _CGAMMA_STRATEGIES:
            raise ValueError(
                f"equilibrium.eqm_decay_strategy must be one of {sorted(_CGAMMA_STRATEGIES)}, "
                f"got {eq.eqm_decay_strategy!r}"
            )
        if eq.eqm_gradient_lambda <= 0.0:
            raise ValueError(
                f"equilibrium.eqm_gradient_lambda must be > 0, got {eq.eqm_gradient_lambda}"
            )
        if strategy == "truncated" and not (0.0 <= eq.eqm_decay_a < 1.0):
            raise ValueError(
                f"equilibrium.eqm_decay_a must be in [0, 1) for truncated decay, got {eq.eqm_decay_a}"
            )
        if strategy == "piecewise":
            if not (0.0 < eq.eqm_decay_a < 1.0):
                raise ValueError(
                    f"equilibrium.eqm_decay_a must be in (0, 1) for piecewise decay, got {eq.eqm_decay_a}"
                )
            if eq.eqm_decay_b < 1.0:
                raise ValueError(
                    f"equilibrium.eqm_decay_b must be >= 1 for piecewise decay, got {eq.eqm_decay_b}"
                )

    def _eqm_hilbert_loss(self, log_x1: torch.Tensor) -> LossDict:
        B, L, D = log_x1.shape
        device, dt = log_x1.device, log_x1.dtype
        log_x0 = _scrambled_log_x0(self.cfg, bsz=B, seq_len=L, device=device, dtype=dt)
        gamma = torch.rand(B, device=device, dtype=dt)
        log_x_gamma = (1.0 - gamma[:, None, None]) * log_x0 + gamma[:, None, None] * log_x1
        # Sign convention: target velocity points data → noise (log_x0 - log_x1).
        # The matching integrator subtracts v: x ← x - v(x) * dt (noise → data).
        u_tgt = self._c_gamma(gamma) * (log_x0 - log_x1)
        v_pred = self.forward(log_x_gamma)
        loss = self._velocity_loss_fn(v_pred, u_tgt)
        return {
            TRAINING_LOSS_KEY: loss,
            "flow_loss": loss.detach(),
            "velocity_loss": loss.detach(),
        }

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        if "log_x" not in batch:
            raise KeyError("BayesianAuditorStage1 requires batch['log_x']")
        return self._eqm_hilbert_loss(batch["log_x"])

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        return self._eqm_hilbert_loss(batch["log_x"])

    @torch.no_grad()
    def audit(self, batch: Any) -> LossDict:
        return self.eval_step(batch)

    def _per_token_soft_hilbert(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-token soft Hilbert distance ``d_H(f(x), x)`` of shape ``(B, L)``.

        ``f`` is the full forward pass (backbone + velocity head). The velocity
        head enforces the sum-to-zero tangent-space constraint on ``f`` by
        construction, so no further centering of ``f`` is needed. The input
        ``log_x`` is centered to match ILR/CLR convention before computing the
        Nielsen LogSumExp soft Hilbert distance.
        """
        f = self.forward(log_x)
        x_c = log_x - log_x.mean(dim=-1, keepdim=True)
        return nielsen_soft_hilbert_distance(f, x_c, alpha=self.cfg.training.soft_hilbert_alpha)

    @torch.no_grad()
    def energy_score(self, log_x: torch.Tensor) -> torch.Tensor:
        """Stage 1 geometric energy ``g(x) = -d_H(f(x), x)``.

        Sequence-level scalar (averaged over tokens). This is the energy
        implied by the EqM + Hilbert training objective: on the valid manifold
        the learned ``f`` is close to ``x`` so ``g(x)`` is near zero; far from
        the manifold ``f`` and ``x`` diverge in the soft Hilbert metric and
        ``g(x)`` becomes large and negative.

        For AUROC-style anomaly scoring use `ood_score` (which returns the
        sign-flipped quantity ``d_H`` so that higher = more OOD).

        Args:
            log_x: (B, L, D) sequences in ILR / log-simplex coordinates.

        Returns:
            (B,) tensor of per-sequence geometric energies.
        """
        was_training = self.training
        self.eval()
        d_per_token = self._per_token_soft_hilbert(log_x)
        energy = -d_per_token.mean(dim=-1)
        if was_training:
            self.train()
        return energy

    @torch.no_grad()
    def ood_score(self, log_x: torch.Tensor) -> torch.Tensor:
        """Sequence-level OOD score ``d_H(f(x), x)`` (higher = more OOD).

        This is ``-energy_score(log_x)``, sign-flipped so it can be used
        directly as an anomaly score in AUROC-style benchmarks (where the
        positive class is "invalid").
        """
        return -self.energy_score(log_x)

    @torch.no_grad()
    def residual_score(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-sample EqM residual ``||v(log_x)||_2`` — flow-residual UQ signal.

        Intuition: on the valid manifold the learned velocity field approaches a
        fixed point, so its magnitude is small; off-manifold sequences induce a
        large restoring velocity. This signal is complementary to the
        geometric `energy_score` and lets benchmarks compare flow-residual UQ
        against the Hilbert energy from a single Stage 1 model.

        Args:
            log_x: (B, L, D) sequences in log-simplex coordinates.

        Returns:
            (B,) tensor of L2-norms of the velocity flattened over (L, D).
        """
        was_training = self.training
        self.eval()
        v = self.forward(log_x)
        score = v.reshape(v.shape[0], -1).norm(dim=1)
        if was_training:
            self.train()
        return score

    @torch.no_grad()
    def score_per_sample(self, log_x: torch.Tensor) -> torch.Tensor:
        """Sequence-level OOD score for benchmark parity (= `ood_score`).

        Returns the geometric soft-Hilbert anomaly score ``d_H(f(x), x)``
        (higher = more OOD). The previous behavior
        of aliasing `residual_score` (velocity norm) is intentionally
        replaced — `auroc_auditor` for a Stage 1 model now reflects the
        geometric energy instead of the velocity norm. The flow-residual is
        still available via `residual_score` (and surfaces in the benchmark
        as `auroc_residual`).
        """
        return self.ood_score(log_x)

    @torch.no_grad()
    def token_latents(self, log_x: torch.Tensor) -> torch.Tensor:
        """Per-token backbone features, shape ``(B, L, d_model)``.

        Stage 1 has no learned latent projection (that is Stage 2's
        ``latent_head``), so we expose the backbone hidden states directly as
        the "latent" channel for diagnostic plots. The returned tensor is
        detached and lives on the input device.
        """
        was_training = self.training
        self.eval()
        h = self.backbone(log_x)
        if was_training:
            self.train()
        return h.detach()

    @torch.no_grad()
    def inducing_points(self) -> torch.Tensor | None:
        """Stage 1 has no GP; no inducing points exist."""
        return None

    @torch.no_grad()
    def per_token_uq(self, log_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Per-token geometric energy and flow residual for benchmark parity.

        Returns three tensors compatible with the GP auditor's `per_token_uq`:

        * ``energy`` (B, L): per-token Hilbert energy ``-d_H`` of
          ``(f(x), x)`` — directly comparable to the composed model's GP
          per-token energy channel.
        * ``variance`` (B, L): per-token velocity-norm proxy
          ``||v(x_t)||_2`` (no GP available at Stage 1 — this is reused as
          a flow-residual variance surrogate).
        * ``noise``: 0.0 (no aleatoric noise estimate at this stage).
        """
        was_training = self.training
        self.eval()
        v = self.forward(log_x)
        per_tok_resid = v.norm(dim=-1)
        d_per_token = self._per_token_soft_hilbert(log_x)
        per_tok_energy = -d_per_token
        if was_training:
            self.train()
        return per_tok_energy, per_tok_resid, 0.0

    @torch.no_grad()
    def integrate(
        self,
        log_x: torch.Tensor,
        steps: int | None = None,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Project ``log_x`` toward the valid manifold via ``steps`` Euler updates.

        For CLR mode the velocity head outputs vectors in V_K (zero-sum), so an
        explicit re-centering after each step guards against numerical drift. For
        ILR mode there is no sum-to-zero constraint; re-centering is skipped.
        """
        eq = self.cfg.equilibrium
        steps = steps if steps is not None else eq.generate_steps
        dt = dt if dt is not None else eq.generate_stepsize
        clr = self.cfg.hf_dataset.transform_mode.lower() == "clr"
        x = log_x.clone()
        for _ in range(steps):
            x = x - self.forward(x) * dt
            if clr:
                x = x - x.mean(dim=-1, keepdim=True)  # project to V_K (CLR only)
        return x

    @torch.no_grad()
    def generate(
        self,
        n: int,
        steps: int | None = None,
        stepsize: float | None = None,
    ) -> torch.Tensor:
        """Generate samples from uniform log-space noise via ``x ← x - v(x) * dt``.

        Each Euler step is followed by re-centering onto the V_K tangent space
        to prevent numerical drift off the ILR subspace.
        """
        eq = self.cfg.equilibrium
        steps = steps if steps is not None else eq.generate_steps
        stepsize = stepsize if stepsize is not None else eq.generate_stepsize
        device = self.cfg.training.device
        L = self.cfg.dataset.L
        D = feature_dim(self.cfg)

        clr = self.cfg.hf_dataset.transform_mode.lower() == "clr"
        was_training = self.training
        self.eval()
        x = _uniform_log_x0(n, L, D, device, torch.float32)
        x = x + eq.generate_init_noise * torch.randn_like(x)
        if clr:
            x = x - x.mean(dim=-1, keepdim=True)  # ensure noise starts in V_K
        for _ in range(steps):
            x = x - self.forward(x) * stepsize
            if clr:
                x = x - x.mean(dim=-1, keepdim=True)  # project to V_K (CLR only)
        if was_training:
            self.train()
        return x.detach()


@register("bayesian_auditor_stage1")
def build_bayesian_auditor_stage1(cfg: Config) -> BayesianAuditorStage1:
    return BayesianAuditorStage1(cfg)
