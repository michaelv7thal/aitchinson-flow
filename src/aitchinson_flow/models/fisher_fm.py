"""Fisher-Flow / Fisher Flow Matching (Davis et al. 2024, arXiv:2405.14664,
bib key ``davis2024fisherfm``).

This arm is a **budget-matched reimplementation of Fisher-Flow**, added for
completeness of the Fisher-Rao transport family in the benchmark. Read the
recipe below before editing; it was pinned from the paper (abstract + §3), not
guessed.

Fisher-Flow and :class:`StatisticalFlowMatching` (SFM; Cheng et al. 2024,
arXiv:2405.16441) are **independent, concurrent constructions** which arrive at
the same geometry: both map the simplex Δ^{K-1} to the positive orthant of the
sphere S^{K-1} via π: μ ↦ √μ (Fisher–Rao becomes the round-sphere metric) and
transport along great-circle geodesics. Neither is a variant of the other, and
this module must be read as an implementation of Davis et al. on its own terms.
The sphere helpers are imported from ``sfm.py`` rather than copied purely to
avoid two copies of the same primitives; that is a code-sharing decision, not a
claim about the methods.

Fisher-Flow's own recipe adds a **Riemannian minibatch optimal-transport
coupling** between the source and target samples (paper §3.4): "solve for the OT
plan π using the squared distance on 𝕊^d_+ as the cost … using the Sinkhorn
algorithm," minimising E_{(x₀,x₁)~π'}[d²(x₀,x₁)].

Recipe pinned from the paper (the six questions the plan asked):
  1. Source  p₀ = U(𝕊^d_+) — uniform simplex mapped through π.
  2. Target  x₁ = π(σ(one-hot)), where σ: Δ → Δ̊ is the paper's smoothing map
     into the simplex *interior* ("e.g. label smoothing", §3), which keeps the
     target off the boundary of the positive orthant as the method intends. The
     paper fixes no smoothing amount, so we use the repo-wide default
     ε = 1e-4 of ``TransformationConfig.label_smoothing``, i.e. the same
     x = (1−ε)·e_i + (ε/K)·1 mixture the clr-space arms already use. AMBIGUITY
     (documented, not invented): the amount of smoothing, not its presence.
     ``label_smoothing=0.0`` recovers the exact-vertex target if ever needed.
  3. Coupling  Riemannian minibatch OT, cost = summed squared geodesic distance
     over the sequence (product-manifold reading). The paper used entropic
     Sinkhorn; at the matched batch (8 sequences) the exact assignment is the
     natural hyperparameter-free reading of the OT plan, so we default to exact
     ``scipy.optimize.linear_sum_assignment`` (``ot_reg=0``) and provide a
     log-domain Sinkhorn behind ``ot_reg>0`` as the faithful-to-paper
     alternative. AMBIGUITY (documented): exact-assignment vs entropic Sinkhorn.
  4. Velocity/loss  constant-speed geodesic velocity u_t = d(x₀,x₁)·unit(log_{x_t}x₁)
     (the numerically stable form of the paper's log_{x_t}(x₁)/(1−t)),
     tangent-projected output, Riemannian MSE, t ~ U(0,1).
  5. Sampler  exp-map Euler on the sphere; NFE from config.
  6. text8  none in the paper ("not fully developed for language modeling
     domains"). So this arm has no exact likelihood: its ``bpd`` is a
     reconstruction diagnostic and it belongs in ``_IDENTITY_PATH_BPC``.

Caveat worth stating in any write-up: at batch 8 the OT plan couples only 8
sequences, so whatever path-straightening it can buy is weak at this budget.
That is a property of the benchmark budget, not a defect of the method.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.transformer_backbone import DFMBackbone, DFMHead

# Geometry is shared with SFM — import, do not copy (these are treated as public;
# two eval scripts already import them under aliases).
from aitchinson_flow.models.sfm import (
    _exp_map,
    _geodesic,
    _log_map,
    _normalize,
    _project_tangent,
    _simplex_to_sphere,
    _sphere_to_simplex,
)


class FisherFlowMatching(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.backbone = DFMBackbone(cfg)
        self.head = DFMHead(cfg)

    # ----- velocity field v(x, t) ∈ T_x S ---------------------------------- #
    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.backbone(x, t))            # (B, L, K)
        return _project_tangent(x, raw)

    # ----- source / target on the sphere ----------------------------------- #
    def _sphere_source(self, B: int, L: int, device, dtype) -> torch.Tensor:
        """p₀ = U(𝕊^d₊): uniform w.r.t. the *Riemannian volume* of the orthant,
        which is what the paper specifies ("p₀ = 𝒰(𝕊^d₊), by default", and the
        uninformative prior √det G(x)/∫√det G(x)).

        Folding a standard Gaussian into the orthant and normalising gives that
        measure exactly, by rotational symmetry. Note this is NOT the pushforward
        of the flat Dirichlet(1,…,1): π is an isometry from the Fisher–Rao
        simplex, so the uniform sphere measure pulls back to the Fisher–Rao
        volume, i.e. Jeffreys' Dir(½,…,½). Sampling Dir(1,…,1) here would put a
        visibly different prior on the sphere (E[max μ] 0.144 vs 0.201 at K=27).
        """
        z = torch.randn(B, L, self.K, device=device, dtype=dtype).abs()
        return _normalize(z)

    def _sphere_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(token_ids.long(), self.K).to(torch.float32)
        s = self.cfg.fisher_fm.label_smoothing
        if s > 0.0:
            # σ: Δ → Δ̊ — smooth the vertex toward the barycentre before π (paper
            # §3, "e.g. label smoothing"). s=0 ⇒ exact vertex ⇒ identical to SFM.
            oh = (1.0 - s) * oh + s / self.K
        return _simplex_to_sphere(oh)                   # = e_c vertices when s=0

    # ----- Riemannian minibatch OT coupling (Fisher-Flow's delta) ---------- #
    @torch.no_grad()
    def _ot_reorder(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Reorder the source batch ``x0`` (B,L,K) so each target sequence
        ``x1[j]`` is paired with its OT-matched source, minimising the summed
        squared geodesic distance over positions (product-sphere cost). Returns
        a permutation of ``x0`` aligned to ``x1``'s index order."""
        B = x0.shape[0]
        if B < 2:
            return x0
        # C[j, i] = Σ_ℓ d²(x1_j[ℓ], x0_i[ℓ])   (targets × sources)
        d = _geodesic(x1.unsqueeze(1), x0.unsqueeze(0))     # (B, B, L)
        cost = (d * d).sum(-1)                              # (B, B)
        reg = self.cfg.fisher_fm.ot_reg
        if reg > 0.0:
            # Entropic OT (log-domain Sinkhorn) — the paper's exact procedure —
            # then sample a source per target from the plan's conditional rows.
            plan = _sinkhorn(cost, reg, self.cfg.fisher_fm.ot_iters)  # (B, B)
            rows = plan / plan.sum(-1, keepdim=True).clamp(min=1e-12)
            col = torch.multinomial(rows, 1).squeeze(-1)             # (B,)
        else:
            # Exact OT assignment (hyperparameter-free; natural at small batch).
            _, col = linear_sum_assignment(cost.detach().cpu().numpy())
            col = torch.as_tensor(col, device=x0.device, dtype=torch.long)
        return x0[col]

    # ----- training: MSE flow matching against the geodesic velocity ------- #
    def _loss(self, batch: Any) -> LossDict:
        token_ids = batch["token_ids"].long()
        device = token_ids.device
        x1 = self._sphere_target(token_ids)             # (B, L, K)
        B, L, _ = x1.shape
        x0 = self._sphere_source(B, L, device, x1.dtype)

        if self.cfg.fisher_fm.use_ot:
            x0 = self._ot_reorder(x0, x1)               # Fisher-Flow OT coupling

        eps = self.cfg.fisher_fm.t_eps
        t = torch.rand(B, device=device) * (1.0 - eps)  # (B,) in [0, 1-eps)
        x_t = _exp_map(x0, t[:, None, None] * _log_map(x0, x1))

        # constant-speed geodesic velocity at x_t:
        #   u_t = d(x0,x1) · unit(log_{x_t} x1)   (avoids the 1/(1-t) form).
        lm = _log_map(x_t, x1)
        u_t = _geodesic(x0, x1).unsqueeze(-1) * _normalize(lm)
        v = self.velocity(x_t, t)
        loss = F.mse_loss(v, u_t)
        return {"fm": loss.detach(), TRAINING_LOSS_KEY: loss}

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._loss(batch)

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        return self._loss(batch)

    # ----- decode helpers (μ = x²) for bench / recovery API parity --------- #
    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        return _sphere_to_simplex(x).clamp(min=1e-12).log()

    def decode_to_token_ids(self, x: torch.Tensor) -> torch.Tensor:
        return (x * x).argmax(-1)

    # ----- sampling: exp-map Euler ODE on the sphere, t:0→1 ---------------- #
    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        nfe: int | None = None,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        t_start: float | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Integrate ẋ = v(x, t) on the sphere from ``t_start`` (default 0) to 1
        with exponential-map Euler steps; return argmax token ids of μ = x².

        ``x_init`` (e.g. a perturbed-data init from ``recovery_check.py``) is
        renormalised onto S^{K-1} and used as the starting point; ``max_steps``
        is the cross-arm alias for ``nfe``. Sampling has no coupling — the OT
        plan only ever touches the training pairing."""
        if nfe is None:
            nfe = max_steps if max_steps is not None else self.cfg.fisher_fm.sample_nfe
        device = next(self.parameters()).device
        start = float(t_start) if t_start is not None else 0.0
        if x_init is not None:
            x = _normalize(x_init.to(device).float())
        else:
            x = self._sphere_source(B, L, device, torch.float32)
        t_grid = torch.linspace(start, 1.0, nfe + 1, device=device)
        for i in range(nfe):
            t = t_grid[i]
            dt = float(t_grid[i + 1] - t)
            v = self.velocity(x, t.expand(x.shape[0]))
            x = _normalize(_exp_map(x, dt * v))
        return (x * x).argmax(-1)             # token ids from μ = x²

    @torch.no_grad()
    def bpd(
        self, token_ids: torch.Tensor, *, max_steps: int | None = None
    ) -> torch.Tensor:
        """Reconstruction-NLL diagnostic for cross-arm API parity — **NOT** a
        likelihood. Fisher-Flow reports no exact text8 likelihood, so this arm
        prints ``—`` for peer-comparable BPC (see ``_IDENTITY_PATH_BPC``)."""
        del max_steps
        device = next(self.parameters()).device
        ids = token_ids.to(device).long()
        x1 = self._sphere_target(ids)
        logp = self.decode_to_logprobs(_normalize(x1 + 0.05 * torch.randn_like(x1)))
        nll = F.nll_loss(logp.reshape(-1, self.K), ids.reshape(-1), reduction="mean")
        return nll / math.log(2)


def _sinkhorn(cost: torch.Tensor, reg: float, iters: int) -> torch.Tensor:
    """Log-domain Sinkhorn OT plan for uniform marginals (paper §3.4's entropic
    OT). ``cost`` is (n, n); returns a coupling matrix whose row/col marginals
    are ≈ 1/n. Used only when ``cfg.fisher_fm.ot_reg > 0``."""
    n = cost.shape[0]
    log_mu = -math.log(n)                      # uniform log-marginal
    log_K = -cost / reg                        # (n, n)
    log_u = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_mu - torch.logsumexp(log_K + log_u[:, None], dim=0)
    return (log_u[:, None] + log_K + log_v[None, :]).exp()


@register("FisherFM")
def build_fisher_fm(cfg: Config) -> FisherFlowMatching:
    return FisherFlowMatching(cfg)
