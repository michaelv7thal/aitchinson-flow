"""Statistical Flow Matching (SFM; Cheng et al. 2024, arXiv:2405.16441).

Categorical flow matching on the statistical (Fisher–Rao) manifold. The
probability simplex Δ^{K-1} is mapped to the positive orthant of the sphere
S^{K-1} by the diffeomorphism

    π:  μ ↦ x = √μ          (inverse  π⁻¹: x ↦ μ = x²),

under which the Fisher–Rao metric pulls back to the round sphere metric, so
information geometry becomes ordinary spherical geometry. Flow matching is then
performed *on the sphere*:

  * source   x₀ = √μ₀,  μ₀ ~ Uniform(Δ) = Dir(1,…,1)   (the same prior as
    DirichletFM, mapped through π);
  * target   x₁ = √(one-hot) = e_c   (a sphere vertex);
  * path     x_t = exp_{x₀}(t · log_{x₀} x₁)   (constant-speed great-circle
    geodesic), with tangent (geodesic) velocity
        u_t(x_t) = d(x₀,x₁) · unit(log_{x_t} x₁);
  * a transformer velocity field v(x_t, t) ∈ T_{x_t}S is regressed against u_t
    with an MSE flow-matching loss (Eq. 8 of the paper).

Sampling integrates ẋ = v(x_t, t) on the sphere from t=0→1 with exponential-map
Euler steps and reads tokens from μ = x² (categorical).

Unlike :class:`SFLM` (arXiv:2605.11125 — a hyperspherical flow on a *learned
codebook* sphere of dimension ``d_embed``), SFM works directly on the K-dim
Fisher sphere with **no codebook**: the embedding *is* the √μ map. It reuses the
DirichletFM backbone (``DFMBackbone`` + ``DFMHead``) so the comparison is
fair-compute.

The exact continuous-normalizing-flow likelihood / peer-comparable BPC (paper
Eqs. 12–14) is computed by a separate post-hoc readout, not here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.transformer_backbone import DFMBackbone, DFMHead


# --------------------------------------------------------------------------- #
# Sphere primitives (S^{K-1} ⊂ R^K; same idiom as sflm_ebm.py, acting on the
# last axis). Kept local to mirror the per-module convention in this package.
# --------------------------------------------------------------------------- #
def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def _geodesic(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    dot = (p * q).sum(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(dot)


def _project_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v - (p * v).sum(-1, keepdim=True) * p


def _exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.cos(vn) * p + torch.sin(vn) * (v / vn)


def _log_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``log_p(q)``: tangent at p toward q with norm = geodesic distance."""
    omega = _geodesic(p, q).unsqueeze(-1)
    s = torch.sin(omega).clamp(min=1e-7)
    return (omega / s) * (q - torch.cos(omega) * p)


def _simplex_to_sphere(mu: torch.Tensor) -> torch.Tensor:
    """π: μ ↦ √μ — the Fisher–Rao diffeomorphism onto the positive sphere."""
    return _normalize(mu.clamp(min=0.0).sqrt())


def _sphere_to_simplex(x: torch.Tensor) -> torch.Tensor:
    """π⁻¹: x ↦ x² (renormalised onto the simplex)."""
    mu = x * x
    return mu / mu.sum(-1, keepdim=True).clamp(min=1e-12)


class StatisticalFlowMatching(nn.Module):
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
        ones = torch.ones(self.K, device=device, dtype=dtype)
        mu0 = Dirichlet(ones).sample((B, L))            # uniform on the simplex
        return _simplex_to_sphere(mu0)

    def _sphere_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(token_ids.long(), self.K).to(torch.float32)
        return _simplex_to_sphere(oh)                   # = e_c vertices

    # ----- training: MSE flow matching against the geodesic velocity ------- #
    def _loss(self, batch: Any) -> LossDict:
        token_ids = batch["token_ids"].long()
        device = token_ids.device
        x1 = self._sphere_target(token_ids)             # (B, L, K)
        B, L, _ = x1.shape
        x0 = self._sphere_source(B, L, device, x1.dtype)

        eps = self.cfg.sfm.t_eps
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
        is the cross-arm alias for ``nfe``."""
        if nfe is None:
            nfe = max_steps if max_steps is not None else self.cfg.sfm.sample_nfe
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
        likelihood. The peer-comparable bits/char is the exact CNF bound (paper
        Eqs. 12–14) computed by a separate post-hoc readout (forthcoming)."""
        del max_steps
        device = next(self.parameters()).device
        ids = token_ids.to(device).long()
        x1 = self._sphere_target(ids)
        logp = self.decode_to_logprobs(_normalize(x1 + 0.05 * torch.randn_like(x1)))
        nll = F.nll_loss(logp.reshape(-1, self.K), ids.reshape(-1), reduction="mean")
        return nll / math.log(2)


@register("SFM")
def build_sfm(cfg: Config) -> StatisticalFlowMatching:
    return StatisticalFlowMatching(cfg)
