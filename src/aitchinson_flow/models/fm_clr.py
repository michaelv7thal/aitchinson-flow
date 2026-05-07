"""FMonCLR — direct flow matching on CLR features.

Third continuous baseline alongside EqM and DFM. Designed as a clean
ablation over EqM's conservative-gradient indirection: regress the velocity
``f(x_γ; γ)`` *directly* to the FM target ``c(γ)·(x0 - x1)`` (no
``∇⟨x,f⟩`` step), then Euler-sample over γ.

Triangulation purpose (RESULTS.md, plan §C):
- If FMonCLR ≈ DFM on KL_bi, the conservative-grad indirection is the
  specific culprit and EqM-Euler (W1) should help.
- If FMonCLR ≈ EqM, continuous-on-simplex is generally hard regardless of
  the conservative-grad indirection.
Either outcome breaks the "n=1 continuous model" problem in the writeup.

Reuses EqM's TransformerBackbone (γ-conditioning included), VelocityHead,
and the ``_c_gamma`` decay schedule. Sampling uses the same Euler-γ loop as
``EqM.sample_euler`` so the head-to-head is fair.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.models.eqm import EquilibriumFlowMatching


class FMonCLR(nn.Module):
    """Direct (non-conservative) flow matching on CLR features."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        # Force time conditioning on — FM needs γ as a path index, not a knob.
        # If a user accidentally leaves time_conditioning="off" the encoder
        # silently ignores γ and we'd train an averaged-over-γ velocity.
        if getattr(cfg.eqm, "time_conditioning", "off") == "off":
            from dataclasses import replace
            cfg = replace(cfg, eqm=replace(cfg.eqm, time_conditioning="add"))
        self.cfg = cfg
        self.backbone = TransformerBackbone(cfg=cfg)
        self.velocity_head = VelocityHead(cfg=cfg)

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.velocity_head(self.backbone(x, gamma))

    def _c_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        # Identical schedule to EqM — keep one source of truth.
        return EquilibriumFlowMatching._c_gamma(self, gamma)

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._fm_loss(batch["x"], token_ids=batch.get("token_ids"))

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        # MSE is symmetric on the centred V_d subspace; no_grad fine.
        return self._fm_loss(batch["x"], token_ids=batch.get("token_ids"))

    def _fm_loss(
        self, x1: torch.Tensor, *, token_ids: torch.Tensor | None = None
    ) -> LossDict:
        B, L, K = x1.shape
        s = self.cfg.eqm
        device, dt = x1.device, x1.dtype

        x0 = s.source_sigma * torch.randn(B, L, K, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)

        gamma = torch.rand(B, device=device, dtype=dt).pow(s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1

        v = self.forward(x_gamma, gamma)
        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        flow_loss = F.mse_loss(v, u_tgt)
        total = flow_loss
        out: LossDict = {"flow_loss": flow_loss}

        # Aux CE on linear-decay implied x1 = x_γ − λ·v. Same anchor idea as
        # EqM, but here `v` is the raw velocity (no grad-of-energy step), so
        # there is no second-order autograd path through this term.
        if s.lambda_ce > 0.0 and token_ids is not None:
            mask = gamma >= s.ce_min_gamma
            if mask.any():
                lam = s.gradient_lambda
                pred_x1 = x_gamma[mask] - lam * v[mask]
                log_probs = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
                ids = token_ids[mask].long()
                ce = F.nll_loss(log_probs.reshape(-1, K), ids.reshape(-1))
                total = total + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total

        # γ-bucket diagnostics (same buckets as EqM for cross-comparison).
        for key, m in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if m.any():
                out[key] = F.mse_loss(v[m], u_tgt[m]).detach()

        return out

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        nfe: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler-γ sampler. Honours either ``max_steps`` (EqM convention used
        by ``runner._unigram_kl_probe`` and ``scripts/eval_full.py``) or
        ``nfe`` for symmetry with DFM."""
        s = self.cfg.eqm
        steps = max_steps if max_steps is not None else (
            nfe if nfe is not None else s.euler_nfe
        )
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K

        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)

        gammas = torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]
        h = 1.0 / steps
        for g in gammas:
            g_b = g.expand(B)
            v = self.forward(x, g_b)
            x = x - h * v
            x = x - x.mean(dim=-1, keepdim=True)
        return x

    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.logsumexp(x, dim=-1, keepdim=True)

    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Inner-product readout ``⟨x, f(x; γ=sample_gamma)⟩`` for OOD
        scoring. Not a trained energy (FMonCLR has no conservative-grad
        objective) — included for parity with EqM in eval_ood.py."""
        s = self.cfg.eqm
        B = x.shape[0]
        gamma = torch.full((B,), float(s.sample_gamma), device=x.device, dtype=x.dtype)
        v = self.forward(x, gamma)
        return (x * v).sum(dim=(1, 2))


@register("FMonCLR")
def build_fmclr(cfg: Config) -> FMonCLR:
    return FMonCLR(cfg)
