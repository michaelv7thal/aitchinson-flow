"""FMonCLR — standard (Lipman) conditional flow matching on CLR features.

The textbook Lipman et al. (2022) conditional-OT flow-matching baseline,
adapted to the CLR (zero-mean R^K) simplex embedding:

    x_t   = (1 − t)·x0 + t·x1,      t ~ U[0, 1]      (straight-line OT path)
    target = (x0 − x1)                               (CONSTANT cond. velocity)
    loss   = E‖ f(x_t; t) − (x0 − x1) ‖²  + λ_ce·CE  (regression + token anchor)
    sample : x ← x − h·f(x; t),  t: 0 → 1            (ODE integration → data)

Note the repo-wide **data→noise** sign convention: the network learns the
*negative* OT velocity ``x0 − x1`` (= −dx_t/dt), and the sampler subtracts it
(`x ← x − h·v`) to move source→data — exactly as ``EqM.sample_euler`` and
``sde_flow_sample`` do. This is mathematically identical to the canonical
``v = x1 − x0`` / ``x ← x + h·v`` form, just a sign labelling, and lets
FMonCLR reuse the shared Euler-γ / SDE samplers unchanged.

What this is NOT (and previously was): there is **no** ``c(γ)`` decay on the
target and **no** ``γ=√U`` importance reshaping (t is uniform). Those were EqM
borrows that made the old FMonCLR a non-Lipman ablation whose Euler integral
only reached the noise↔data midpoint; the constant target here transports the
full path to the data endpoint.

The one EqM-family term deliberately kept is the auxiliary token CE on the
implied-x1 reconstruction (gated by ``lambda_ce`` / ``ce_min_gamma``) — added
for cross-arm **comparability** with EqM / EqM_OneHot / EqMAE, not because
textbook Lipman needs it. For the constant-velocity field the implied clean
state is ``x1 = x_t − (1−t)·v`` (exact at convergence).

Reuses EqM's TransformerBackbone (t-conditioning, forced on) and VelocityHead.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register


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

        # Source p0: centred Gaussian on the V_d (zero-mean CLR) subspace.
        x0 = s.source_sigma * torch.randn(B, L, K, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)

        # Conditional-OT path x_t = (1−t)x0 + t x1 with UNIFORM t (the standard
        # Lipman time weighting — no γ=√U importance reshaping).
        t = torch.rand(B, device=device, dtype=dt)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1

        v = self.forward(x_t, t)
        # CONSTANT conditional-OT velocity (no c(γ) decay). Data→noise sign
        # convention: target = x0 − x1 = −dx_t/dt, so the sampler's x ← x − h·v
        # integrates source→data (matches EqM.sample_euler / sde_flow_sample).
        u_tgt = x0 - x1

        flow_loss = F.mse_loss(v, u_tgt)
        total = flow_loss
        out: LossDict = {"flow_loss": flow_loss}

        # Aux CE on the implied-x1 reconstruction — the same token anchor the
        # EqM family carries, kept for cross-arm comparability (not part of
        # textbook Lipman). For the constant-velocity field the state→clean
        # inverse of x_t = (1−t)x0 + t·x1 with v = x0 − x1 is
        #     x1 = x_t − (1−t)·v
        # (exact at convergence, mirroring EqM's x_γ − λ·grad_g). Masked to the
        # signal regime t ≥ ce_min_gamma, same as EqM.
        if s.lambda_ce > 0.0 and token_ids is not None:
            mask = t >= s.ce_min_gamma
            if mask.any():
                pred_x1 = x_t[mask] - (1.0 - t[mask])[:, None, None] * v[mask]
                log_probs = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
                ids = token_ids[mask].long()
                ce = F.nll_loss(log_probs.reshape(-1, K), ids.reshape(-1))
                total = total + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total

        # t-bucket diagnostics (same bucket keys as EqM for cross-comparison).
        for key, m in (
            ("g<.33", t < 0.33),
            ("g<.66", (t >= 0.33) & (t < 0.66)),
            ("g<1", t >= 0.66),
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
        method: str | None = None,
        alpha: float | None = None,
        use_grad: bool | None = None,
    ) -> torch.Tensor:
        """Euler-γ sampler (deterministic) or SDE Euler-γ (Phase R).

        Honours either ``max_steps`` (EqM convention used by
        ``runner._unigram_kl_probe`` and ``scripts/eval_full.py``) or
        ``nfe`` for symmetry with DFM. ``method="sde"`` routes to the
        Langevin Euler sampler with diffusion coefficient ``alpha``.
        ``use_grad`` selects raw f vs. ∇⟨x,f⟩ — FMonCLR's training target
        is raw f, so the natural choice is False (the default).
        """
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

        if method == "sde":
            from aitchinson_flow.sampling.sde import sde_flow_sample
            return sde_flow_sample(
                self,
                x,
                n_steps=steps,
                use_grad=bool(use_grad) if use_grad is not None else False,
                alpha=float(alpha) if alpha is not None else 0.0,
                time_conditioned=True,
                project_zero_mean=True,
                grad_clip=getattr(self.cfg.eqm, "sample_grad_clip", None),
            )

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
