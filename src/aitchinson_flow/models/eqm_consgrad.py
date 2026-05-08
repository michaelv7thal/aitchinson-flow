"""Phase T (CAPSTONE_EXPERIMENTS.md §6) — EqM trained directly on the
conservative gradient.

Hypothesis: the gap between Euler-on-``f`` (KL_bi 1.682 in W1) and
Euler-on-``∇⟨x,f⟩`` (1.391) on the same EqM checkpoint exists because FM
regression supervises ``f`` directly, leaving the data-pulling Jacobian
information unsupervised. This model's *output* is the conservative
gradient; its training target is the FM target. Euler-on-output of this
model is therefore equivalent (in objective) to Euler-on-``∇⟨x,f⟩`` of an
ordinary EqM checkpoint, but with the training signal aligned with the
sampler.

Architecture is identical to ``EquilibriumFlowMatching`` (same backbone,
same VelocityHead). The only difference is that ``forward`` re-runs the
backbone, takes the inner product with ``x``, and back-propagates to
return ``∇_x ⟨x, f(x)⟩`` as the model's output. The training step then
regresses *that* tensor to ``c(γ)·(x_0 − x_1)``.

Notes:
* ``create_graph=True`` is required only at training time — eval and
  sampling run the same outer ``torch.autograd.grad`` with
  ``create_graph=False``, so memory cost at inference is just one extra
  backward pass per call.
* The aux CE on the implied-x1 reconstruction is preserved (linear-decay
  ``x1 ≈ x_γ − λ·grad_g``); since the model's output already *is*
  ``grad_g`` this becomes ``pred_x1 = x_γ − λ·model_out``.
* The auditor-hinge branch is intentionally omitted — the writeup uses
  this model only for generation, not auditing.

Sample method:
* ``method="euler"`` (default in W1) calls the model directly and runs
  Euler. This is the "natural sampler" for this model — what the
  writeup PASS criterion compares against the original EqM's
  Euler-on-``∇⟨x,f⟩``.
* ``method="nag"`` falls back to the EqM NAG-GD path on the conservative
  gradient. Since the model's output already *is* the gradient, NAG is
  applied to it directly.
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


class EqMConsGrad(nn.Module):
    """Equilibrium Flow Matching with the conservative gradient as output."""

    def __init__(self, cfg: Config, loss_fn: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.backbone = TransformerBackbone(cfg=cfg)
        self.velocity_head = VelocityHead(cfg=cfg)

    def _raw_velocity(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity_head(self.backbone(x, gamma))

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        h_ctx: torch.Tensor | None = None,  # noqa: ARG002 — parity with EqM signature
    ) -> torch.Tensor:
        """Return ``∇_x ⟨x, f(x; γ)⟩`` — the model's actual output.

        Outside of training (``self.training=False``) the outer autograd
        graph is not retained, so this is essentially a single extra
        backward pass on top of the forward.
        """
        was_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            x_req = x if x.requires_grad else x.detach().requires_grad_(True)
            v = self._raw_velocity(x_req, gamma)
            energy = (x_req * v).sum()
            grad = torch.autograd.grad(
                energy,
                x_req,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
        if not was_grad:
            grad = grad.detach()
        return grad

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._loss(batch["x"], token_ids=batch.get("token_ids"))

    def eval_step(self, batch: Any) -> LossDict:
        return self._loss(batch["x"], token_ids=batch.get("token_ids"))

    def _loss(
        self,
        x1: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
    ) -> LossDict:
        B, L, D = x1.shape
        device, dt = x1.device, x1.dtype
        s = self.cfg.eqm

        x0 = s.source_sigma * torch.randn(B, L, D, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)

        gamma = torch.rand(B, device=device, dtype=dt).pow(s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        x_gamma.requires_grad_(True)

        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        # Model output IS the conservative gradient.
        grad_g = self.forward(x_gamma, gamma)

        flow_loss = self.loss_fn(grad_g, u_tgt)
        total = flow_loss
        out: LossDict = {"flow_loss": flow_loss}

        if s.lambda_ce > 0.0 and token_ids is not None:
            mask = gamma >= s.ce_min_gamma
            if mask.any():
                lam = s.gradient_lambda
                pred_x1 = x_gamma[mask] - lam * grad_g[mask]
                log_probs = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
                ids = token_ids[mask].long()
                ce = F.nll_loss(log_probs.reshape(-1, D), ids.reshape(-1))
                total = total + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total

        for key, m in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if m.any():
                out[key] = self.loss_fn(grad_g[m], u_tgt[m])

        return out

    # Reuse the same _c_gamma decay schedule as EqM.
    def _c_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        return EquilibriumFlowMatching._c_gamma(self, gamma)

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
        use_grad: bool | None = None,  # accepted for API parity; consgrad output is already the gradient
    ) -> torch.Tensor:
        """Default to Euler-γ on the model's output (which IS the gradient).

        ``method="sde"`` → Langevin Euler with diffusion coeff α.
        ``method="nag"`` → NAG-GD on the model's output (treated as
        ``∇⟨x,f⟩`` directly), reusing EqM's NAG implementation.
        """
        s = self.cfg.eqm
        steps = (
            max_steps if max_steps is not None
            else (nfe if nfe is not None else s.euler_nfe)
        )

        chosen = method if method is not None else "euler"

        if chosen == "sde":
            from aitchinson_flow.sampling.sde import sde_flow_sample
            device = next(self.parameters()).device
            K = self.cfg.text8_dataset.K
            sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
            if x_init is not None:
                x0 = x_init.to(device).detach()
            else:
                x0 = sigma * torch.randn(B, L, K, device=device)
                x0 = x0 - x0.mean(dim=-1, keepdim=True)
            time_cond = getattr(s, "time_conditioning", "off") != "off"
            return sde_flow_sample(
                self,
                x0,
                n_steps=steps,
                use_grad=False,  # model's output is already the gradient
                alpha=float(alpha) if alpha is not None else 0.0,
                time_conditioned=time_cond,
                project_zero_mean=True,
            )

        if chosen == "nag":
            # Treat self.forward(x) (which is ∇E) as the gradient, run EqM NAG.
            return self._sample_nag(B, L, max_steps=steps, x_init=x_init)

        # Default: Euler on the model's output.
        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)
        time_cond = getattr(s, "time_conditioning", "off") != "off"
        gammas = torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]
        h = 1.0 / steps
        for g in gammas:
            g_b = g.expand(B) if time_cond else None
            v = self.forward(x, g_b)
            x = x - h * v
            x = x - x.mean(dim=-1, keepdim=True)
        return x

    def _sample_nag(
        self,
        B: int,
        L: int,
        *,
        max_steps: int,
        x_init: torch.Tensor | None,
    ) -> torch.Tensor:
        s = self.cfg.eqm
        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K
        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = s.source_sigma * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)
        x_last = x.clone()
        grad = self.forward(x).detach()
        for _ in range(max_steps):
            if grad.reshape(B, -1).norm(dim=-1).max() < s.sample_g_min:
                break
            x_last = x
            x = x - s.sample_eta * grad
            grad = self.forward(x + s.sample_mu * (x - x_last)).detach()
        return x

    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.logsumexp(x, dim=-1, keepdim=True)


@register("EqMConsGrad")
def build_eqm_consgrad(cfg: Config) -> EqMConsGrad:
    from aitchinson_flow.losses import build_loss
    return EqMConsGrad(cfg, loss_fn=build_loss(cfg))
