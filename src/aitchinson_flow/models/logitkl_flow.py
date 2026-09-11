"""Logit-KL Flow Matching (arXiv:2411.16821).

Tokens are embedded as scaled one-hot logits ``l_1 = γ_l · onehot(i) ∈ R^K``.
The probability path is linear interpolation in logit space::

    l_0 ~ σ·N(0, I_K)
    l_t = (1 − t)·l_0 + t·l_1,   t ∈ [0, 1]

The denoiser is regressed onto the *clean logits*::

    v̂(l_t, t) ≈ E[l_1 | l_t]   trained via   ‖v̂(l_t, t) − l_1‖²

(Optimal v̂* is the posterior mean of the clean logit given l_t.)
No conservative-gradient autograd, no aux CE — pure clean-logit regression.

Sampling is hybrid (Algorithm 2 of the paper, Sec. 4):

* For ``t < split_t`` (default 0.28): deterministic Euler step in the
  prediction parameterisation. Given v̂_t = E[l_1 | l_t], compute the
  implied source ``z_t = (l_t − t·v̂_t) / (1 − t)`` and project forward to
  ``t' = t + h`` via ``l_{t'} = (1 − t')·z_t + t'·v̂_t``.
* For ``t ≥ split_t``: same projection, plus Gaussian re-noising
  ``l_{t'} ← l_{t'} + σ_{t'}·ε`` with ``σ_t = sqrt(1 − t²)·noise_scale``.
  This keeps the iterate inside the noise scale that the model saw during
  training and is the published recipe's only stochastic ingredient.

Decoding is ``argmax(softmax(l_1_pred))`` from the final-step prediction.

The model uses ``DFMBackbone`` (sinusoidal-t conditioning, MATH SDP backend
not required since there is no second-order autograd). State dict layout:
``backbone.*`` + ``head.*`` matching DFM, so the existing eval and probe
plumbing (which dispatches on ``hasattr(model, 'decode_to_logprobs')``)
treats LogitKLFlow as a discrete-output model — like DFM.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.transformer_backbone import DFMBackbone, DFMHead
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register


class LogitKLFlow(nn.Module):
    """Logit-KL Flow Matching denoiser + hybrid det/stochastic sampler."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DFMBackbone(cfg=cfg)
        self.head = DFMHead(cfg=cfg)

    def forward(self, l_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """l_t: (B, L, K) continuous logits, t: (B,) ∈ [0, 1] → (B, L, K)
        clean-logit prediction v̂(l_t, t)."""
        h = self.backbone(l_t, t)
        return self.head(h)

    def _clean_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """l_1 = γ_l · one_hot(token_ids) — the regression target."""
        K = self.cfg.text8_dataset.K
        gamma_l = self.cfg.logitkl.gamma_l
        return gamma_l * F.one_hot(token_ids.long(), K).to(
            dtype=next(self.parameters()).dtype
        )

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        token_ids = batch["token_ids"].long()  # (B, L)
        B = token_ids.shape[0]
        device = token_ids.device

        l_1 = self._clean_logits(token_ids)  # (B, L, K)
        l_0 = self.cfg.logitkl.source_sigma * torch.randn_like(l_1)

        t = torch.rand(B, device=device, dtype=l_1.dtype)
        l_t = (1.0 - t)[:, None, None] * l_0 + t[:, None, None] * l_1

        v_hat = self.forward(l_t, t)  # (B, L, K)
        loss = F.mse_loss(v_hat, l_1)
        out: LossDict = {"flow_loss": loss.detach(), TRAINING_LOSS_KEY: loss}

        # γ-bucket diagnostics for cross-comparability with EqM/FMonCLR.
        for key, m in (
            ("g<.33", t < 0.33),
            ("g<.66", (t >= 0.33) & (t < 0.66)),
            ("g<1", t >= 0.66),
        ):
            if m.any():
                out[key] = F.mse_loss(v_hat[m], l_1[m]).detach()

        return out

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        return self.training_step(batch, 0)

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        nfe: int | None = None,
        method: str | None = None,
        alpha: float | None = None,
        use_grad: bool | None = None,  # noqa: ARG002 — kept for sample API parity
    ) -> torch.Tensor:
        """Hybrid det-then-stochastic sampler. Returns (B, L) long token IDs.

        Walks t from 0 → 1 in ``nfe`` Euler steps. At each step we predict
        the clean logits, infer the implied source ``z_t``, and project to
        ``t_next``; if ``t_next ≥ split_t`` we add Gaussian noise scaled by
        ``σ_{t_next} = sqrt(1 − t_next²) · noise_scale``.

        Decoding takes argmax of the final-step clean-logit prediction.
        """
        c = self.cfg.logitkl
        if nfe is None:
            nfe = c.sampler_nfe
        K = self.cfg.text8_dataset.K
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        l = c.source_sigma * torch.randn(B, L, K, device=device, dtype=dtype)
        t_grid = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)

        # Phase R "method=sde" branch: the published recipe already injects
        # noise after t≥split_t. Phase R wants a tunable diffusion knob that
        # behaves like additive Langevin at every step (so α=0 is the
        # deterministic-Euler control). We override the published noise
        # schedule with sqrt(2·α·h) ξ at every step when method="sde".
        sde_mode = method == "sde"
        sde_alpha = float(alpha) if (sde_mode and alpha is not None) else 0.0
        h_step = 1.0 / float(nfe)
        sde_noise = float((2.0 * sde_alpha * h_step) ** 0.5) if sde_mode else 0.0

        v_hat = None
        for i in range(nfe):
            t_curr = t_grid[i]
            t_next = t_grid[i + 1]

            t_curr_b = t_curr.expand(B)
            v_hat = self.forward(l, t_curr_b)  # E[l_1 | l_t]

            # Implied source z_t from the linear interp identity.
            if t_curr.item() <= 0.0:
                z_t = l
            else:
                z_t = (l - t_curr * v_hat) / (1.0 - t_curr).clamp(min=1e-6)

            # Project to t_next.
            l = (1.0 - t_next) * z_t + t_next * v_hat

            if sde_mode:
                if sde_noise > 0.0:
                    l = l + sde_noise * torch.randn_like(l)
            elif t_curr.item() >= c.sampler_split_t:
                # Original published stochastic re-noising path.
                sigma_next = (
                    (1.0 - t_next * t_next).clamp(min=0.0).sqrt()
                    * c.sampler_noise_scale
                )
                l = l + sigma_next * torch.randn_like(l)

        # Final clean-logit prediction at t≈1 — use the last v̂ if any, else
        # query at t=1 (degenerate path with nfe=0).
        if v_hat is None:
            t_one = torch.ones(B, device=device, dtype=dtype)
            v_hat = self.forward(l, t_one)
        return v_hat.argmax(dim=-1).long()


@register("LogitKLFlow")
def build_logitkl_flow(cfg: Config) -> LogitKLFlow:
    return LogitKLFlow(cfg)
