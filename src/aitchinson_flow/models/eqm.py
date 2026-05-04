from __future__ import annotations
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.transformer_backbone import TransformerBackbone, VelocityHead
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.losses import build_loss


class EquilibriumFlowMatching(nn.Module):
    def __init__(self, cfg: Config, loss_fn: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.backbone = TransformerBackbone(cfg=cfg)
        self.velocity_head = VelocityHead(cfg=cfg)

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._eqm_loss(batch["x"])

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        out = self._eqm_loss(batch["x"])
        if self.cfg.training.eval_bpd:
            out["bpd"] = self.bpd(
                batch["token_ids"], max_steps=self.cfg.training.eval_bpd_max_steps
            )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.velocity_head(self.backbone(x))

    @torch.enable_grad()
    def _eqm_loss(self, x1: torch.Tensor) -> LossDict:
        B, L, D = x1.shape
        device, dt = x1.device, x1.dtype

        # 1. Source Distribution
        x0 = 0.1 * torch.randn(B, L, D, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)  # Stay in V_d

        # 2. Trajectory Interpolation
        gamma = torch.rand(B, device=device, dtype=dt)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1

        x_gamma.requires_grad_(True)

        # 3. Target Gradient (Data-to-Noise direction)
        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        # 4. Forward Pass to get raw velocity predictions
        v = self.forward(x_gamma)

        # 5. Explicit Energy Computation (Dot Product approach)
        # g(x) = x * f(x). We sum the batch to get a single scalar for autograd.
        # Since batch elements are independent, the derivative w.r.t x_gamma[i]
        # correctly isolates to only the i-th sequence.
        energy = (x_gamma * v).sum()

        # 6. Compute the conservative vector field (grad_g)
        # This requires a double backward pass during the actual optimization step,
        # hence create_graph=True.
        grad_g = torch.autograd.grad(
            outputs=energy, inputs=x_gamma, create_graph=True, retain_graph=True
        )[0]

        # 7. Loss Calculation
        # Match the explicit, conservative gradient to the target gradient.
        flow_loss = self.loss_fn(grad_g, u_tgt)
        out: LossDict = {TRAINING_LOSS_KEY: flow_loss}

        # 8. Logging metrics across trajectory phases
        for key, mask in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if mask.any():
                # Ensure we evaluate the mask on grad_g, not the raw v
                out[key] = self.loss_fn(grad_g[mask], u_tgt[mask])

        return out

    def _c_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        strategy = self.cfg.eqm.decay_strategy.strip().lower()
        one_minus_gamma = 1.0 - gamma
        c_gamma = one_minus_gamma

        if strategy == "linear":
            c_gamma = one_minus_gamma

        elif strategy == "truncated":
            a = torch.as_tensor(
                self.cfg.eqm.decay_a, device=gamma.device, dtype=gamma.dtype
            )
            denominator = torch.clamp(1.0 - a, min=1e-7)
            c_gamma = torch.where(
                gamma <= a, torch.ones_like(gamma), one_minus_gamma / denominator
            )

        elif strategy == "piecewise":
            a = torch.as_tensor(
                self.cfg.eqm.decay_a, device=gamma.device, dtype=gamma.dtype
            )
            b = torch.as_tensor(
                self.cfg.eqm.decay_b, device=gamma.device, dtype=gamma.dtype
            )
            left = b - ((b - 1.0) / a) * gamma
            right = one_minus_gamma / (1.0 - a)
            c_gamma = torch.where(gamma <= a, left, right)

        else:
            raise ValueError(
                f"Unknown equilibrium.eqm_decay_strategy={self.cfg.eqm.decay_strategy!r}; "
            )

        c_gamma = c_gamma / torch.as_tensor(
            self.cfg.eqm.gradient_lambda, device=gamma.device, dtype=gamma.dtype
        )

        return c_gamma[:, None, None]

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        eta: float | None = None,
        mu: float | None = None,
        g_min: float | None = None,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """NAG-GD sampling with adaptive stopping (Algorithm 2, EqM paper).

        Iterates x ← x − η·∇E(x + μ(x − x_prev)) until max_batch ||∇E|| < g_min
        or max_steps is reached. Defaults are read from cfg.eqm.

        x_init: optional starting CLR tensor (B, L, K). When None, initialises
                from small centred Gaussian noise (unconditional generation).
        """
        s = self.cfg.eqm
        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        g_min = g_min if g_min is not None else s.sample_g_min
        max_steps = max_steps if max_steps is not None else s.sample_max_steps

        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K

        if x_init is not None:
            x = x_init.to(device)
        else:
            x = 0.01 * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)

        x_last = x.clone()
        grad = self._compute_grad(x)

        for _ in range(max_steps):
            if grad.reshape(B, -1).norm(dim=-1).max() < g_min:
                break
            x_last = x
            x = x - eta * grad
            grad = self._compute_grad(x + mu * (x - x_last))

        return x

    def _compute_grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """CLR features → log-probabilities over the K-character vocab."""
        return x - torch.logsumexp(x, dim=-1, keepdim=True)

    @torch.enable_grad()
    def position_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position gradient norm — proxy for how far each position is from the data manifold.

        High value  → model is pushing hard on this position (uncertain / corrupted).
        Near zero   → position already sits at an energy minimum (confident / clean).

        Returns (B, L) tensor (no_grad context safe to call from outside).
        """
        x_req = x.detach().requires_grad_(True)
        v = self.forward(x_req)
        energy = (x_req * v).sum()
        grad = torch.autograd.grad(energy, x_req)[0]  # (B, L, K)
        return grad.norm(dim=-1)  # (B, L)

    @torch.no_grad()
    def bpd(
        self, token_ids: torch.Tensor, *, max_steps: int | None = None
    ) -> torch.Tensor:
        """Reconstruction bits-per-character.

        Encodes each ground-truth sequence to CLR, adds small Gaussian noise
        to perturb it off the manifold, then runs NAG-GD to recover the nearest
        energy minimum.  NLL of the recovered distribution against the original
        tokens measures how faithfully the model reconstructs the data.

        This is strictly more meaningful than comparing noise-seeded samples to
        specific GT sequences, which inflates BPD as GD converges (the samples
        commit to wrong characters with increasing confidence).
        """
        from aitchinson_flow.data.transforms import token_ids_to_features

        B, L = token_ids.shape
        K = self.cfg.text8_dataset.K
        ls = self.cfg.transformation.label_smoothing
        device = next(self.parameters()).device

        x1 = token_ids_to_features(token_ids.to(device), K, label_smoothing=ls)
        x_init = x1 + 0.1 * torch.randn_like(x1)
        x_init = x_init - x_init.mean(dim=-1, keepdim=True)

        x = self.sample(B, L, max_steps=max_steps, x_init=x_init)
        log_probs = self.decode_to_logprobs(x)
        nll = F.nll_loss(
            log_probs.reshape(-1, log_probs.shape[-1]),
            token_ids.reshape(-1).to(device),
            reduction="mean",
        )
        return nll / math.log(2)


@register("EqM")
def build_eqm(cfg: Config) -> EquilibriumFlowMatching:
    return EquilibriumFlowMatching(cfg, loss_fn=build_loss(cfg))
