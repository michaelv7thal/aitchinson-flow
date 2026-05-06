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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.velocity_head(self.backbone(x))

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._eqm_loss(batch["x"], token_ids=batch.get("token_ids"))

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        out = self._eqm_loss(batch["x"], token_ids=batch.get("token_ids"))
        if self.cfg.training.eval_bpd:
            out["bpd"] = self.bpd(
                batch["token_ids"], max_steps=self.cfg.training.eval_bpd_max_steps
            )
        return out

    @torch.enable_grad()
    def _eqm_loss(
        self, x1: torch.Tensor, *, token_ids: torch.Tensor | None = None
    ) -> LossDict:
        B, L, D = x1.shape
        device, dt = x1.device, x1.dtype
        s = self.cfg.eqm

        # 1. Source distribution — same σ as inference (cfg.eqm.source_sigma).
        x0 = s.source_sigma * torch.randn(B, L, D, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)  # stay in V_d

        # 2. γ importance sampling — push mass toward γ ≈ 1 (where signal lives).
        gamma = torch.rand(B, device=device, dtype=dt).pow(s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1

        x_gamma.requires_grad_(True)

        # 3. Target gradient (data-to-noise direction).
        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        # 4. Forward pass.
        v = self.forward(x_gamma)

        # 5. Conservative gradient via autograd of E(x) = ⟨x, f(x)⟩.
        energy = (x_gamma * v).sum()
        grad_g = torch.autograd.grad(
            outputs=energy, inputs=x_gamma, create_graph=True, retain_graph=True
        )[0]

        # 6. Flow loss — regress conservative gradient to FM target.
        flow_loss = self.loss_fn(grad_g, u_tgt)
        total_loss = flow_loss
        out: LossDict = {"flow_loss": flow_loss}

        # 7. Aux CE on implied-x1 reconstruction (linear decay: x1 ≈ x_γ − λ·grad_g).
        # Anchors per-token attractors so unconditional sampling doesn't collapse to
        # the unigram mode. Only applied where γ ≥ ce_min_gamma (the signal regime).
        if s.lambda_ce > 0.0 and token_ids is not None:
            ce_mask = gamma >= s.ce_min_gamma
            if ce_mask.any():
                lam = s.gradient_lambda
                pred_x1 = x_gamma[ce_mask] - lam * grad_g[ce_mask]
                log_probs = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
                ce = F.nll_loss(
                    log_probs.reshape(-1, D), token_ids[ce_mask].reshape(-1).long()
                )
                total_loss = total_loss + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total_loss

        # 8. γ-bucket diagnostics.
        for key, mask in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if mask.any():
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
        grad_clip: float | None = None,
        return_best: bool | None = None,
    ) -> torch.Tensor:
        """NAG-GD sampling with adaptive stopping (Algorithm 2, EqM paper).

        Iterates x ← x − η·∇E(x + μ(x − x_prev)) until max_batch ||∇E|| < g_min
        or max_steps is reached. Defaults are read from cfg.eqm.

        x_init: optional starting CLR tensor (B, L, K). When None, initialises
                from a centred Gaussian with σ = cfg.eqm.source_sigma so the
                inference x0 distribution matches the training source.
        grad_clip: per-position L2 cap on ∇E. Prevents the cold-start
                spike at the noise init from being amplified by NAG momentum
                into a basin overshoot. None disables.
        return_best: when True, return the lowest mean-‖∇E‖ iterate seen
                across the trajectory rather than the final iterate. NAG
                reliably overshoots the basin around step 60 on this model;
                the best iterate is what should leave the sampler.
        """
        s = self.cfg.eqm
        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        g_min = g_min if g_min is not None else s.sample_g_min
        max_steps = max_steps if max_steps is not None else s.sample_max_steps
        if grad_clip is None:
            grad_clip = s.sample_grad_clip
        if return_best is None:
            return_best = s.sample_return_best

        def _clip(g: torch.Tensor) -> torch.Tensor:
            if grad_clip is None:
                return g
            n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n.clamp(max=grad_clip) / n)

        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K

        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = s.source_sigma * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)

        x_last = x.clone()
        grad = _clip(self._compute_grad(x))

        best_x = x.clone()
        best_g = grad.norm(dim=-1).mean().item() if return_best else float("inf")

        for _ in range(max_steps):
            if grad.reshape(B, -1).norm(dim=-1).max() < g_min:
                break
            x_last = x
            x = x - eta * grad
            grad = _clip(self._compute_grad(x + mu * (x - x_last)))
            if return_best:
                g_mean = grad.norm(dim=-1).mean().item()
                if g_mean < best_g:
                    best_g = g_mean
                    best_x = x.clone()

        return best_x if return_best else x

    def _compute_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Conservative gradient ∇_x ⟨x, f(x)⟩ — matches the training target."""
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            energy = (x_req * self.forward(x_req)).sum()
            grad = torch.autograd.grad(energy, x_req, create_graph=False)[0]
        return grad.detach()

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
