"""PerTokenBayesianAuditorWiki — GP-energy EBM with a product kernel over
(per-token text latent, GPT-2 context latent).

Operates on the WikiText-2 / GPT-2 cache produced by ``scripts/cache_wiki.py``
and served by ``WikiAuditorDataset``. Each batch carries:

    x, x_invalid               — (B, L, K=64) per-position top-K CLR features
    token_ids, token_ids_invalid — (B, L)
    h_clean, h_invalid          — (B, L, H=768) GPT-2 last-hidden-state

Model:

    z_j = pos_proj(backbone(x)_j)                        # (B, L, d_latent)
    h_j = ctx_proj(h_clean_j).detach()                   # (B, L, d_latent_ctx)
    E(x, h) = Σ_j GP_mean(z_j, h_j)        — product kernel
    v(x, h) = -∇_x E(x, h)                  — gradient flows only through z

Training:
    • FM regression on -∇E with linear-interpolant target c(γ)(x₀ − x₁).
    • Contrastive energy hinge: E(x, h) → 0; E(x_invalid, h_invalid) > margin.
    • SVGP KL regulariser.

The product kernel says: two (z, h) pairs are similar iff their text
features are similar AND their context features are similar. So the GP can
represent text-token plausibility *conditioned on the LM's local
distribution*, instead of a single one-size-fits-all "is this CLR vector
plausible?" judgement.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.models.sparse_gp import _ProductSparseGP
from aitchinson_flow.transformer_backbone import TransformerBackbone


class PerTokenBayesianAuditorWiki(nn.Module):
    """Per-token GP-energy EBM on the WikiText-2/GPT-2 cache."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K  # K=64 for the wiki cache
        self.L = cfg.text8_dataset.L

        # TransformerBackbone reads x (B, L, K) → h (B, L, d_model).
        # We deliberately keep ``context_features="off"`` so GPT-2 context
        # enters only through the GP's product kernel, NOT through the
        # backbone — the velocity gradient flows only through the text path.
        from dataclasses import replace

        ctx_cfg = replace(cfg, eqm=replace(cfg.eqm, context_features="off"))
        self.backbone = TransformerBackbone(ctx_cfg)
        self.pos_proj = nn.Linear(cfg.transformer.d_model, cfg.transformer.d_latent)

        # GPT-2 context projector. h is detached at use-time.
        ctx_hidden = int(cfg.bayes_auditor.ctx_hidden)
        ctx_dim = int(cfg.bayes_auditor.ctx_proj_dim)
        self.ctx_proj = nn.Linear(ctx_hidden, ctx_dim)

        # Product-kernel SVGP.
        self.gp = _ProductSparseGP(
            d_tok=cfg.transformer.d_latent,
            d_ctx=ctx_dim,
            num_inducing=int(cfg.bayes_auditor.num_inducing),
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.pos_proj.parameters()) + \
               list(self.ctx_proj.parameters()) + list(self.gp.parameters())

    # ----- featurisation ------------------------------------------------- #

    def _z_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, L, K) → z (B, L, d_latent). Math SDPA so second-order
        autograd works through the attention path."""
        with sdpa_kernel(SDPBackend.MATH):
            h = self.backbone(x)
        return self.pos_proj(h)

    def _h_per_token(self, h_ctx: torch.Tensor) -> torch.Tensor:
        """h_ctx (B, L, ctx_hidden) → ctx (B, L, ctx_proj_dim). Always
        called inside ``no_grad`` — context is a conditioning signal, not a
        function the velocity differentiates through."""
        with torch.no_grad():
            return self.ctx_proj(h_ctx.detach())

    # ----- energy / velocity --------------------------------------------- #

    def energy_per_sample(
        self,
        x: torch.Tensor,
        h_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """E(x, h) = Σ_j GP_mean(z_j, h_j). Returns (B,)."""
        B, L, _ = x.shape
        z = self._z_per_token(x)
        h = self._h_per_token(h_ctx)
        out = self.gp(z.reshape(B * L, -1), h.reshape(B * L, -1))
        return out.mean.view(B, L).sum(dim=1)

    def _grad_energy(
        self,
        x: torch.Tensor,
        h_ctx: torch.Tensor,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        x_req = x if x.requires_grad else x.detach().requires_grad_(True)
        energy = self.energy_per_sample(x_req, h_ctx).sum()
        grad = torch.autograd.grad(
            outputs=energy,
            inputs=x_req,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return grad

    # ----- training step ------------------------------------------------- #

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._loss(batch)

    def eval_step(self, batch: Any) -> LossDict:
        return self._loss(batch)

    @torch.enable_grad()
    def _loss(self, batch: dict[str, torch.Tensor]) -> LossDict:
        s = self.cfg.bayes_auditor
        eqm_s = self.cfg.eqm

        x1 = batch["x"]                       # (B, L, K) CLR features
        x1_invalid = batch.get("x_invalid")
        h_clean = batch.get("h_clean")
        h_invalid = batch.get("h_invalid", h_clean)
        if h_clean is None:
            raise ValueError(
                "PerTokenBayesianAuditorWiki requires GPT-2 features in batch['h_clean']; "
                "load via WikiAuditorDataset with with_hidden=True."
            )

        B, L, K = x1.shape
        device = x1.device
        dt = x1.dtype

        # ----- FM regression on the CLR text features -----
        x0 = eqm_s.source_sigma * torch.randn(B, L, K, device=device, dtype=dt)
        # Project x0 to V_d (zero-mean on the K axis) so it lives on the CLR hyperplane.
        x0 = x0 - x0.mean(dim=-1, keepdim=True)

        gamma = torch.rand(B, device=device, dtype=dt).pow(eqm_s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        c_g = (1.0 - gamma) / max(float(eqm_s.gradient_lambda), 1e-8)
        u_tgt = c_g[:, None, None] * (x0 - x1)

        x_gamma_req = x_gamma.detach().requires_grad_(True)
        grad_E = self._grad_energy(x_gamma_req, h_clean, create_graph=True)
        v_pred = -grad_E
        flow_loss = F.mse_loss(v_pred, u_tgt)

        out: LossDict = {"flow_loss": flow_loss.detach()}
        total = flow_loss

        # ----- Contrastive hinge: E(clean) → 0, E(invalid) > margin -----
        if s.lambda_hinge > 0.0 and x1_invalid is not None:
            E_clean = self.energy_per_sample(x1, h_clean)
            E_invalid = self.energy_per_sample(x1_invalid, h_invalid)
            hinge_clean = E_clean.pow(2).mean()
            hinge_invalid = F.relu(s.margin_energy - E_invalid).mean()
            hinge = hinge_clean + hinge_invalid
            total = total + s.lambda_hinge * hinge
            out["hinge_loss"] = hinge.detach()
            out["E_clean"] = E_clean.mean().detach()
            out["E_invalid"] = E_invalid.mean().detach()

        # ----- KL term -----
        if s.lambda_kl > 0.0:
            kl = self.gp.kl_divergence() / float(B)
            total = total + s.lambda_kl * kl
            out["kl"] = kl.detach()

        out[TRAINING_LOSS_KEY] = total
        return out

    # ----- sampling ------------------------------------------------------ #

    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        h_ctx: torch.Tensor | None = None,
        eta: float | None = None,
        mu: float | None = None,
        grad_clip: float | None = None,
        return_best: bool | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """NAG-GD on -∇E. Requires ``h_ctx`` of shape (B, L, ctx_hidden) for
        unconditional sampling; pass via kwargs from the eval harness.

        If no h_ctx is provided, we use a zero-vector context (i.e. an
        "uninformative" context). This is unconditional sampling without
        a real LM prompt — the result will be GP-prior-shaped at the
        context end, which is fine as a diagnostic."""
        s = self.cfg.eqm
        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        max_steps = max_steps if max_steps is not None else s.sample_max_steps
        if grad_clip is None:
            grad_clip = s.sample_grad_clip
        if return_best is None:
            return_best = s.sample_return_best

        device = next(self.backbone.parameters()).device
        K = self.K
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma

        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)

        if h_ctx is None:
            ctx_hidden = int(self.cfg.bayes_auditor.ctx_hidden)
            h_ctx = torch.zeros(B, L, ctx_hidden, device=device)

        def _clip(g: torch.Tensor) -> torch.Tensor:
            if grad_clip is None:
                return g
            n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n.clamp(max=grad_clip) / n)

        with torch.enable_grad():
            grad = _clip(self._grad_energy(x, h_ctx, create_graph=False).detach())
        x_last = x.clone()
        best_x = x.clone()
        best_g = grad.norm(dim=-1).mean().item() if return_best else float("inf")

        for _ in range(int(max_steps)):
            x_last = x
            x = x - eta * grad
            x = x - x.mean(dim=-1, keepdim=True)  # stay on V_d
            with torch.enable_grad():
                grad = _clip(
                    self._grad_energy(
                        x + mu * (x - x_last), h_ctx, create_graph=False
                    ).detach()
                )
            if return_best:
                g_mean = grad.norm(dim=-1).mean().item()
                if g_mean < best_g:
                    best_g = g_mean
                    best_x = x.clone()

        return best_x if return_best else x

    # ----- decode (degenerate — for argmax slot recovery diagnostic) ----- #

    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """Top-K CLR features → log-probs over the K slots (softmax of CLR,
        not of full vocab logits). For full-vocab decode you'd consult
        ``cache["clean_topk_idx"][i]`` to map slot → vocab id."""
        return F.log_softmax(x, dim=-1)

    # ----- OOD scoring API ---------------------------------------------- #

    @torch.no_grad()
    def ood_score(
        self,
        x: torch.Tensor,
        h_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token (energy, variance) — the native GP-EBM diagnostic.
        Returns ((B, L), (B, L))."""
        B, L, _ = x.shape
        z = self._z_per_token(x)
        h = self._h_per_token(h_ctx)
        out = self.gp(z.reshape(B * L, -1), h.reshape(B * L, -1))
        return out.mean.view(B, L), out.variance.view(B, L)


@register("PerTokenBayesianAuditorWiki")
def build_per_token_bayes_auditor_wiki(cfg: Config) -> PerTokenBayesianAuditorWiki:
    return PerTokenBayesianAuditorWiki(cfg)
