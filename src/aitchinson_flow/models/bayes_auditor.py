"""BayesianAuditorAE — GP-energy conservative-gradient EBM with a contrastive
hinge on perturbation-based negatives.

Pipeline:
    token_ids → frozen AE → z₁ (B, L, d_latent)
    f_θ(x, γ) = LatentBackbone(x, γ)             — (B, L, d_latent)
    pool       z_pooled = mean over positions    — (B, d_latent)
    GP head    E_θ(x) = GP_mean(z_pooled(x))     — scalar per sample
    velocity   v_θ(x) = -∇_x E_θ(x)              — conservative

Training (per batch from CorruptingCollate which produces ``token_ids`` and
``token_ids_invalid``):
    • FM regression: target c(γ)·(x₀ − x₁), regress v_θ at x_γ.
    • Contrastive energy hinge: E_θ(x₁) → 0 ; E_θ(x_invalid) > margin.
    • Variance hinge: var(x₁) small ; var(x_invalid) > margin_var.
    • SVGP KL regulariser: KL(q(u) ∥ p(u)) / B  (scaled to per-sample).

This is the supervisor-suggested EBM-via-contrastive-hinge design from your
project's discussions, instantiated in the same AE-latent space the
EqMAE/ScoreDSM/EqMDSM cells use, so it can be compared head-to-head on the
existing recovery / KL diagnostics.

The energy is a Gaussian Process posterior mean — not the bilinear
⟨x, f(x)⟩ form. The conservative-gradient parameterisation is preserved
(autograd through GP_mean(extract(x))) but the function class is now in the
RKHS of the chosen kernel (RBF / Matern). This matches the
``BayesianGenerator`` / ``BayesianAuditor`` code class the supervisor's
prior notes describe.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention import sdpa_kernel, SDPBackend

from aitchinson_flow.config import Config
from aitchinson_flow.data.transforms import token_ids_to_features
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.models.autoencoder import TextAutoencoder
from aitchinson_flow.models.eqm_ae import _LatentBackbone
from aitchinson_flow.models.sparse_gp import _SparseGP
from aitchinson_flow.transformer_backbone import TransformerBackbone


class BayesianAuditorAE(nn.Module):
    """GP-energy conservative EBM trained with FM regression + contrastive hinge."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d_latent = cfg.autoencoder.d_latent

        # Frozen pretrained AE — encode token_ids to z, decode z to logits.
        self.ae = TextAutoencoder(cfg)
        ae_path = cfg.bayes_auditor.ae_ckpt_path or cfg.eqm_ae.ae_ckpt_path
        if not ae_path:
            raise ValueError(
                "cfg.bayes_auditor.ae_ckpt_path (or eqm_ae.ae_ckpt_path) must be set"
            )
        payload = torch.load(ae_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        missing, unexpected = self.ae.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[BayesianAuditorAE] AE load: missing={len(missing)} unexpected={len(unexpected)}; "
                f"first missing: {missing[:3]}; first unexpected: {unexpected[:3]}"
            )
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()
        if not hasattr(self.ae, "_warmup_epoch_seen"):
            self.ae._warmup_epoch_seen = 0

        # LatentBackbone re-used from EqMAE: input (B, L, d_latent) → output (B, L, d_latent).
        self.backbone = _LatentBackbone(cfg=cfg)

        # SVGP head over the *pooled* backbone latent. Pure-PyTorch SVGP
        # (Matern-5/2, non-whitened q(u), adaptive jitter) — chosen over
        # gpytorch because eager computation gives a clean graph for
        # ``create_graph=True`` in the velocity-side double-backward.
        self.gp = _SparseGP(
            d_latent=self.d_latent,
            num_inducing=int(cfg.bayes_auditor.num_inducing),
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.gp.parameters())

    # ----- encode / decode (delegate to frozen AE) ----------------------- #

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.ae.encode(token_ids)

    def decode_to_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z)

    def decode_to_logprobs(self, z: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.decode_to_logits(z), dim=-1)

    # ----- energy / velocity -------------------------------------------- #

    def _extract_pooled(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run backbone, then mean-pool across positions to get a per-sample
        feature for the GP. (B, L, d_latent) → (B, d_latent)."""
        h = self.backbone(x, gamma, pad_mask=pad_mask)
        if pad_mask is None:
            return h.mean(dim=1)
        valid = (~pad_mask).to(h.dtype).unsqueeze(-1)
        return (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def energy_per_sample(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """E_θ(x) = GP_mean(pool(backbone(x))). Returns (B,)."""
        z = self._extract_pooled(x, gamma=gamma, pad_mask=pad_mask)
        return self.gp(z).mean

    def _grad_energy(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        x_req = x if x.requires_grad else x.detach().requires_grad_(True)
        z = self._extract_pooled(x_req, gamma=gamma, pad_mask=pad_mask)
        energy = self.gp(z).mean.sum()
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
        token_ids = batch["token_ids"]
        pad_mask = batch.get("pad_mask")
        device = token_ids.device

        # Positive latents x₁; negative latents x₁_invalid from CorruptingCollate.
        with torch.no_grad():
            x1 = self.ae.encode_sample(token_ids, pad_mask=pad_mask)
        if "token_ids_invalid" in batch:
            with torch.no_grad():
                x1_invalid = self.ae.encode_sample(batch["token_ids_invalid"], pad_mask=pad_mask)
        else:
            x1_invalid = None

        B, L, d = x1.shape
        dt = x1.dtype

        # ----- FM regression (linear interpolant, same as EqMAE) -----
        x0 = eqm_s.source_sigma * torch.randn(B, L, d, device=device, dtype=dt)
        gamma = torch.rand(B, device=device, dtype=dt).pow(eqm_s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        c_g = (1.0 - gamma) / max(float(eqm_s.gradient_lambda), 1e-8)
        u_tgt = c_g[:, None, None] * (x0 - x1)

        # Velocity = -∇ E
        x_gamma_req = x_gamma.detach().requires_grad_(True)
        grad_E = self._grad_energy(x_gamma_req, gamma=gamma, pad_mask=pad_mask, create_graph=True)
        v_pred = -grad_E

        if pad_mask is None:
            flow_loss = F.mse_loss(v_pred, u_tgt)
        else:
            valid = (~pad_mask).to(v_pred.dtype).unsqueeze(-1)
            sq = (v_pred - u_tgt).pow(2) * valid
            denom = (valid.sum() * d).clamp(min=1.0)
            flow_loss = sq.sum() / denom

        out: LossDict = {"flow_loss": flow_loss.detach()}
        total = flow_loss

        # ----- Contrastive hinge on the energy of clean vs invalid -----
        if s.lambda_hinge > 0.0 and x1_invalid is not None:
            E_clean = self.energy_per_sample(x1, gamma=torch.ones(B, device=device, dtype=dt))
            E_invalid = self.energy_per_sample(
                x1_invalid, gamma=torch.ones(B, device=device, dtype=dt)
            )
            hinge_clean = (E_clean.pow(2)).mean()
            hinge_invalid = F.relu(s.margin_energy - E_invalid).mean()
            hinge = hinge_clean + hinge_invalid
            total = total + s.lambda_hinge * hinge
            out["hinge_loss"] = hinge.detach()
            out["E_clean"] = E_clean.mean().detach()
            out["E_invalid"] = E_invalid.mean().detach()

        # ----- KL term over the variational distribution -----
        if s.lambda_kl > 0.0:
            kl = self.gp.kl_divergence() / float(B)
            total = total + s.lambda_kl * kl
            out["kl"] = kl.detach()

        # ----- Aux CE on implied-x₁ (same trick as EqMAE) -----
        if eqm_s.lambda_ce > 0.0:
            ce_mask = gamma >= eqm_s.ce_min_gamma
            if ce_mask.any():
                lam = eqm_s.gradient_lambda
                # implied x1 = x_γ - λ · ∇_x E (negative-velocity formulation)
                pred_x1 = x_gamma[ce_mask] - lam * grad_E[ce_mask]
                ce_pad = pad_mask[ce_mask] if pad_mask is not None else None
                ids = token_ids[ce_mask].long()
                logits = self.decode_to_logits(pred_x1)
                log_probs = F.log_softmax(logits, dim=-1)
                if ce_pad is None:
                    ce = F.nll_loss(log_probs.reshape(-1, self.K), ids.reshape(-1))
                else:
                    valid = (~ce_pad).reshape(-1).to(log_probs.dtype)
                    ce_per = F.nll_loss(
                        log_probs.reshape(-1, self.K),
                        ids.reshape(-1),
                        reduction="none",
                    )
                    ce = (ce_per * valid).sum() / valid.sum().clamp(min=1.0)
                total = total + eqm_s.lambda_ce * ce
                out["ce"] = ce.detach()

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
        eta: float | None = None,
        mu: float | None = None,
        g_min: float | None = None,
        grad_clip: float | None = None,
        return_best: bool | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """NAG-GD on -∇E. Mirrors EqMAE.sample (same hyperparams from
        ``cfg.eqm.sample_*``)."""
        s = self.cfg.eqm
        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        g_min = g_min if g_min is not None else s.sample_g_min
        max_steps = max_steps if max_steps is not None else s.sample_max_steps
        if grad_clip is None:
            grad_clip = s.sample_grad_clip
        if return_best is None:
            return_best = s.sample_return_best

        device = next(self.backbone.parameters()).device
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, self.d_latent, device=device)

        def _clip(g: torch.Tensor) -> torch.Tensor:
            if grad_clip is None:
                return g
            n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n.clamp(max=grad_clip) / n)

        gamma_b = torch.ones(B, device=device, dtype=torch.get_default_dtype())

        # NAG-GD on the energy gradient.
        with torch.enable_grad():
            grad = _clip(self._grad_energy(x, gamma=gamma_b, create_graph=False).detach())
        x_last = x.clone()
        best_x = x.clone()
        best_g = grad.norm(dim=-1).mean().item() if return_best else float("inf")

        for _ in range(int(max_steps)):
            if grad.reshape(B, -1).norm(dim=-1).max() < g_min:
                break
            x_last = x
            x = x - eta * grad
            with torch.enable_grad():
                grad = _clip(
                    self._grad_energy(x + mu * (x - x_last), gamma=gamma_b, create_graph=False).detach()
                )
            if return_best:
                g_mean = grad.norm(dim=-1).mean().item()
                if g_mean < best_g:
                    best_g = g_mean
                    best_x = x.clone()

        return best_x if return_best else x

    # ----- diagnostic UQ surface ---------------------------------------- #

    @torch.no_grad()
    def score_energy(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        B = x.shape[0]
        gamma = torch.ones(B, device=x.device, dtype=x.dtype)
        return self.energy_per_sample(x, gamma=gamma)

    @torch.no_grad()
    def score_variance(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        """GP predictive variance — the native UQ signal of this model class. (B,)"""
        z = self._extract_pooled(x)
        return self.gp(z).variance

    def score_gradient_norm(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        B = x.shape[0]
        gamma = torch.ones(B, device=x.device, dtype=x.dtype)
        with torch.enable_grad():
            grad = self._grad_energy(x, gamma=gamma, create_graph=False)
        return grad.detach().flatten(start_dim=1).norm(dim=-1)

    @torch.no_grad()
    def score_curvature(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        # Pooled-GP energy has near-trivial curvature in x (the variation is
        # mostly in the kernel response); we report NaN to signal "not the
        # native UQ signal — use score_variance instead".
        return torch.full((x.shape[0],), float("nan"), device=x.device, dtype=x.dtype)


@register("BayesianAuditorAE")
def build_bayes_auditor_ae(cfg: Config) -> BayesianAuditorAE:
    return BayesianAuditorAE(cfg)


class BayesianAuditorRaw(nn.Module):
    """Bayes-auditor variant on raw-text CLR features (no AE).

    Replaces the frozen-AE latent path of BayesianAuditorAE with a direct
    TransformerBackbone on per-position CLR features ``x ∈ R^{B,L,K}``.
    Conservative energy and contrastive hinge are unchanged; the GP head
    sits on the mean-pooled backbone hidden state.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d_model = cfg.transformer.d_model

        self.backbone = TransformerBackbone(cfg=cfg)
        self.gp = _SparseGP(
            d_latent=self.d_model,
            num_inducing=int(cfg.bayes_auditor.num_inducing),
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.gp.parameters())

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """For recovery_check compatibility: token_ids → CLR features."""
        ls = self.cfg.transformation.label_smoothing
        return token_ids_to_features(token_ids, self.K, label_smoothing=ls)

    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """CLR features are already pre-softmax logits over K positions."""
        return F.log_softmax(x, dim=-1)

    def _extract_pooled(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with sdpa_kernel(SDPBackend.MATH):
            h = self.backbone(x)
        if pad_mask is None:
            return h.mean(dim=1)
        valid = (~pad_mask).to(h.dtype).unsqueeze(-1)
        return (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def energy_per_sample(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.gp(self._extract_pooled(x, pad_mask=pad_mask)).mean

    def _grad_energy(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        create_graph: bool,
    ) -> torch.Tensor:
        x_req = x if x.requires_grad else x.detach().requires_grad_(True)
        energy = self.energy_per_sample(x_req, pad_mask=pad_mask).sum()
        grad = torch.autograd.grad(
            outputs=energy,
            inputs=x_req,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        # Project to V_d (zero-mean across K) so the gradient stays on the
        # CLR hyperplane.
        grad = grad - grad.mean(dim=-1, keepdim=True)
        return grad

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._loss(batch)

    def eval_step(self, batch: Any) -> LossDict:
        return self._loss(batch)

    @torch.enable_grad()
    def _loss(self, batch: dict[str, torch.Tensor]) -> LossDict:
        s = self.cfg.bayes_auditor
        eqm_s = self.cfg.eqm
        x1 = batch["x"]
        token_ids = batch["token_ids"]
        x1_invalid = batch.get("x_invalid")
        pad_mask = batch.get("pad_mask")
        device = x1.device
        dt = x1.dtype
        B, L, K = x1.shape

        # FM regression on CLR features.
        x0 = eqm_s.source_sigma * torch.randn(B, L, K, device=device, dtype=dt)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)
        gamma = torch.rand(B, device=device, dtype=dt).pow(eqm_s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        c_g = (1.0 - gamma) / max(float(eqm_s.gradient_lambda), 1e-8)
        u_tgt = c_g[:, None, None] * (x0 - x1)

        x_gamma_req = x_gamma.detach().requires_grad_(True)
        grad_E = self._grad_energy(x_gamma_req, pad_mask=pad_mask, create_graph=True)
        v_pred = -grad_E

        if pad_mask is None:
            flow_loss = F.mse_loss(v_pred, u_tgt)
        else:
            valid = (~pad_mask).to(v_pred.dtype).unsqueeze(-1)
            sq = (v_pred - u_tgt).pow(2) * valid
            denom = (valid.sum() * K).clamp(min=1.0)
            flow_loss = sq.sum() / denom

        out: LossDict = {"flow_loss": flow_loss.detach()}
        total = flow_loss

        # Contrastive hinge on energies of clean vs invalid CLR features.
        if s.lambda_hinge > 0.0 and x1_invalid is not None:
            E_clean = self.energy_per_sample(x1, pad_mask=pad_mask)
            E_invalid = self.energy_per_sample(x1_invalid, pad_mask=pad_mask)
            hinge_clean = E_clean.pow(2).mean()
            hinge_invalid = F.relu(s.margin_energy - E_invalid).mean()
            hinge = hinge_clean + hinge_invalid
            total = total + s.lambda_hinge * hinge
            out["hinge_loss"] = hinge.detach()
            out["E_clean"] = E_clean.mean().detach()
            out["E_invalid"] = E_invalid.mean().detach()

        if s.lambda_kl > 0.0:
            kl = self.gp.kl_divergence() / float(B)
            total = total + s.lambda_kl * kl
            out["kl"] = kl.detach()

        # Aux CE on implied-x1 reconstruction. CLR features are pre-softmax
        # logits over K classes, so we apply log_softmax and NLL directly —
        # no learnable decoder needed.
        if eqm_s.lambda_ce > 0.0:
            ce_mask = gamma >= eqm_s.ce_min_gamma
            if ce_mask.any():
                lam = eqm_s.gradient_lambda
                pred_x1 = x_gamma[ce_mask] - lam * grad_E[ce_mask]
                ids = token_ids[ce_mask].long()
                log_probs = F.log_softmax(pred_x1, dim=-1)
                ce_pad = pad_mask[ce_mask] if pad_mask is not None else None
                if ce_pad is None:
                    ce = F.nll_loss(log_probs.reshape(-1, self.K), ids.reshape(-1))
                else:
                    valid = (~ce_pad).reshape(-1).to(log_probs.dtype)
                    ce_per = F.nll_loss(
                        log_probs.reshape(-1, self.K),
                        ids.reshape(-1),
                        reduction="none",
                    )
                    ce = (ce_per * valid).sum() / valid.sum().clamp(min=1.0)
                total = total + eqm_s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total
        return out

    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        eta: float | None = None,
        mu: float | None = None,
        g_min: float | None = None,
        grad_clip: float | None = None,
        return_best: bool | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        s = self.cfg.eqm
        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        g_min = g_min if g_min is not None else s.sample_g_min
        max_steps = max_steps if max_steps is not None else s.sample_max_steps
        if grad_clip is None:
            grad_clip = s.sample_grad_clip
        if return_best is None:
            return_best = s.sample_return_best

        device = next(self.backbone.parameters()).device
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        if x_init is not None:
            x = x_init.to(device).detach()
            x = x - x.mean(dim=-1, keepdim=True)
        else:
            x = sigma * torch.randn(B, L, self.K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)

        def _clip(g: torch.Tensor) -> torch.Tensor:
            if grad_clip is None:
                return g
            n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n.clamp(max=grad_clip) / n)

        with torch.enable_grad():
            grad = _clip(self._grad_energy(x, create_graph=False).detach())
        x_last = x.clone()
        best_x = x.clone()
        best_g = grad.norm(dim=-1).mean().item() if return_best else float("inf")

        for _ in range(int(max_steps)):
            if grad.reshape(B, -1).norm(dim=-1).max() < g_min:
                break
            x_last = x
            x = x - eta * grad
            x = x - x.mean(dim=-1, keepdim=True)
            with torch.enable_grad():
                grad = _clip(
                    self._grad_energy(
                        x + mu * (x - x_last), create_graph=False
                    ).detach()
                )
            if return_best:
                g_mean = grad.norm(dim=-1).mean().item()
                if g_mean < best_g:
                    best_g = g_mean
                    best_x = x.clone()

        return best_x if return_best else x

    @torch.no_grad()
    def score_energy(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        return self.energy_per_sample(x)

    @torch.no_grad()
    def score_variance(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        z = self._extract_pooled(x)
        return self.gp(z).variance

    def score_gradient_norm(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        with torch.enable_grad():
            grad = self._grad_energy(x, create_graph=False)
        return grad.detach().flatten(start_dim=1).norm(dim=-1)

    @torch.no_grad()
    def score_curvature(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        return torch.full((x.shape[0],), float("nan"), device=x.device, dtype=x.dtype)


@register("BayesianAuditorRaw")
def build_bayes_auditor_raw(cfg: Config) -> BayesianAuditorRaw:
    return BayesianAuditorRaw(cfg)
