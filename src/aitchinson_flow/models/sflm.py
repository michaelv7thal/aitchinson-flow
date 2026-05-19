"""SFLM — proper Stage-1 hyperspherical flow language generator.

S-FLM (arXiv:2605.11125 in spirit) on text8: tokens have a learned
codebook on ``S^{d-1}``, a *time-conditioned* transformer denoiser is
trained with plain cross-entropy on SLERP-noised latents (γ ~ importance-
sampled in [α_lo, α_hi]), and sampling is Euler-over-γ on the sphere via
**x1-prediction + geodesic SLERP step**.

This is the generative counterpart of :class:`SFLMEBM`: same backbone
geometry, but time-conditioning is *kept* (the EqM-move SFLMEBM gave up)
so the sampler integrates a real noise→data transport instead of
gradient-descending a static energy. The post-hoc OOD head (Stage 2) is
intended to mirror :class:`DirichletFMSvgp` — train an SVGP on pooled
hidden states with a contrastive hinge against ``token_ids_invalid`` —
fitted via a separate script once Stage-1 KL is competitive with DFM.

The model exposes the EqM-family inference API
(``encode`` / ``decode_to_logits`` / ``decode_to_logprobs`` /
``sample`` / ``bpd``) so it drops into ``scripts/bench_sflm_ebm.py`` and
``scripts/recovery_check.py`` without per-model code paths. There is
deliberately *no* ``energy()`` — SFLM is not an EBM; the benchmark's
sequence score falls back to mean per-position spilled energy (denoiser
NLL at ``eval_gamma``), exactly as for DFM.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.transformer_backbone import _sinusoidal_embedding


# ---- spherical primitives (shared idiom across sflm_ebm.py / probe) ---- #
def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def _uniform_sphere(shape, device, dtype=torch.float32) -> torch.Tensor:
    return _normalize(torch.randn(shape, device=device, dtype=dtype))


def _geodesic(p, q):
    return torch.arccos((p * q).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def _slerp(p, q, alpha):
    omega = _geodesic(p, q).unsqueeze(-1)
    s = torch.sin(omega).clamp(min=1e-7)
    a = alpha.unsqueeze(-1) if alpha.dim() == p.dim() - 1 else alpha
    return torch.sin((1.0 - a) * omega) / s * p + torch.sin(a * omega) / s * q


# ---- time-conditioned sphere backbone ---- #
class _TimeCondSphereBackbone(nn.Module):
    """S^{d-1} → feature h(z, γ), with γ injected via sinusoidal embedding.

    Mirrors :class:`aitchinson_flow.models.eqm_latent._LatentBackbone`'s
    ``time_conditioning`` plumbing ("add" | "concat") so SFLM matches the
    rest of the codebase's idiom.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d_model = cfg.transformer.d_model
        d_embed = cfg.sflm.d_embed
        L = cfg.text8_dataset.L

        self.in_proj = nn.Linear(d_embed, d_model)
        self.pos_emb = nn.Embedding(L, d_model)

        self._tcond = cfg.sflm.time_conditioning
        if self._tcond not in ("add", "concat"):
            raise ValueError(f"sflm.time_conditioning={self._tcond!r}")
        if self._tcond == "add":
            self.t_proj = nn.Linear(d_model, d_model)
        else:
            self.t_proj = nn.Linear(2 * d_model, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg.transformer.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Linear(d_model, d_embed)

    def forward(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        B, L, _ = z.shape
        h = self.in_proj(z)
        pos = torch.arange(L, device=z.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        t_emb = _sinusoidal_embedding(gamma, h.shape[-1])  # (B, d_model)
        if self._tcond == "add":
            h = h + self.t_proj(t_emb)[:, None, :]
        else:
            h = self.t_proj(
                torch.cat([h, t_emb[:, None, :].expand(B, L, -1)], dim=-1)
            )
        with sdpa_kernel(SDPBackend.MATH):
            h = self.transformer(h)
        return self.out_proj(h)


class SFLM(nn.Module):
    def __init__(self, cfg: Config, loss_fn: nn.Module | None = None) -> None:
        super().__init__()
        del loss_fn
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d = cfg.sflm.d_embed
        self.tau = cfg.sflm.tau
        self.codebook = nn.Parameter(torch.randn(self.K, self.d) * 0.1)
        self.backbone = _TimeCondSphereBackbone(cfg)

    # ----- codebook / encode / decode --------------------------------- #
    def codebook_normalized(self) -> torch.Tensor:
        return _normalize(self.codebook)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.codebook_normalized()[token_ids.long()]

    def _logits(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z, gamma)
        return (h @ self.codebook_normalized().t()) / self.tau

    def decode_to_logits(
        self, z: torch.Tensor, gamma: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        """Bench / recovery_check call ``decode_to_logits(z)`` with no γ;
        in that case we evaluate at ``cfg.sflm.eval_gamma`` (signal-regime
        equivalent of DFM's t≈1)."""
        if gamma is None:
            gamma = self.cfg.sflm.eval_gamma
        if not torch.is_tensor(gamma):
            gamma = torch.full((z.shape[0],), float(gamma),
                               device=z.device, dtype=z.dtype)
        return self._logits(z, gamma)

    def decode_to_logprobs(
        self, z: torch.Tensor, gamma: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        return F.log_softmax(self.decode_to_logits(z, gamma), dim=-1)

    def decode_to_token_ids(
        self, z: torch.Tensor, gamma: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        return self.decode_to_logits(z, gamma).argmax(dim=-1)

    # ----- training: plain CE on SLERP-noised latents ----------------- #
    def _sample_alpha(self, B: int, device) -> torch.Tensor:
        s = self.cfg.sflm
        u = torch.rand(B, 1, device=device)
        if s.alpha_sched == "uniform":
            return s.alpha_lo + (s.alpha_hi - s.alpha_lo) * u
        if s.alpha_sched == "import":
            return s.alpha_hi * u.pow(0.5)
        raise ValueError(f"unknown sflm.alpha_sched={s.alpha_sched!r}")

    def _loss(self, batch: Any) -> LossDict:
        token_ids = batch["token_ids"]
        device = token_ids.device
        z1 = self.encode(token_ids)
        B, L, d = z1.shape
        z0 = _uniform_sphere(z1.shape, device, z1.dtype)
        alpha = self._sample_alpha(B, device)              # (B, 1) in [0, α_hi]
        z_a = _slerp(z0, z1, alpha.expand(B, L))
        gamma = alpha.squeeze(-1)                          # (B,)
        log_p = F.log_softmax(self._logits(z_a, gamma), dim=-1)
        ce = F.nll_loss(
            log_p.reshape(-1, self.K), token_ids.reshape(-1).long()
        )
        return {"ce": ce.detach(), TRAINING_LOSS_KEY: ce}

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._loss(batch)

    def eval_step(self, batch: Any) -> LossDict:
        return self._loss(batch)

    # ----- sampling: Euler-over-γ on the sphere via x1-prediction ----- #
    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Generate by integrating SLERP from γ=0→1: at each step predict
        ``ẑ₁`` from the time-conditioned denoiser (soft expectation over
        the codebook) and take a geodesic SLERP step toward it.

        ``x_init`` (e.g. perturbed-data init from ``recovery_check.py``)
        is renormalised onto S^{d-1} and treated as the γ=0 point."""
        nfe = max_steps if max_steps is not None else self.cfg.sflm.sample_nfe
        device = next(self.parameters()).device
        z = (_normalize(x_init.to(device).detach())
             if x_init is not None
             else _uniform_sphere((B, L, self.d), device))
        e_hat = self.codebook_normalized()  # (K, d)
        gammas = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for k in range(nfe):
            g = gammas[k].expand(B)
            p = F.softmax(self._logits(z, g), dim=-1)        # (B, L, K)
            z1_hat = _normalize(p @ e_hat)                   # (B, L, d)
            # remaining-fraction SLERP step toward ẑ₁
            remain = 1.0 / max(nfe - k, 1)
            step = torch.full((B, L), remain, device=device)
            z = _slerp(z, z1_hat, step)
        return z

    @torch.no_grad()
    def bpd(
        self, token_ids: torch.Tensor, *, max_steps: int | None = None
    ) -> torch.Tensor:
        """Reconstruction BPD analog: encode → light off-sphere perturb →
        sample-from-init → per-position NLL of the decoded distribution."""
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        z1 = self.encode(token_ids)
        z_init = _normalize(z1 + 0.1 * torch.randn_like(z1))
        B, L, _ = z1.shape
        z = self.sample(B, L, max_steps=max_steps, x_init=z_init)
        log_probs = self.decode_to_logprobs(z)
        nll = F.nll_loss(
            log_probs.reshape(-1, self.K),
            token_ids.reshape(-1),
            reduction="mean",
        )
        return nll / math.log(2)


@register("SFLM")
def build_sflm(cfg: Config) -> SFLM:
    return SFLM(cfg)
