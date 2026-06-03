"""SFLMEBM — time-free hyperspherical flow model read as an energy-based model.

Bridges S-FLM (hyperspherical language flow matching, arXiv:2605.11125) and
EqM (equilibrium / energy flow matching, arXiv:2510.02300):

  * Each token has a learned codebook row projected onto ``S^{d-1}``.
  * A *time-free* transformer denoiser ``h_theta(z)`` is trained with
    cross-entropy on SLERP-noised latents (S-FLM path, no time conditioning
    — the EqM move).
  * The implicit energy is the EqM-style log-sum-exp readout
    ``E_theta(z^l) = -tau * logsumexp_v <h_theta(z^l), e_v>/tau``.
  * Generation / recovery is Riemannian gradient descent on ``E_theta``.

The ``scripts/sflm_ebm_probe.py`` study (see ``SFLM_EBM_FINDINGS.md``) found
the energy basin is correct under pure CE (recover≈1.0) — the model is a
strong *verifier / OOD auditor* but a weak unconditional generator. This
class therefore exposes the EqM-family EBM API (``energy``,
``position_uncertainty``, ``score_energy``/``score_gradient_norm``) plus a
paper-style per-position ``spilled_energy`` so it slots into the existing
``scripts/eval_ood.py`` harness alongside EqM / EqMLatent / DFM.

The importance noise schedule and contrastive hinge (``cfg.sflm_ebm``) are
the EqM anti-collapse knobs, kept as one-flag ablations; the headline
behaviour does not depend on them.

Data contract: like ``EqMLatent`` this ignores ``batch["x"]`` (CLR
features) and embeds ``batch["token_ids"]`` internally. When corruption is
enabled (``cfg.text8_dataset.train_corrupt_rate > 0``) the collate adds
``token_ids_invalid``, used as hinge negatives.
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


# --------------------------------------------------------------------------
# Spherical primitives (shared with testing.ipynb / scripts/sflm_ebm_probe.py)
# --------------------------------------------------------------------------
def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def _uniform_sphere(shape, device, dtype=torch.float32) -> torch.Tensor:
    return _normalize(torch.randn(shape, device=device, dtype=dtype))


def _geodesic(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    dot = (p * q).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(dot)


def _slerp(p: torch.Tensor, q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """SLERP p (alpha=0) → q (alpha=1) along the last axis."""
    omega = _geodesic(p, q).unsqueeze(-1)
    sin_omega = torch.sin(omega).clamp(min=1e-7)
    a = alpha.unsqueeze(-1) if alpha.dim() == p.dim() - 1 else alpha
    return (
        torch.sin((1.0 - a) * omega) / sin_omega * p
        + torch.sin(a * omega) / sin_omega * q
    )


def _exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.cos(vn) * p + torch.sin(vn) * (v / vn)


def _project_tangent(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return v - (p * v).sum(dim=-1, keepdim=True) * p


def _log_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``log_p(q)`` — tangent vector at p whose norm equals the geodesic
    distance to q and whose direction is along the great-circle toward q.
    Used to build the Riemannian flow-matching target ``c(α)·log_{z_α}(z₁)``
    (Chen & Lipman 2023) for the option-2 conservative-gradient regression.
    """
    omega = _geodesic(p, q).unsqueeze(-1)
    s = torch.sin(omega).clamp(min=1e-7)
    return (omega / s) * (q - torch.cos(omega) * p)


# --------------------------------------------------------------------------
# Backbone — time-free transformer, S^{d-1} → feature h(z).
# (Mirrors eqm_latent._LatentBackbone but with no γ / context plumbing.)
# --------------------------------------------------------------------------
class _SphereBackbone(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d_model = cfg.transformer.d_model
        d_embed = cfg.sflm_ebm.d_embed
        L = cfg.text8_dataset.L

        self.in_proj = nn.Linear(d_embed, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg.transformer.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Linear(d_model, d_embed)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, _ = z.shape
        h = self.in_proj(z)
        pos = torch.arange(L, device=z.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        # MATH backend: keep parity with the rest of the EqM family (and safe
        # for the first-order autograd used by position_uncertainty).
        with sdpa_kernel(SDPBackend.MATH):
            h = self.transformer(h)
        return self.out_proj(h)


class SFLMEBM(nn.Module):
    def __init__(self, cfg: Config, loss_fn: nn.Module | None = None) -> None:
        super().__init__()
        del loss_fn  # API parity with the simplex factory; unused.
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d = cfg.sflm_ebm.d_embed
        self.tau = cfg.sflm_ebm.tau

        self.codebook = nn.Parameter(torch.randn(self.K, self.d) * 0.1)
        self.backbone = _SphereBackbone(cfg)

    # ----- codebook / encode / decode ------------------------------------- #
    def codebook_normalized(self) -> torch.Tensor:
        return _normalize(self.codebook)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids (B, L) → unit-sphere embeddings (B, L, d)."""
        return self.codebook_normalized()[token_ids.long()]

    def features(self, z: torch.Tensor) -> torch.Tensor:
        return self.backbone(z)

    def decode_to_logits(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, L, d) → (B, L, K) logits = <h(z), e_v> / tau."""
        return (self.features(z) @ self.codebook_normalized().t()) / self.tau

    def decode_to_logprobs(self, z: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.decode_to_logits(z), dim=-1)

    def decode_to_token_ids(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_to_logits(z).argmax(dim=-1)

    # ----- energy / EBM readouts ------------------------------------------ #
    def energy_per_pos(self, z: torch.Tensor) -> torch.Tensor:
        """Per-position implicit energy ``-tau·logsumexp_v <h,e_v>/tau``. (B, L)."""
        scores = self.features(z) @ self.codebook_normalized().t()  # (B, L, K)
        return -self.tau * torch.logsumexp(scores / self.tau, dim=-1)

    @torch.no_grad()
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Sequence-level energy (B,) — the EqM-family OOD readout used by
        ``scripts/eval_ood.py`` (``hasattr(model, 'energy')`` dispatch)."""
        return self.energy_per_pos(z).sum(dim=-1)

    @torch.enable_grad()
    def position_uncertainty(
        self, z: torch.Tensor, *, chunk: int | None = None
    ) -> torch.Tensor:
        """Per-position Riemannian gradient norm of the total energy. (B, L).

        High → the energy is pushing this position hard (off-manifold /
        corrupted); near zero → the position sits in an energy basin. This
        is the per-position spilled-energy analogue used cross-model by
        ``eval_ood`` (``U_pos_mean`` / ``U_pos_max``).

        ``chunk`` bounds peak memory at long L by processing the batch in
        slices — each slice runs its own forward + first-order autograd and
        the caching allocator is flushed between slices. The result is
        *identical* to the full-batch computation: the energy sum is
        separable across sequences (the backbone attends only within a
        sequence), so ``∂/∂z[b]`` depends only on row ``b``. Without this,
        a (B=256, L=256) readout exhausts the 20 GB MIG and the NVML
        allocator asserts (see ``scripts/bench_sflm_ebm.py``)."""
        B, L, _ = z.shape
        if chunk is None:
            # Keep B·L near the (B=256, L=40) footprint the readout was tuned
            # for → ~B=40 per slice at L=256.
            chunk = max(1, (256 * 40) // max(int(L), 1))

        def _one(zb: torch.Tensor) -> torch.Tensor:
            zb = zb.detach().requires_grad_(True)
            e = self.energy_per_pos(zb).sum()
            grad = torch.autograd.grad(e, zb)[0]
            return _project_tangent(zb.detach(), grad).norm(dim=-1)

        if chunk >= B:
            return _one(z)
        outs = []
        for s in range(0, B, chunk):
            outs.append(_one(z[s : s + chunk]))
            if z.is_cuda:
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def spilled_energy(self, z: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Paper-style per-position spilled energy (B, L).

        ``SE(i) = logsumexp_v logits_i - logits_i[token_i]`` — the
        per-position NLL of the actually-placed token. Unlike the
        AR-decoder ``data.wiki._spilled_energy_per_pos`` there is no
        next-token shift (this is a non-causal denoiser scoring the token
        at its own position). Directly comparable across SFLMEBM /
        EqMLatent / DFM since all expose ``decode_to_logits``."""
        logits = self.decode_to_logits(z)  # (B, L, K)
        lse = torch.logsumexp(logits, dim=-1)
        picked = logits.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)
        return lse - picked

    # ----- recovery_check score API --------------------------------------- #
    @torch.no_grad()
    def score_energy(self, z: torch.Tensor) -> torch.Tensor:
        return self.energy(z)

    def score_gradient_norm(self, z: torch.Tensor) -> torch.Tensor:
        return self.position_uncertainty(z).norm(dim=-1)

    # ----- training -------------------------------------------------------- #
    def _sample_alpha(self, B: int, device) -> torch.Tensor:
        s = self.cfg.sflm_ebm
        u = torch.rand(B, 1, device=device)
        if s.alpha_sched == "uniform":
            return s.alpha_lo + (s.alpha_hi - s.alpha_lo) * u
        if s.alpha_sched == "trunc":
            return 0.5 + (s.alpha_hi - 0.5) * u
        if s.alpha_sched == "import":
            return s.alpha_hi * u.pow(0.5)
        raise ValueError(f"unknown sflm_ebm.alpha_sched={s.alpha_sched!r}")

    def _hinge(self, z_pos: torch.Tensor, token_ids_invalid: torch.Tensor | None):
        """Contrastive energy hinge: data energy ``hinge_margin`` below the
        energy of {invalid embeddings, centroid seq, uniform sphere}."""
        s = self.cfg.sflm_ebm
        B, L, d = z_pos.shape
        e_pos = self.energy_per_pos(z_pos).sum(dim=-1)  # (B,)
        negs = [self.energy_per_pos(_uniform_sphere((B, L, d), z_pos.device)).sum(-1)]
        centroid = _normalize(self.codebook_normalized().mean(dim=0))
        negs.append(self.energy_per_pos(centroid.expand(B, L, -1)).sum(dim=-1))
        if token_ids_invalid is not None:
            negs.append(self.energy_per_pos(self.encode(token_ids_invalid)).sum(-1))
        e_neg = torch.cat(negs)
        e_pos_rep = e_pos.repeat(len(negs))
        return torch.relu(s.hinge_margin + e_pos_rep - e_neg).mean()

    @torch.enable_grad()
    def _loss(self, batch: Any) -> LossDict:
        s = self.cfg.sflm_ebm
        token_ids = batch["token_ids"]
        device = token_ids.device
        z1 = self.encode(token_ids)  # (B, L, d) on S^{d-1}
        B, L, d = z1.shape

        z0 = _uniform_sphere(z1.shape, device, z1.dtype)
        alpha = self._sample_alpha(B, device)  # (B, 1)
        z_a = _slerp(z0, z1, alpha.expand(B, L))

        # One features pass — used for CE (logits) and, when λ_fm>0, for the
        # Riemannian-gradient FM target. Skipping double-forward.
        need_fm = s.lambda_fm > 0.0
        if need_fm:
            z_a = z_a.requires_grad_(True)
        feats = self.features(z_a)
        logits = (feats @ self.codebook_normalized().t()) / self.tau
        log_p = F.log_softmax(logits, dim=-1)

        ce_mask = alpha.squeeze(-1) >= s.ce_min_alpha
        if ce_mask.any():
            ce = F.nll_loss(
                log_p[ce_mask].reshape(-1, self.K),
                token_ids[ce_mask].reshape(-1).long(),
            )
        else:  # keep graph connected even if the mask is empty this step
            ce = log_p.sum() * 0.0
        total = ce
        out: LossDict = {"ce": ce.detach()}

        if need_fm:
            # Conservative-gradient FM regression (option 2 —
            # SFLM_EBM_FINDINGS.md). Supervise the Riemannian gradient of
            # the energy to equal the geodesic FM velocity at z_α:
            #     u_tgt = c(α) · log_{z_α}(z₁)        (toward data)
            # so that descending -∇_tan E transports noise→data.
            # create_graph=True ⇒ second-order autograd (the EqM cost).
            E_pos = -self.tau * torch.logsumexp(logits, dim=-1)  # (B, L)
            (grad_E,) = torch.autograd.grad(
                E_pos.sum(), z_a, create_graph=True, retain_graph=True
            )
            g_tan = _project_tangent(z_a, grad_E)
            # c(α) is per-sample; broadcast over L, d.
            if s.fm_c_decay == "linear":
                c_alpha = (1.0 - alpha).unsqueeze(-1)  # (B, 1, 1)
            else:
                raise ValueError(f"sflm_ebm.fm_c_decay={s.fm_c_decay!r}")
            with torch.no_grad():
                u_tgt = c_alpha * _log_map(z_a.detach(), z1)
            loss_fm = F.mse_loss(-g_tan, u_tgt)
            total = total + s.lambda_fm * loss_fm
            out["fm"] = loss_fm.detach()
            out["grad_E_norm"] = g_tan.norm(dim=-1).mean().detach()

        if s.lambda_hinge > 0.0:
            hinge = self._hinge(z1, batch.get("token_ids_invalid"))
            total = total + s.lambda_hinge * hinge
            out["hinge"] = hinge.detach()

        # Diagnostics: energy margin data vs centroid (the SFLM_EBM_FINDINGS
        # primary metric — wants E_data < E_centroid).
        with torch.no_grad():
            e_data = self.energy_per_pos(z1).mean()
            centroid = _normalize(self.codebook_normalized().mean(dim=0))
            e_cent = self.energy_per_pos(centroid.expand(B, L, -1)).mean()
            out["E_data"] = e_data
            out["E_margin"] = (e_cent - e_data).detach()  # >0 ⇒ basin correct

        out[TRAINING_LOSS_KEY] = total
        return out

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._loss(batch)

    def eval_step(self, batch: Any) -> LossDict:
        return self._loss(batch)

    # ----- sampling: Riemannian GD on the energy -------------------------- #
    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        target_step: float | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Adaptive-step Riemannian gradient descent on the total energy.

        ``x_init`` may be off-sphere (recovery_check perturbs encoded tokens
        with ambient Gaussian noise) — it is renormalised onto S^{d-1}
        first. Extra EqM-sampler kwargs (eta/mu/...) are accepted and
        ignored for API compatibility with the runner / recovery_check."""
        s = self.cfg.sflm_ebm
        steps = max_steps if max_steps is not None else s.sample_steps
        tgt = target_step if target_step is not None else s.sample_target_step
        device = next(self.parameters()).device

        if x_init is not None:
            z = _normalize(x_init.to(device).detach())
        else:
            z = _uniform_sphere((B, L, self.d), device)

        for _step in range(steps):
            z = z.detach().requires_grad_(True)
            with torch.enable_grad():
                e = self.energy_per_pos(z).sum()
                g = torch.autograd.grad(e, z)[0]
            g = _project_tangent(z.detach(), g)
            gn = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            eta = (tgt / gn).clamp(max=1.0)
            z = _normalize(_exp_map(z.detach(), -eta * g))
        return z.detach()

    @torch.no_grad()
    def bpd(
        self, token_ids: torch.Tensor, *, max_steps: int | None = None
    ) -> torch.Tensor:
        """Reconstruction bits-per-char: encode GT → perturb off-sphere →
        Riemannian-GD recover → per-position NLL of the decoded dist."""
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


@register("SFLMEBM")
def build_sflm_ebm(cfg: Config) -> SFLMEBM:
    return SFLMEBM(cfg)
