"""SFLMSvgp — Stage-2 SVGP head on a frozen :class:`SFLM` for OOD detection.

Mirror of :class:`DirichletFMSvgp` (``dirichlet_fm_svgp.py``) on the
hyperspherical / SLERP path:

  Stage 1 (separate, ``training_step`` here just delegates):
    Train an :class:`SFLM` (time-conditioned hyperspherical denoiser)
    end-to-end via :mod:`scripts.train_for_sflm_bench`.  Saves an
    ``epoch_final.pt`` checkpoint.

  Stage 2 (``fit_svgp_hinge``):
    Freeze the trained SFLM.  Pool its time-conditioned hidden states at
    a mid-path γ ∈ (0, 1) into a single feature vector per sequence,
    then train a Matérn-5/2 SVGP (post-hoc) together with a trainable
    scalar energy head, via the contrastive energy hinge against
    ``token_ids_invalid`` from :class:`CorruptingCollate`.  The SVGP's
    latent posterior mean, mapped through a Bernoulli likelihood, is the
    sequence-level OOD probability used at inference (mirrors the
    `DFM_SVGP_FINDINGS.md` recipe).

Why mid-path γ for the pooled features?
    γ=0 (pure noise) gives features with no token information; γ=1
    (data) gives features dominated by codebook-nearest-neighbour and
    little contextual disagreement signal — exactly the degeneracy that
    made SFLM's seq-level spilled energy at γ=0.95 collapse to chance
    in `SFLM_EBM_FINDINGS.md`.  Mid-path (e.g. ``cfg.sflm.eval_gamma``
    set to ~0.5 during fit) is where the denoiser is doing real
    contextual reasoning.

Config:  re-uses ``cfg.dfm_svgp`` for everything except the path time,
    which is read from ``cfg.sflm.eval_gamma`` (γ ∈ [0, 1]) — semantically
    different from DFM's t ∈ [1, t_max].  Reusing keeps the SVGP knobs
    (kernel, pooling, n_inducing, margin_energy, lambda_hinge) directly
    comparable across DFM-SVGP and SFLM-SVGP.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, register
from aitchinson_flow.models.sflm import SFLM, _uniform_sphere, _slerp
# Re-use the proven SequencePooler + _SVGPHead from the DFM-SVGP variant
# so SFLM-SVGP and DFM-SVGP share identical heads — apples-to-apples.
from aitchinson_flow.models.dirichlet_fm_svgp import SequencePooler, _SVGPHead


class SFLMSvgp(nn.Module):
    """SFLM + post-hoc SVGP head for sequence-level OOD detection."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.sflm = SFLM(cfg=cfg)

        dfs = cfg.dfm_svgp                              # config reuse — see docstring
        d_embed = cfg.sflm.d_embed                       # input dim of the pooler
        self.pooler = SequencePooler(
            d_model=d_embed, d_embed=dfs.d_embed, mode=dfs.pooling
        )
        self.svgp = _SVGPHead(
            d_model=dfs.d_embed,
            n_inducing=int(dfs.n_inducing),
            kernel=dfs.kernel,
        )
        # Trainable scalar energy head — discriminative hinge target.
        # SVGP stays in the model for post-hoc UQ readouts.
        self.energy_head = nn.Linear(dfs.d_embed, 1)

        # Default: pooler/SVGP frozen during Stage 1 (SFLM trains alone).
        for p in self.pooler.parameters():
            p.requires_grad_(False)

    # ----- Stage 1 — delegate to the SFLM denoiser ----------------------
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.sflm.encode(token_ids)

    def decode_to_logits(self, z, gamma=None):
        return self.sflm.decode_to_logits(z, gamma)

    def decode_to_logprobs(self, z, gamma=None):
        return self.sflm.decode_to_logprobs(z, gamma)

    def training_step(self, batch: Any, step: int) -> LossDict:
        return self.sflm.training_step(batch, step)

    def eval_step(self, batch: Any) -> LossDict:
        return self.sflm.eval_step(batch)

    @torch.no_grad()
    def sample(self, B: int, L: int, **kwargs: Any) -> torch.Tensor:
        return self.sflm.sample(B, L, **kwargs)

    @torch.no_grad()
    def bpd(self, token_ids: torch.Tensor, *, max_steps: int | None = None):
        return self.sflm.bpd(token_ids, max_steps=max_steps)

    # ----- Feature pooling at a mid-path γ ------------------------------
    def _z_at_gamma(
        self, token_ids: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """SLERP-sample a noised input at γ for token_ids; tangent-friendly
        for gradient training of the pooler / energy_head during Stage 2."""
        z1 = self.sflm.encode(token_ids.long())          # (B, L, d) on S^{d-1}
        z0 = _uniform_sphere(z1.shape, z1.device, z1.dtype)
        B, L = token_ids.shape
        alpha = gamma.view(B, 1).expand(B, L)
        return _slerp(z0, z1, alpha)

    def pool_features(
        self, token_ids: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """token_ids (B, L), γ (B,) → pooled features (B, d_embed).

        Mirrors :meth:`DirichletFMSvgp.pool_features`.  The backbone is
        always run with grad enabled when ``self.training`` and the pooler
        is trainable (Stage-2 hinge phase); otherwise it's a no-grad
        feature extractor (post-hoc fit / inference)."""
        z_g = self._z_at_gamma(token_ids, gamma)
        train_backbone = self.training and any(
            p.requires_grad for p in self.pooler.parameters()
        )
        with torch.set_grad_enabled(train_backbone):
            h = self.sflm.backbone(z_g, gamma)            # (B, L, d_embed)
            z = self.pooler(h)
        return z

    # ----- Inference-time OOD scores ------------------------------------
    @torch.no_grad()
    def ood_score(
        self, token_ids: torch.Tensor, gamma: float | None = None
    ) -> dict[str, torch.Tensor]:
        """Per-sequence OOD readouts.  Returns:

          ``energy`` — trained scalar energy head (B,).
          ``prob``   — sigmoid(SVGP latent mean) under Bernoulli likelihood (B,).
          ``std``    — SVGP latent predictive std (B,) — concentrated, kept
                       for honest reporting per ``DFM_SVGP_FINDINGS.md``.
        """
        device = next(self.parameters()).device
        if gamma is None:
            gamma = float(self.cfg.sflm.eval_gamma)
        B = token_ids.shape[0]
        g = torch.full((B,), float(gamma), device=device)
        z = self.pool_features(token_ids.to(device), g)
        e = self.energy_head(z).squeeze(-1)
        # SVGP posterior — Bernoulli-mapped probability is the OOD score.
        # Variance is reported but typically uninformative (concentration
        # of measure; see DFM_SVGP_FINDINGS.md).
        try:
            self.svgp.gp.eval()
            self.svgp.likelihood.eval()
            dist = self.svgp.gp(z)
            prob = torch.sigmoid(dist.mean)
            std = dist.variance.sqrt()
        except Exception:                                # noqa: BLE001
            prob = torch.sigmoid(e)                       # fallback if SVGP unfit
            std = torch.zeros_like(prob)
        return {"energy": e, "prob": prob, "std": std}

    # ----- Stage 2 — hinge-trained SVGP on pooled mid-γ features --------
    def fit_svgp_hinge(
        self,
        loader: Iterable[dict],
        *,
        gamma_eval: float | None = None,
        n_epochs: int = 5,
        max_steps: int | None = None,
        lr: float = 1e-3,
        margin_energy: float | None = None,
        train_pooler: bool = True,
        verbose: bool = True,
        lengthscale_init: float | None = None,
    ) -> dict[str, Any]:
        """Mirror of :meth:`DirichletFMSvgp.fit_svgp_hinge` but on γ-paths.
        Freezes SFLM, trains pooler + energy_head + SVGP via the
        contrastive energy hinge on ``batch['token_ids_invalid']``."""
        import gpytorch as _gp
        dfs = self.cfg.dfm_svgp
        if gamma_eval is None:
            gamma_eval = float(self.cfg.sflm.eval_gamma)
        if margin_energy is None:
            margin_energy = dfs.margin_energy
        device = next(self.parameters()).device

        # 1) Freeze SFLM. Unfreeze pooler + energy_head + SVGP.
        for p in self.sflm.parameters():
            p.requires_grad_(False)
        if train_pooler:
            for p in self.pooler.parameters():
                p.requires_grad_(True)
        for p in self.energy_head.parameters():
            p.requires_grad_(True)
        for p in self.svgp.gp.parameters():
            p.requires_grad_(True)
        for p in self.svgp.likelihood.parameters():
            p.requires_grad_(True)

        # 2) Re-seed SVGP inducing points + standardisation buffers from a
        # batch of positive features (matches DFM-SVGP fit_svgp_hinge §3).
        warmup_target = max(self.svgp.n_inducing * 4, 1024)
        feats: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for batch in loader:
                ids = batch["token_ids"].to(device).long()
                g = torch.full((ids.shape[0],), gamma_eval, device=device)
                feats.append(self.pool_features(ids, g))
                if sum(f.shape[0] for f in feats) >= warmup_target:
                    break
        Xs = torch.cat(feats, dim=0)
        mu, sigma = Xs.mean(dim=0), Xs.std(dim=0).clamp(min=1e-6)
        self.svgp._mu = mu.detach()
        self.svgp._sigma = sigma.detach()
        perm = torch.randperm(Xs.shape[0], device=device)[: self.svgp.n_inducing]
        with torch.no_grad():
            self.svgp.gp.variational_strategy.inducing_points.copy_(
                ((Xs[perm] - mu) / sigma).detach()
            )
            if lengthscale_init is not None:
                self.svgp.gp.covar_module.base_kernel.lengthscale = float(
                    lengthscale_init
                )

        # 3) Joint training: pooler + energy_head + SVGP under hinge loss.
        self.train()
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        mll = _gp.mlls.VariationalELBO(
            self.svgp.likelihood, self.svgp.gp,
            num_data=max(warmup_target, 1),
        )

        history: list[dict[str, float]] = []
        step = 0
        for ep in range(n_epochs):
            for batch in loader:
                ids = batch["token_ids"].to(device).long()
                if "token_ids_invalid" not in batch:
                    continue
                inv = batch["token_ids_invalid"].to(device).long()
                B = ids.shape[0]
                g = torch.full((B,), gamma_eval, device=device)
                z_clean = self.pool_features(ids, g)
                z_inv = self.pool_features(inv, g)
                e_clean = self.energy_head(z_clean).squeeze(-1)
                e_inv = self.energy_head(z_inv).squeeze(-1)
                hinge = e_clean.pow(2).mean() + F.relu(
                    margin_energy - e_inv
                ).mean()
                # SVGP ELBO on the same pooled positives (latent-mean ≈ 0
                # for in-distribution, ≥ margin for OOD). Targets =
                # Bernoulli labels {0=clean, 1=invalid}.
                z_all = torch.cat([z_clean, z_inv], dim=0)
                z_std = (z_all - self.svgp._mu) / self.svgp._sigma
                y = torch.cat([torch.zeros(B), torch.ones(B)]).to(device)
                out = self.svgp.gp(z_std)
                elbo = -mll(out, y)
                loss = hinge + elbo
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if max_steps and step >= max_steps:
                    break
                if verbose and step % 50 == 0:
                    print(f"[svgp-hinge] ep={ep} step={step:5d} "
                          f"hinge={float(hinge):.4f} elbo={float(elbo):.4f} "
                          f"E_clean={float(e_clean.mean()):.3f} "
                          f"E_inv={float(e_inv.mean()):.3f}")
                history.append(dict(
                    step=step, hinge=float(hinge), elbo=float(elbo),
                    E_clean=float(e_clean.mean()),
                    E_inv=float(e_inv.mean()),
                ))
            if max_steps and step >= max_steps:
                break
        self.eval()
        return {"history": history, "n_steps": step}


@register("SFLMSvgp")
def build_sflm_svgp(cfg: Config) -> SFLMSvgp:
    return SFLMSvgp(cfg)
