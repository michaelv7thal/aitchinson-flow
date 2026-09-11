"""ScoreDSM — direct-score denoising-score-matching on a frozen AE latent.

The backbone outputs the noise-conditional score directly (or, equivalently,
ε-prediction). Training: ε-MSE at log-uniform σ ∈ [σ_min, σ_max]. Sampling:
NCSN-style annealed Langevin (``sampling/annealed_langevin.py``).

This is the standard "diffusion-LM in continuous latent space" recipe (Li
et al. 2022 Diffusion-LM, Han et al. 2023 SSD-LM, Dieleman et al. 2022
CDCD) applied to this codebase's AE latent. Its purpose in the
``sweeps/dsm_vs_eqm.yaml`` 3-cell comparison is to provide a clean
direct-score baseline against which the energy-gradient parameterisation
``EqMDSM`` is measured — that's the A/B/C contribution outlined in
``docs/archive/POSITIONING.md (pre-pivot background; its Claim 2 is contradicted by the paper)``.

Implementation notes:
  * Backbone reused from EqMAE (``_LatentBackbone``); σ-conditioning rides
    on the existing γ-conditioning path with γ := log(σ). Set
    ``cfg.eqm.time_conditioning ∈ {"add", "concat"}`` or σ is ignored.
  * The network output is ε-prediction (``ε̂``); the score is then
    ``s = -ε̂ / σ`` (Tweedie's formula in NCSN form).
  * Loss: ``E[‖ε̂ - ε‖²]`` with optional λ(σ) weighting from
    ``cfg.dsm.loss_weighting`` ("constant" or "snr+1").
  * Aux CE: from x̂_1 = x̃ - σ·ε̂ (the implied clean point), decode and
    NLL against token ids, masked to small-σ samples.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.models.autoencoder import TextAutoencoder
from aitchinson_flow.models.eqm_ae import _LatentBackbone
from aitchinson_flow.sampling.annealed_langevin import (
    annealed_langevin_sample,
    make_sigma_ladder,
)


class ScoreDSM(nn.Module):
    """Direct-score parameterisation of a noise-conditional score model."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d_latent = cfg.autoencoder.d_latent

        # Frozen AE — encode/decode bridges discrete tokens ↔ continuous latents.
        self.ae = TextAutoencoder(cfg)
        ae_path = cfg.dsm.ae_ckpt_path or cfg.eqm_ae.ae_ckpt_path
        if not ae_path:
            raise ValueError("cfg.dsm.ae_ckpt_path (or eqm_ae.ae_ckpt_path) must be set")
        payload = torch.load(ae_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        missing, unexpected = self.ae.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[ScoreDSM] AE load: missing={len(missing)} unexpected={len(unexpected)}; "
                f"first missing: {missing[:3]}; first unexpected: {unexpected[:3]}"
            )
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()
        if not hasattr(self.ae, "_warmup_epoch_seen"):
            self.ae._warmup_epoch_seen = 0

        # Reuse EqMAE backbone — σ-conditioning is routed through the
        # existing γ-embedding path (we pass log(σ) instead of γ).
        self.backbone = _LatentBackbone(cfg=cfg)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()

    # ----- encode / decode (delegate to frozen AE) ----------------------- #

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.ae.encode(token_ids)

    def decode_to_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z)

    def decode_to_logprobs(self, z: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.decode_to_logits(z), dim=-1)

    # ----- noise-level conditioning -------------------------------------- #

    def _log_sigma_cond(self, sigma: torch.Tensor) -> torch.Tensor:
        """The backbone reads its conditioning input as a (B,) tensor passed
        as ``gamma``. We send log(σ) so the sinusoidal embedding spans the
        log-σ range. Normalised to roughly match the [0, 1] γ-range the
        embedding was implicitly tuned for."""
        s = self.cfg.dsm
        lo, hi = math.log(s.sigma_min), math.log(s.sigma_max)
        return (sigma.log() - lo) / max(hi - lo, 1e-8)

    def eps_predict(
        self, x_noised: torch.Tensor, sigma: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ε̂_θ(x̃, σ): network output, same shape as ``x_noised``."""
        cond = self._log_sigma_cond(sigma)
        return self.backbone(x_noised, cond, pad_mask=pad_mask)

    def score(self, x_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """s_θ(x̃, σ) = -ε̂_θ / σ. Same shape as ``x_noised``."""
        eps_hat = self.eps_predict(x_noised, sigma)
        return -eps_hat / sigma[:, None, None].clamp(min=1e-8)

    # ----- training step ------------------------------------------------- #

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._dsm_loss(batch["token_ids"], pad_mask=batch.get("pad_mask"))

    def eval_step(self, batch: Any) -> LossDict:
        return self._dsm_loss(batch["token_ids"], pad_mask=batch.get("pad_mask"))

    @torch.enable_grad()
    def _dsm_loss(
        self,
        token_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> LossDict:
        s = self.cfg.dsm
        device = token_ids.device

        with torch.no_grad():
            x1 = self.ae.encode_sample(token_ids, pad_mask=pad_mask)
        B, L, d = x1.shape
        dt = x1.dtype

        # Sample σ per batch element. Log-uniform is standard for NCSN /
        # Karras-style training.
        if s.sigma_distribution == "log_uniform":
            log_sigma = torch.empty(B, device=device, dtype=dt).uniform_(
                math.log(s.sigma_min), math.log(s.sigma_max)
            )
            sigma = log_sigma.exp()
        elif s.sigma_distribution == "uniform":
            sigma = torch.empty(B, device=device, dtype=dt).uniform_(
                s.sigma_min, s.sigma_max
            )
        else:
            raise ValueError(f"unknown sigma_distribution={s.sigma_distribution!r}")

        eps = torch.randn(B, L, d, device=device, dtype=dt)
        x_noised = x1 + sigma[:, None, None] * eps

        eps_hat = self.eps_predict(x_noised, sigma, pad_mask=pad_mask)

        # Weighted ε-MSE.
        if s.loss_weighting == "constant":
            weight = torch.ones_like(sigma)
        elif s.loss_weighting == "snr+1":
            # λ(σ) = σ²+1 — Karras 2022 EDM-style: balances small/large σ.
            weight = sigma.pow(2) + 1.0
        else:
            raise ValueError(f"unknown loss_weighting={s.loss_weighting!r}")

        sq = (eps_hat - eps).pow(2)  # (B, L, d)
        if pad_mask is None:
            per_sample = sq.flatten(start_dim=1).mean(dim=-1)
        else:
            valid_f = (~pad_mask).to(dt).unsqueeze(-1)
            sq_masked = sq * valid_f
            denom = (valid_f.sum(dim=(1, 2)) * d).clamp(min=1.0)
            per_sample = sq_masked.sum(dim=(1, 2)) / denom
        dsm_loss = (weight * per_sample).mean()

        total = dsm_loss
        out: LossDict = {"dsm_loss": dsm_loss}

        # Aux CE on implied-x1 = x̃ − σ·ε̂. Mask to small σ so the implied
        # x1 is reliable. ce_min_logsig is in log-σ units; default 0.0
        # ⇒ σ ≤ e^0 = 1.0 — but invert the gate to σ ≤ exp(ce_min_logsig).
        if s.lambda_ce > 0.0:
            ce_mask = sigma.log() <= s.ce_min_logsig
            if ce_mask.any():
                x_hat_1 = x_noised[ce_mask] - sigma[ce_mask, None, None] * eps_hat[ce_mask]
                ce_pad = pad_mask[ce_mask] if pad_mask is not None else None
                ids = token_ids[ce_mask].long()
                logits = self.decode_to_logits(x_hat_1)
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
                total = total + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total

        # Diagnostic: σ-bucketed DSM loss. Same buckets as EqM's γ buckets.
        log_sigma_norm = self._log_sigma_cond(sigma)
        for key, mask_b in (
            ("s<.33", log_sigma_norm < 0.33),
            ("s<.66", (log_sigma_norm >= 0.33) & (log_sigma_norm < 0.66)),
            ("s<1", log_sigma_norm >= 0.66),
        ):
            if mask_b.any():
                out[key] = per_sample[mask_b].mean().detach()

        return out

    # ----- sampling ------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        start_sigma: float | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """NCSN-style annealed Langevin. ``max_steps`` is ignored; the
        sampler walks the full σ-ladder (cfg.dsm.n_sigma × cfg.dsm.steps_per_sigma).

        For ``x_init`` (recovery): set ``start_sigma`` to the perturbation
        magnitude; the ladder is restricted to σ ≤ start_sigma."""
        del max_steps
        s = self.cfg.dsm
        device = next(self.backbone.parameters()).device

        def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return self.score(x, sigma)

        return annealed_langevin_sample(
            score_fn,
            shape=(B, L, self.d_latent),
            sigma_min=s.sigma_min,
            sigma_max=s.sigma_max,
            n_sigma=s.n_sigma,
            steps_per_sigma=s.steps_per_sigma,
            eps=s.sampler_eps,
            x_init=x_init,
            start_sigma=start_sigma,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    # ----- diagnostic UQ surface (parallels EqMAE's score_* API) --------- #
    # ScoreDSM has no scalar energy by construction — it is a free vector
    # field. ``score_energy`` returns NaN to signal "undefined"; the
    # gradient norm is the score magnitude evaluated at a reference σ.

    @torch.no_grad()
    def score_energy(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        return torch.full((x.shape[0],), float("nan"), device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def score_gradient_norm(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        sigma = torch.full(
            (x.shape[0],),
            float(self.cfg.dsm.sigma_min),
            device=x.device,
            dtype=x.dtype,
        )
        return self.score(x, sigma).flatten(start_dim=1).norm(dim=-1)

    @torch.no_grad()
    def score_curvature(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        return torch.full((x.shape[0],), float("nan"), device=x.device, dtype=x.dtype)


@register("ScoreDSM")
def build_score_dsm(cfg: Config) -> ScoreDSM:
    return ScoreDSM(cfg)


# --------------------------------------------------------------------------- #
# Simplex-CLR ScoreDSM (no AE) — for the Dirichlet ablation that brackets
# comp_* on the training-signal axis.
# --------------------------------------------------------------------------- #


class ScoreDSM_CLR(nn.Module):
    """Direct-score DSM on raw CLR text features (B, L, K). No AE.

    Mirrors the AE-latent ``ScoreDSM`` but reads ``batch["x"]`` so the
    ``transformation.dirichlet_sampling`` knob applies — which is the
    point: we want a DSM cell that can be ablated against the deterministic
    vs Dirichlet x₁ axis the way ``comp_*`` ablates EqM-FM.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        from aitchinson_flow.transformer_backbone import TransformerBackbone

        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d_model = cfg.transformer.d_model
        self.backbone = TransformerBackbone(cfg=cfg)
        # ε-prediction head: backbone hidden (d_model) → ε in R^K.
        self.eps_head = nn.Linear(self.d_model, self.K)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.backbone.parameters()) + list(self.eps_head.parameters())

    # ----- noise-level conditioning (same as AE ScoreDSM) ----------------- #

    def _log_sigma_cond(self, sigma: torch.Tensor) -> torch.Tensor:
        s = self.cfg.dsm
        lo, hi = math.log(s.sigma_min), math.log(s.sigma_max)
        return (sigma.log() - lo) / max(hi - lo, 1e-8)

    def eps_predict(self, x_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        cond = self._log_sigma_cond(sigma)
        h = self.backbone(x_noised, cond)
        eps = self.eps_head(h)
        # Project to V_d (zero-mean across K) so ε̂ matches the noise we
        # actually injected (which we centre before training).
        return eps - eps.mean(dim=-1, keepdim=True)

    def score(self, x_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return -self.eps_predict(x_noised, sigma) / sigma[:, None, None].clamp(min=1e-8)

    # ----- recovery_check compat ----------------------------------------- #

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        from aitchinson_flow.data.transforms import token_ids_to_features

        ls = self.cfg.transformation.label_smoothing
        return token_ids_to_features(token_ids, self.K, label_smoothing=ls)

    def decode_to_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=-1)

    # ----- training step ------------------------------------------------- #

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._dsm_loss(batch)

    def eval_step(self, batch: Any) -> LossDict:
        return self._dsm_loss(batch)

    @torch.enable_grad()
    def _dsm_loss(self, batch: dict[str, torch.Tensor]) -> LossDict:
        s = self.cfg.dsm
        x1 = batch["x"]
        token_ids = batch["token_ids"]
        B, L, K = x1.shape
        device = x1.device
        dt = x1.dtype

        # σ ~ log-uniform (NCSN standard).
        if s.sigma_distribution == "log_uniform":
            log_sigma = torch.empty(B, device=device, dtype=dt).uniform_(
                math.log(s.sigma_min), math.log(s.sigma_max)
            )
            sigma = log_sigma.exp()
        elif s.sigma_distribution == "uniform":
            sigma = torch.empty(B, device=device, dtype=dt).uniform_(
                s.sigma_min, s.sigma_max
            )
        else:
            raise ValueError(f"unknown sigma_distribution={s.sigma_distribution!r}")

        eps = torch.randn(B, L, K, device=device, dtype=dt)
        eps = eps - eps.mean(dim=-1, keepdim=True)  # noise lives in V_d
        x_noised = x1 + sigma[:, None, None] * eps

        eps_hat = self.eps_predict(x_noised, sigma)

        if s.loss_weighting == "constant":
            weight = torch.ones_like(sigma)
        elif s.loss_weighting == "snr+1":
            weight = sigma.pow(2) + 1.0
        else:
            raise ValueError(f"unknown loss_weighting={s.loss_weighting!r}")

        sq = (eps_hat - eps).pow(2)
        per_sample = sq.flatten(start_dim=1).mean(dim=-1)
        dsm_loss = (weight * per_sample).mean()

        total = dsm_loss
        out: LossDict = {"dsm_loss": dsm_loss}

        # Aux CE on implied-x₁ = x̃ − σ·ε̂. CLR features are pre-softmax
        # logits over K, so we log_softmax directly — no decoder.
        if s.lambda_ce > 0.0:
            ce_mask = sigma.log() <= s.ce_min_logsig
            if ce_mask.any():
                x_hat_1 = (
                    x_noised[ce_mask]
                    - sigma[ce_mask, None, None] * eps_hat[ce_mask]
                )
                ids = token_ids[ce_mask].long()
                log_probs = F.log_softmax(x_hat_1, dim=-1)
                ce = F.nll_loss(log_probs.reshape(-1, K), ids.reshape(-1))
                total = total + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total

        log_sigma_norm = self._log_sigma_cond(sigma)
        for key, mask_b in (
            ("s<.33", log_sigma_norm < 0.33),
            ("s<.66", (log_sigma_norm >= 0.33) & (log_sigma_norm < 0.66)),
            ("s<1", log_sigma_norm >= 0.66),
        ):
            if mask_b.any():
                out[key] = per_sample[mask_b].mean().detach()

        return out

    # ----- sampling ------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        B: int,
        L: int,
        *,
        max_steps: int | None = None,
        x_init: torch.Tensor | None = None,
        start_sigma: float | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del max_steps
        s = self.cfg.dsm
        device = next(self.backbone.parameters()).device

        def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            return self.score(x, sigma)

        x = annealed_langevin_sample(
            score_fn,
            shape=(B, L, self.K),
            sigma_min=s.sigma_min,
            sigma_max=s.sigma_max,
            n_sigma=s.n_sigma,
            steps_per_sigma=s.steps_per_sigma,
            eps=s.sampler_eps,
            x_init=x_init,
            start_sigma=start_sigma,
            device=device,
            dtype=torch.get_default_dtype(),
        )
        # Project final sample back to V_d.
        return x - x.mean(dim=-1, keepdim=True)

    # ----- diagnostic API ------------------------------------------------ #

    @torch.no_grad()
    def score_energy(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        return torch.full((x.shape[0],), float("nan"), device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def score_gradient_norm(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        sigma = torch.full(
            (x.shape[0],),
            float(self.cfg.dsm.sigma_min),
            device=x.device,
            dtype=x.dtype,
        )
        return self.score(x, sigma).flatten(start_dim=1).norm(dim=-1)

    @torch.no_grad()
    def score_curvature(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        return torch.full((x.shape[0],), float("nan"), device=x.device, dtype=x.dtype)


@register("ScoreDSM_CLR")
def build_score_dsm_clr(cfg: Config) -> ScoreDSM_CLR:
    return ScoreDSM_CLR(cfg)
