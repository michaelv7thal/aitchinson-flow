"""EqMDSM — energy-gradient denoising-score-matching on a frozen AE latent.

Same training signal as ``ScoreDSM`` (ε-prediction at multiple noise
levels), but the score is parameterised as the gradient of a scalar
bilinear energy:

    E_θ(x, σ) = ⟨x, f_θ(x, σ)⟩,
    s_θ(x, σ) = -∇_x E_θ(x, σ) = -(f_θ(x, σ) + J_{f_θ}(x, σ)^⊤ x),
    ε̂_θ(x̃, σ) = -σ · s_θ(x̃, σ) = σ · ∇_{x̃} E_θ(x̃, σ).

This is the "should EBMs model the energy or the score?" axis (Salimans &
Ho 2021) instantiated for text in this codebase. It pairs DSM's
multi-modal Bayes-optimum (which avoids the §3 attractor described in
``NOTE_WHY_EBM_INIT_STUCK.md``) with EqM's conservative-field
parameterisation (which preserves the energy-landscape semantics:
recovery, basins, exact log-likelihood up to Z).

Trade-off the experiment is designed to measure:
  * Pro: every iterate has a well-defined energy; recovery / curvature
    diagnostics carry over unchanged.
  * Con: training requires double backward through the backbone
    (``create_graph=True`` to get ∇E inside the loss). Same constraint
    as EqM — the math-attention SDPA backend is enforced by the
    backbone for this reason.
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
from aitchinson_flow.sampling.annealed_langevin import annealed_langevin_sample


class EqMDSM(nn.Module):
    """Energy-gradient parameterisation of a noise-conditional score model."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d_latent = cfg.autoencoder.d_latent

        self.ae = TextAutoencoder(cfg)
        ae_path = cfg.dsm.ae_ckpt_path or cfg.eqm_ae.ae_ckpt_path
        if not ae_path:
            raise ValueError("cfg.dsm.ae_ckpt_path (or eqm_ae.ae_ckpt_path) must be set")
        payload = torch.load(ae_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        missing, unexpected = self.ae.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[EqMDSM] AE load: missing={len(missing)} unexpected={len(unexpected)}; "
                f"first missing: {missing[:3]}; first unexpected: {unexpected[:3]}"
            )
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()
        if not hasattr(self.ae, "_warmup_epoch_seen"):
            self.ae._warmup_epoch_seen = 0

        self.backbone = _LatentBackbone(cfg=cfg)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()

    # ----- encode / decode ------------------------------------------------ #

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.ae.encode(token_ids)

    def decode_to_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z)

    def decode_to_logprobs(self, z: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.decode_to_logits(z), dim=-1)

    # ----- conditioning --------------------------------------------------- #

    def _log_sigma_cond(self, sigma: torch.Tensor) -> torch.Tensor:
        s = self.cfg.dsm
        lo, hi = math.log(s.sigma_min), math.log(s.sigma_max)
        return (sigma.log() - lo) / max(hi - lo, 1e-8)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw backbone output f_θ(x, σ). The score is derived from this
        via the conservative-energy formula in ``score()``."""
        cond = self._log_sigma_cond(sigma) if sigma is not None else None
        return self.backbone(x, cond, pad_mask=pad_mask)

    # ----- energy / score / ε-prediction --------------------------------- #

    def energy_per_sample(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sample scalar energy E_θ(x, σ) = ⟨x, f_θ(x, σ)⟩. Returns (B,)."""
        v = self.forward(x, sigma, pad_mask=pad_mask)
        if pad_mask is None:
            return (x * v).sum(dim=(1, 2))
        valid_f = (~pad_mask).to(x.dtype).unsqueeze(-1)
        return (x * v * valid_f).sum(dim=(1, 2))

    def _grad_energy(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        *,
        create_graph: bool,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """∇_x E_θ(x, σ). Returned with the same shape as ``x``."""
        x_req = x if x.requires_grad else x.detach().requires_grad_(True)
        v = self.forward(x_req, sigma, pad_mask=pad_mask)
        if pad_mask is None:
            energy_total = (x_req * v).sum()
        else:
            valid_f = (~pad_mask).to(x_req.dtype).unsqueeze(-1)
            energy_total = (x_req * v * valid_f).sum()
        grad = torch.autograd.grad(
            outputs=energy_total,
            inputs=x_req,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
        return grad

    def eps_predict(
        self,
        x_noised: torch.Tensor,
        sigma: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """ε̂_θ(x̃, σ) = σ · ∇_{x̃} E_θ(x̃, σ). Differentiable for training
        when ``create_graph=True``."""
        grad = self._grad_energy(x_noised, sigma, create_graph=create_graph, pad_mask=pad_mask)
        return sigma[:, None, None] * grad

    @torch.no_grad()
    def score(self, x_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """s_θ(x̃, σ) = -ε̂_θ / σ = -∇_x E_θ(x, σ). No graph created — for sampling."""
        with torch.enable_grad():
            grad = self._grad_energy(x_noised, sigma, create_graph=False)
        return -grad

    # ----- training step -------------------------------------------------- #

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

        # ε̂ = σ · ∇_{x̃} E_θ(x̃, σ); needs double backward through the
        # backbone to train.
        eps_hat = self.eps_predict(x_noised, sigma, pad_mask=pad_mask, create_graph=True)

        if s.loss_weighting == "constant":
            weight = torch.ones_like(sigma)
        elif s.loss_weighting == "snr+1":
            weight = sigma.pow(2) + 1.0
        else:
            raise ValueError(f"unknown loss_weighting={s.loss_weighting!r}")

        sq = (eps_hat - eps).pow(2)
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

        # Aux CE on implied-x1 = x̃ − σ·ε̂ = x̃ − σ²·∇E. Flows back through
        # the conservative-gradient parameterisation, so the energy gets
        # pushed toward per-token attractors.
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

    # ----- diagnostic UQ surface ----------------------------------------- #
    # Mirrors EqMAE's score_energy / score_gradient_norm / score_curvature.
    # The "reference σ" for evaluation defaults to σ_min — i.e. the energy
    # at the "data-end" of the noise schedule, which is the cleanest
    # analog of EqMAE's gamma=1 energy.

    def _ref_sigma(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (x.shape[0],),
            float(self.cfg.dsm.sigma_min),
            device=x.device,
            dtype=x.dtype,
        )

    @torch.no_grad()
    def score_energy(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        sigma = self._ref_sigma(x)
        return self.energy_per_sample(x, sigma)

    def score_gradient_norm(self, x: torch.Tensor, **_) -> torch.Tensor:  # noqa: ARG002
        sigma = self._ref_sigma(x)
        with torch.enable_grad():
            grad = self._grad_energy(x, sigma, create_graph=False)
        return grad.detach().flatten(start_dim=1).norm(dim=-1)

    def score_curvature(
        self,
        x: torch.Tensor,
        *,
        n_samples: int = 4,
        chunk_size: int = 16,
        **_,
    ) -> torch.Tensor:
        """Hutchinson trace estimate of ∇²_x E. Chunked + cache-cleared to
        keep the second-order-autograd memory bounded."""
        sigma_full = self._ref_sigma(x)
        B = x.shape[0]
        traces = torch.zeros(B, device=x.device, dtype=x.dtype)
        cs = max(1, int(chunk_size))
        for start in range(0, B, cs):
            end = min(B, start + cs)
            x_chunk = x[start:end].detach()
            sigma_chunk = sigma_full[start:end]
            t_chunk = torch.zeros(end - start, device=x.device, dtype=x.dtype)
            for _ in range(n_samples):
                v_probe = (
                    torch.empty_like(x_chunk).uniform_(0.0, 1.0).round_().mul_(2.0).sub_(1.0)
                )
                with torch.enable_grad():
                    x_req = x_chunk.requires_grad_(True)
                    grad = self._grad_energy(
                        x_req, sigma_chunk, create_graph=True
                    )
                    inner = (grad * v_probe).sum()
                    hvp = torch.autograd.grad(inner, x_req, retain_graph=False)[0]
                t_chunk = t_chunk + (hvp.detach() * v_probe).flatten(start_dim=1).sum(dim=-1)
            traces[start:end] = t_chunk / float(n_samples)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return traces


@register("EqMDSM")
def build_eqm_dsm(cfg: Config) -> EqMDSM:
    return EqMDSM(cfg)
