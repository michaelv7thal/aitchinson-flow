"""EqMLatent — EqM with a learned token embedding instead of the CLR simplex.

The pivot from ``models/eqm.py``: instead of running flow matching on the
CLR-coded probability simplex (where all token pairs are equidistant and the
target field is multi-valued at vertex ambiguities), embed each token into
``R^{d_embed}`` via a learnable ``nn.Embedding`` layer trained jointly with
the flow model. The model decodes generated points back to logits via a
*tied* readout (``z @ embed.weight.T``) — the embedding learns to be both
flow-friendly *and* linearly separable for the CE auxiliary.

What stays the same as the simplex EqM:
    * conservative-gradient parameterisation: model is ``∇_x ⟨x, f(x)⟩``
    * NAG-GD and Euler samplers (now in d-dim, not K-dim)
    * γ-bucket diagnostics and the aux CE on the implied-x1 reconstruction
    * EBM-style energy / position uncertainty / OOD readouts

What changes:
    * Source noise lives in ``R^{d_embed}`` (centred Gaussian, no V_d projection)
    * The dataset's precomputed CLR features in ``batch["x"]`` are unused;
      embedding lookup happens *inside* training_step
    * ``decode_to_logprobs`` no longer does ``x − logsumexp(x)`` on CLR; it
      maps z → tied logits → log_softmax
    * MSE on the regression (no Hilbert/Aitchison distinction — the geometry
      is now whatever the embedding learns)
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


class _LatentBackbone(nn.Module):
    """Mirror of ``TransformerBackbone`` but with input projection from
    ``d_embed`` instead of K, and no LM-context plumbing.

    Kept as a separate class (not a parametrised version of the existing
    backbone) to avoid complicating the well-tested simplex EqM path.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d_model = cfg.transformer.d_model
        d_embed = cfg.embedding.d_embed
        L = cfg.text8_dataset.L

        self.input_proj = nn.Linear(d_embed, d_model)
        self.pos_emb = nn.Embedding(L, d_model)

        self._time_mode = getattr(cfg.eqm, "time_conditioning", "off")
        if self._time_mode not in ("off", "add", "concat"):
            raise ValueError(
                f"unknown eqm.time_conditioning={self._time_mode!r}; "
                "expected 'off', 'add', or 'concat'"
            )
        if self._time_mode == "add":
            self.t_proj = nn.Linear(d_model, d_model)
        elif self._time_mode == "concat":
            self.t_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.t_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg.transformer.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)

        if self._time_mode != "off":
            if gamma is None:
                gamma = torch.ones(B, device=x.device, dtype=x.dtype)
            t_emb = _sinusoidal_embedding(gamma, h.shape[-1])
            if self._time_mode == "add":
                h = h + self.t_proj(t_emb)[:, None, :]
            else:
                t_broadcast = t_emb[:, None, :].expand(B, L, -1)
                h = self.t_proj(torch.cat([h, t_broadcast], dim=-1))

        with sdpa_kernel(SDPBackend.MATH):
            return self.transformer(h)


class _LatentVelocityHead(nn.Module):
    """Velocity output head: ``d_model → d_embed``, no V_d centering.

    The simplex EqM's ``VelocityHead`` subtracts the mean across K to keep
    the output in the zero-mean subspace; that's specific to CLR features
    on the simplex. In learned-embedding space there's no analogous
    constraint, so we drop the centering.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.transformer.d_model, cfg.embedding.d_embed)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class EquilibriumFlowMatchingLatent(nn.Module):
    """EqM in learned-embedding latent space.

    See module docstring for the conceptual diff vs ``EquilibriumFlowMatching``.
    """

    def __init__(self, cfg: Config, loss_fn: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        K = cfg.text8_dataset.K
        d = cfg.embedding.d_embed
        self.K = K
        self.d = d

        self.embed = nn.Embedding(K, d)
        init_std = float(cfg.embedding.init_std_factor) / math.sqrt(d)
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

        # Pre-computed fixed embeddings (PPMI-SVD or skip-gram). When loaded,
        # the embedding layer is frozen so x_1 = embed(token) is a *fixed*
        # target throughout training — the flow regression no longer chases a
        # moving target.
        self._embed_fixed: bool = False
        fixed_path = getattr(cfg.embedding, "fixed_path", None)
        if fixed_path:
            payload = torch.load(fixed_path, map_location="cpu", weights_only=False)
            w = payload["embeddings"] if isinstance(payload, dict) else payload
            if w.shape != self.embed.weight.shape:
                raise ValueError(
                    f"fixed embedding shape {tuple(w.shape)} does not match "
                    f"(K={K}, d={d}) = {tuple(self.embed.weight.shape)}"
                )
            with torch.no_grad():
                self.embed.weight.copy_(w.to(self.embed.weight.dtype))
            self.embed.weight.requires_grad_(False)
            self._embed_fixed = True

        self.backbone = _LatentBackbone(cfg=cfg)
        self.velocity_head = _LatentVelocityHead(cfg=cfg)

        if cfg.embedding.tie_decoder:
            self.decoder: nn.Module | None = None  # use tied embed.weight
        else:
            self.decoder = nn.Linear(d, K, bias=False)

        # Plain MSE on the regression; loss_fn is accepted for API parity
        # with the simplex factory but ignored.
        del loss_fn
        self.loss_fn = nn.MSELoss()

        # Step counter for embedding freeze-warmup. Updated by training_step
        # with the global ``step`` argument when provided.
        self._step: int = 0
        self._frozen_now: bool = False
        if cfg.embedding.freeze_steps > 0:
            self.embed.weight.requires_grad_(False)
            self._frozen_now = True

    # ----- encode / decode ------------------------------------------------- #

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids (B, L) → embeddings (B, L, d_embed)."""
        return self.embed(token_ids.long())

    def decode_to_logits(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, L, d) → (B, L, K) logits via tied or untied readout."""
        if self.decoder is None:
            return z @ self.embed.weight.t()
        return self.decoder(z)

    def decode_to_logprobs(self, z: torch.Tensor) -> torch.Tensor:
        """For API parity with simplex EqM (consumed by eval scripts and the
        unigram-KL probe in the runner)."""
        return F.log_softmax(self.decode_to_logits(z), dim=-1)

    def decode_to_token_ids(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_to_logits(z).argmax(dim=-1)

    # ----- forward / energy ------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity_head(self.backbone(x, gamma))

    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Sequence-level energy ``E(x) = ⟨x, f(x)⟩``. (B,)."""
        s = self.cfg.eqm
        time_cond = getattr(s, "time_conditioning", "off") != "off"
        gamma = (
            torch.full((x.shape[0],), float(s.sample_gamma),
                       device=x.device, dtype=x.dtype)
            if time_cond else None
        )
        v = self.forward(x, gamma)
        return (x * v).sum(dim=(1, 2))

    @torch.enable_grad()
    def position_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position grad-norm of ``E(x) = ⟨x, f(x)⟩`` (B, L)."""
        x_req = x.detach().requires_grad_(True)
        v = self.forward(x_req)
        energy = (x_req * v).sum()
        grad = torch.autograd.grad(energy, x_req)[0]
        return grad.norm(dim=-1)

    # ----- training step --------------------------------------------------- #

    def training_step(self, batch: Any, step: int) -> LossDict:
        # Optional embedding warmup: keep frozen for the first N steps then
        # release. Skipped entirely when ``_embed_fixed`` is set
        # (cfg.embedding.fixed_path) — those embeddings stay frozen forever.
        if (
            self._frozen_now
            and not self._embed_fixed
            and step >= self.cfg.embedding.freeze_steps
        ):
            self.embed.weight.requires_grad_(True)
            self._frozen_now = False
        out = self._eqm_loss(batch["token_ids"])
        return out

    def eval_step(self, batch: Any) -> LossDict:
        out = self._eqm_loss(batch["token_ids"])
        return out

    @torch.enable_grad()
    def _eqm_loss(self, token_ids: torch.Tensor) -> LossDict:
        s = self.cfg.eqm
        device = token_ids.device

        x1 = self.encode(token_ids)  # (B, L, d) — gradients flow into self.embed
        B, L, d = x1.shape
        dt = x1.dtype

        # 1. Source distribution in R^d (no V_d projection).
        x0 = s.source_sigma * torch.randn(B, L, d, device=device, dtype=dt)

        # 2. γ importance sampling (unchanged from simplex EqM).
        gamma = torch.rand(B, device=device, dtype=dt).pow(s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        x_gamma.requires_grad_(True)

        # 3. Linear-decay FM target with optional gradient_lambda rescale.
        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        # 4. Conservative gradient via autograd of E = ⟨x, f(x)⟩.
        v = self.forward(x_gamma, gamma)
        energy = (x_gamma * v).sum()
        grad_g = torch.autograd.grad(
            outputs=energy, inputs=x_gamma,
            create_graph=True, retain_graph=True,
        )[0]

        # 5. Plain MSE — geometry is now learned.
        flow_loss = F.mse_loss(grad_g, u_tgt)
        total_loss = flow_loss
        out: LossDict = {"flow_loss": flow_loss}

        # 6. Aux CE on implied-x1 → tied/untied logits → NLL against token_ids.
        if s.lambda_ce > 0.0:
            ce_mask = gamma >= s.ce_min_gamma
            if ce_mask.any():
                lam = s.gradient_lambda
                pred_x1 = x_gamma[ce_mask] - lam * grad_g[ce_mask]  # (M, L, d)
                logits = self.decode_to_logits(pred_x1)              # (M, L, K)
                log_probs = F.log_softmax(logits, dim=-1)
                ids = token_ids[ce_mask].long()
                ce = F.nll_loss(log_probs.reshape(-1, self.K), ids.reshape(-1))
                total_loss = total_loss + s.lambda_ce * ce
                out["ce"] = ce.detach()

        out[TRAINING_LOSS_KEY] = total_loss

        # 7. γ-bucket diagnostics — same buckets as the simplex EqM.
        for key, mask in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if mask.any():
                out[key] = F.mse_loss(grad_g[mask], u_tgt[mask])

        # 8. Embedding-collapse diagnostics — track whether different rows of
        # the embedding stay distinct.
        with torch.no_grad():
            w = self.embed.weight
            norms = w.norm(dim=-1)
            out["embed.norm_mean"] = norms.mean().detach()
            d_pairs = torch.cdist(w, w)
            d_pairs.fill_diagonal_(float("inf"))
            out["embed.pairwise_min"] = d_pairs.min().detach()
            out["embed.pairwise_mean"] = (
                d_pairs.where(d_pairs.isfinite(), torch.zeros_like(d_pairs))
                .sum()
                / (self.K * (self.K - 1))
            ).detach()

        return out

    def _c_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        """Linear/truncated/piecewise decay — copied from simplex EqM."""
        s = self.cfg.eqm
        strategy = s.decay_strategy.strip().lower()
        one_minus_gamma = 1.0 - gamma

        if strategy == "linear":
            c = one_minus_gamma
        elif strategy == "truncated":
            a = torch.as_tensor(s.decay_a, device=gamma.device, dtype=gamma.dtype)
            denominator = torch.clamp(1.0 - a, min=1e-7)
            c = torch.where(gamma <= a, torch.ones_like(gamma),
                            one_minus_gamma / denominator)
        elif strategy == "piecewise":
            a = torch.as_tensor(s.decay_a, device=gamma.device, dtype=gamma.dtype)
            b = torch.as_tensor(s.decay_b, device=gamma.device, dtype=gamma.dtype)
            left = b - ((b - 1.0) / a) * gamma
            right = one_minus_gamma / (1.0 - a)
            c = torch.where(gamma <= a, left, right)
        else:
            raise ValueError(f"unknown decay_strategy={s.decay_strategy!r}")

        c = c / torch.as_tensor(s.gradient_lambda, device=gamma.device, dtype=gamma.dtype)
        return c[:, None, None]

    # ----- sampling ----------------------------------------------------------- #

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
        method: str | None = None,
    ) -> torch.Tensor:
        """NAG-GD or Euler sampling in R^{d_embed}. Mirrors simplex EqM API."""
        s = self.cfg.eqm
        chosen = method if method is not None else getattr(s, "sampler", "nag")
        if chosen == "euler":
            nfe = max_steps if max_steps is not None else s.euler_nfe
            return self._sample_euler(B, L, nfe=nfe, x_init=x_init)

        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        g_min = g_min if g_min is not None else s.sample_g_min
        max_steps = max_steps if max_steps is not None else s.sample_max_steps
        if grad_clip is None:
            grad_clip = s.sample_grad_clip
        if return_best is None:
            return_best = s.sample_return_best

        device = next(self.parameters()).device

        def _clip(g: torch.Tensor) -> torch.Tensor:
            if grad_clip is None:
                return g
            n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n.clamp(max=grad_clip) / n)

        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = s.source_sigma * torch.randn(B, L, self.d, device=device)

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
        s = self.cfg.eqm
        time_cond = getattr(s, "time_conditioning", "off")
        if time_cond != "off":
            B = x.shape[0]
            gamma = torch.full((B,), float(s.sample_gamma),
                               device=x.device, dtype=x.dtype)
        else:
            gamma = None
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            energy = (x_req * self.forward(x_req, gamma)).sum()
            grad = torch.autograd.grad(energy, x_req, create_graph=False)[0]
        return grad.detach()

    def _sample_euler(
        self,
        B: int,
        L: int,
        *,
        nfe: int,
        x_init: torch.Tensor | None,
    ) -> torch.Tensor:
        """Euler integrator over γ on the *conservative gradient* of the model's
        energy ``E(x;γ) = ⟨x, f(x;γ)⟩``.

        EqM is trained so that ``∇_x⟨x, f(x;γ)⟩ ≈ c(γ)·(x₀−x₁)``; the FM target
        is the conservative gradient, *not* the raw model output ``f``. So the
        sampler must descend ``∇_x⟨x, f⟩``. Walking the raw ``f`` is a different
        vector field with no training-time supervision and gives degenerate
        samples (``f`` ≈ 0 fits the regression trivially via skew-symmetric
        Jacobian; only the gradient pulls toward data).
        """
        s = self.cfg.eqm
        device = next(self.parameters()).device
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, self.d, device=device)

        time_cond = getattr(s, "time_conditioning", "off")
        gammas = torch.linspace(0.0, 1.0, nfe + 1, device=device)[:-1]
        h = 1.0 / nfe
        for g in gammas:
            g_b = g.expand(B) if time_cond != "off" else None
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                energy = (x_req * self.forward(x_req, g_b)).sum()
                v = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
            x = x - h * v
        return x

    # ----- BPD ---------------------------------------------------------------- #

    @torch.no_grad()
    def bpd(self, token_ids: torch.Tensor, *, max_steps: int | None = None) -> torch.Tensor:
        """Reconstruction bits-per-character.

        Encodes the GT tokens to embeddings, perturbs to push slightly off the
        manifold, runs NAG-GD to recover the nearest energy minimum, decodes
        the recovered point to logits, and reports per-position NLL.
        """
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        z1 = self.encode(token_ids)
        z_init = z1 + 0.1 * torch.randn_like(z1)

        B, L, _ = z1.shape
        z = self.sample(B, L, max_steps=max_steps, x_init=z_init)
        log_probs = self.decode_to_logprobs(z)
        nll = F.nll_loss(
            log_probs.reshape(-1, self.K),
            token_ids.reshape(-1),
            reduction="mean",
        )
        return nll / math.log(2)


@register("EqMLatent")
def build_eqm_latent(cfg: Config) -> EquilibriumFlowMatchingLatent:
    if not cfg.embedding.enabled:
        # Tolerate calling with embedding.enabled=False; some scripts read
        # ``cfg.training.model_name`` before flipping the flag.
        pass
    return EquilibriumFlowMatchingLatent(cfg)
