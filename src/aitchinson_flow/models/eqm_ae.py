"""EqMAE — Equilibrium Flow Matching in a frozen pretrained-AE latent space.

This is the AE-pretraining counterpart to ``EqMLatent``. The difference:

  * ``EqMLatent`` learns an ``nn.Embedding(K, d_embed)`` jointly with the flow
    (or loads it from a fixed PPMI/skip-gram artifact via ``embedding.fixed_path``).
    The "encoder" is a per-token lookup; positions are independent.
  * ``EqMAE`` loads a frozen ``TextAutoencoder`` whose encoder is
    *contextual* — z[t] depends on the surrounding tokens. The flow runs in
    that contextual latent space.

The motivation: a contextual AE pre-trained on the corpus learns where text
"lives" in latent space; the flow only has to learn the basin structure on
that manifold, not invent the manifold itself. The decoder is what
guarantees "valid text" at sample time, not the energy field.

Wiring:
    cfg.training.model_name = "EqMAE"
    cfg.eqm_ae.ae_ckpt_path = "runs/textae_d128/epoch_final.pt"

The EqM mechanics (conservative-gradient parameterisation, NAG-GD sampling,
γ-bucket diagnostics, aux CE on implied-x1) are reused unchanged from
``EqMLatent`` — only the encode/decode functions swap to the AE.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.models.autoencoder import TextAutoencoder
from aitchinson_flow.transformer_backbone import _sinusoidal_embedding


class _BigramHeadLatent(nn.Module):
    """Non-factorised bigram joint head for EqMAE.

    Reads adjacent implied-x1 latents (the same vectors the aux CE decodes)
    and emits K² logits per position pair, parameterising P(token_t, token_{t+1}
    | implied_x1[t], implied_x1[t+1]). Unlike the factorised log p(a)+log p(b)
    construction (a no-op architecturally — see eqm.py Phase 5 note), this can
    represent any bigram joint and therefore *can* push the encoder/velocity to
    encode digraph structure that survives back into the conservative gradient.

    Memory at K=27, d_latent=128, B=64, L=128:
        Linear params: 2·128·729 ≈ 187K
        Logits tensor: (B, L-1, K²) ≈ 24 MB.
    """

    def __init__(self, d_latent: int, K: int) -> None:
        super().__init__()
        self.K = K
        self.proj = nn.Linear(2 * d_latent, K * K)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (M, L, d_latent) → (M, L-1, K²)."""
        pairs = torch.cat([z[:, :-1], z[:, 1:]], dim=-1)
        return self.proj(pairs)


class _LatentBackbone(nn.Module):
    """Same shape as ``eqm_latent._LatentBackbone`` but with input/output dims
    driven by ``cfg.autoencoder.d_latent`` instead of ``cfg.embedding.d_embed``."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d_model = cfg.transformer.d_model
        d_latent = cfg.autoencoder.d_latent
        L = cfg.text8_dataset.L

        self.input_proj = nn.Linear(d_latent, d_model)
        self.pos_emb = nn.Embedding(L, d_model)

        self._time_mode = getattr(cfg.eqm, "time_conditioning", "off")
        if self._time_mode not in ("off", "add", "concat"):
            raise ValueError(f"unknown eqm.time_conditioning={self._time_mode!r}")
        if self._time_mode == "add":
            self.t_proj = nn.Linear(d_model, d_model)
        elif self._time_mode == "concat":
            self.t_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.t_proj = None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.transformer.nhead,
            dim_feedforward=4 * d_model,
            dropout=cfg.transformer.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=cfg.transformer.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Linear(d_model, d_latent)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
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
                t_b = t_emb[:, None, :].expand(B, L, -1)
                h = self.t_proj(torch.cat([h, t_b], dim=-1))
        with sdpa_kernel(SDPBackend.MATH):
            h = self.transformer(h, src_key_padding_mask=pad_mask)
        return self.out_proj(h)


class EquilibriumFlowMatchingAE(nn.Module):
    """EqM operating in a frozen pretrained-AE latent space."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        self.d_latent = cfg.autoencoder.d_latent

        # Build and load the AE; freeze it. The AE config is implicit in
        # cfg.autoencoder so the architecture must match the pretrained
        # checkpoint's config block.
        self.ae = TextAutoencoder(cfg)
        ae_path = cfg.eqm_ae.ae_ckpt_path
        if not ae_path:
            raise ValueError("cfg.eqm_ae.ae_ckpt_path must be set")
        payload = torch.load(ae_path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        missing, unexpected = self.ae.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[EqMAE] AE load: missing={len(missing)} unexpected={len(unexpected)}; "
                f"first missing: {missing[:3]}; first unexpected: {unexpected[:3]}"
            )
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()

        # Backbone L = max(training.L, autoencoder pos_emb size). Variable-L
        # training requires the backbone's pos_emb table to be sized to L_max.
        self.backbone = _LatentBackbone(cfg=cfg)
        # Initialise the AE's _warmup_epoch_seen attribute so VAE warmup
        # doesn't crash when EqMAE wraps it.
        if not hasattr(self.ae, "_warmup_epoch_seen"):
            self.ae._warmup_epoch_seen = 0

        # Optional non-factorised bigram joint head — only instantiated when
        # eqm.lambda_bigram_joint > 0 so the parameter count and memory cost
        # are paid only when the term is active.
        if getattr(cfg.eqm, "lambda_bigram_joint", 0.0) > 0.0:
            self.bigram_head: _BigramHeadLatent | None = _BigramHeadLatent(
                d_latent=self.d_latent, K=self.K
            )
        else:
            self.bigram_head = None

        # Plain MSE on the regression — geometry is whatever the AE learned.
        self.loss_fn = nn.MSELoss()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        params = list(self.backbone.parameters())
        if self.bigram_head is not None:
            params.extend(self.bigram_head.parameters())
        return params

    # ----- encode / decode (delegate to frozen AE) ------------------------- #

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.ae.encode(token_ids)

    def decode_to_logits(self, z: torch.Tensor) -> torch.Tensor:
        # Note: the AE's decoder uses transformer with no_grad-incompatible
        # dropout-friendly path; just call eval-mode forward. Gradient is
        # allowed because the aux CE backprops through it (params are still
        # frozen by requires_grad=False).
        return self.ae.decode(z)

    def decode_to_logprobs(self, z: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.decode_to_logits(z), dim=-1)

    # ----- forward / energy ------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone(x, gamma, pad_mask=pad_mask)

    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        s = self.cfg.eqm
        time_cond = getattr(s, "time_conditioning", "off") != "off"
        gamma = (
            torch.full((x.shape[0],), float(s.sample_gamma), device=x.device, dtype=x.dtype)
            if time_cond
            else None
        )
        v = self.forward(x, gamma)
        return (x * v).sum(dim=(1, 2))

    # ----- UQ / scoring API ------------------------------------------------- #
    # The conservative-gradient parameterisation means the trained model
    # exposes a scalar energy g(x) = ⟨x, f(x)⟩ at every point in latent
    # space. These methods surface energy / its gradient norm / a Hutchinson
    # trace estimate of its Hessian for post-training uncertainty
    # quantification on generated or recovered samples.
    #   * score_energy(x): how "in-distribution" is x (lower = more likely)
    #   * score_gradient_norm(x): distance from the nearest local minimum
    #     (larger = sampler hasn't fully settled)
    #   * score_curvature(x): basin sharpness (larger positive = sharper
    #     local minimum = higher confidence)
    # All three respect pad_mask so padded positions don't contribute.

    def _score_gamma(self, x: torch.Tensor) -> torch.Tensor | None:
        s = self.cfg.eqm
        if getattr(s, "time_conditioning", "off") == "off":
            return None
        return torch.full(
            (x.shape[0],), float(s.sample_gamma), device=x.device, dtype=x.dtype
        )

    @torch.no_grad()
    def score_energy(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sample scalar energy g(x) = ⟨x, f(x)⟩. Returns (B,)."""
        gamma = self._score_gamma(x)
        v = self.forward(x, gamma, pad_mask=pad_mask)
        if pad_mask is None:
            return (x * v).sum(dim=(1, 2))
        valid_f = (~pad_mask).to(x.dtype).unsqueeze(-1)
        return (x * v * valid_f).sum(dim=(1, 2))

    def score_gradient_norm(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sample L2 norm of ∇_x g(x). Larger = further from a local
        minimum of the energy. Returns (B,)."""
        gamma = self._score_gamma(x)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v = self.forward(x_req, gamma, pad_mask=pad_mask)
            if pad_mask is None:
                energy_total = (x_req * v).sum()
            else:
                valid_f = (~pad_mask).to(x_req.dtype).unsqueeze(-1)
                energy_total = (x_req * v * valid_f).sum()
            grad = torch.autograd.grad(energy_total, x_req, create_graph=False)[0]
        if pad_mask is None:
            return grad.detach().flatten(start_dim=1).norm(dim=-1)
        valid_f = (~pad_mask).to(grad.dtype).unsqueeze(-1)
        return (grad.detach() * valid_f).flatten(start_dim=1).norm(dim=-1)

    def score_curvature(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        n_samples: int = 4,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """Hutchinson estimate of tr(∇²_x g(x)) with Rademacher probes.
        Larger positive = sharper local minimum = higher confidence.

        Memory-heavy: each probe sample needs the create_graph=True graph
        from the energy forward + a second backward through it. We chunk
        along the batch dimension and empty the CUDA cache between chunks
        to keep peak memory bounded; n_samples is the number of probe
        vectors averaged. Costs ``n_samples`` × (1 fwd + 2 bwd) per chunk.
        Returns (B,)."""
        gamma = self._score_gamma(x)
        B = x.shape[0]
        traces = torch.zeros(B, device=x.device, dtype=x.dtype)
        cs = max(1, int(chunk_size))
        for start in range(0, B, cs):
            end = min(B, start + cs)
            x_chunk = x[start:end].detach()
            pad_chunk = pad_mask[start:end] if pad_mask is not None else None
            gamma_chunk = gamma[start:end] if gamma is not None else None
            valid_f = (
                (~pad_chunk).to(x_chunk.dtype).unsqueeze(-1)
                if pad_chunk is not None
                else None
            )
            trace_chunk = torch.zeros(end - start, device=x.device, dtype=x.dtype)
            for _ in range(n_samples):
                # Rademacher probe v built in-place to avoid a temporary float tensor.
                v_probe = (
                    torch.empty_like(x_chunk).uniform_(0.0, 1.0).round_().mul_(2.0).sub_(1.0)
                )
                with torch.enable_grad():
                    x_req = x_chunk.requires_grad_(True)
                    v = self.forward(x_req, gamma_chunk, pad_mask=pad_chunk)
                    if valid_f is None:
                        energy_total = (x_req * v).sum()
                    else:
                        energy_total = (x_req * v * valid_f).sum()
                    grad = torch.autograd.grad(energy_total, x_req, create_graph=True)[0]
                    inner = (grad * v_probe).sum()
                    Hv = torch.autograd.grad(inner, x_req, create_graph=False)[0]
                if valid_f is None:
                    est = (Hv * v_probe).sum(dim=(1, 2)).detach()
                else:
                    est = (Hv * v_probe * valid_f).sum(dim=(1, 2)).detach()
                trace_chunk = trace_chunk + est
                del v, grad, Hv, inner, energy_total, v_probe
            traces[start:end] = trace_chunk / float(n_samples)
            # Release per-chunk graph memory before the next chunk allocates.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return traces

    # ----- training step --------------------------------------------------- #

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        return self._eqm_loss(batch["token_ids"], pad_mask=batch.get("pad_mask"))

    def eval_step(self, batch: Any) -> LossDict:
        return self._eqm_loss(batch["token_ids"], pad_mask=batch.get("pad_mask"))

    @torch.enable_grad()
    def _eqm_loss(
        self,
        token_ids: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> LossDict:
        s = self.cfg.eqm
        device = token_ids.device

        # x1 from the frozen contextual AE encoder. For VAE mode we use
        # encode_sample so x1 is drawn fresh from the posterior at each
        # FM step — the latent-space analogue of Dirichlet thickening,
        # which keeps the FM target lively at γ→1 instead of trivially
        # zero. encode_sample is the identity for plain AE.
        with torch.no_grad():
            x1 = self.ae.encode_sample(token_ids, pad_mask=pad_mask)
        B, L, d = x1.shape
        dt = x1.dtype

        x0 = s.source_sigma * torch.randn(B, L, d, device=device, dtype=dt)

        gamma = torch.rand(B, device=device, dtype=dt).pow(s.gamma_power)
        x_gamma = (1.0 - gamma[:, None, None]) * x0 + gamma[:, None, None] * x1
        x_gamma.requires_grad_(True)

        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        v = self.forward(x_gamma, gamma, pad_mask=pad_mask)
        # Energy only over valid positions; padding contributes zero so
        # the gradient is automatically zero there.
        if pad_mask is None:
            energy = (x_gamma * v).sum()
        else:
            valid_f = (~pad_mask).to(x_gamma.dtype).unsqueeze(-1)
            energy = (x_gamma * v * valid_f).sum()
        grad_g = torch.autograd.grad(
            outputs=energy, inputs=x_gamma, create_graph=True, retain_graph=True
        )[0]

        if pad_mask is None:
            flow_loss = self.loss_fn(grad_g, u_tgt)
        else:
            # Masked MSE: mean over valid (token × d_latent) entries.
            valid_f = (~pad_mask).to(grad_g.dtype).unsqueeze(-1)
            sq = (grad_g - u_tgt).pow(2) * valid_f
            denom = (valid_f.sum() * grad_g.shape[-1]).clamp(min=1.0)
            flow_loss = sq.sum() / denom
        total = flow_loss
        out: LossDict = {"flow_loss": flow_loss}

        # CE and bigram-joint NLL both consume the implied-x1 reconstruction,
        # so compute it once if either is active.
        wants_anchor = s.lambda_ce > 0.0 or (
            self.bigram_head is not None
            and getattr(s, "lambda_bigram_joint", 0.0) > 0.0
        )
        if wants_anchor:
            ce_mask = gamma >= s.ce_min_gamma
            if ce_mask.any():
                lam = s.gradient_lambda
                pred_x1 = x_gamma[ce_mask] - lam * grad_g[ce_mask]
                ce_pad = pad_mask[ce_mask] if pad_mask is not None else None
                ids = token_ids[ce_mask].long()

                if s.lambda_ce > 0.0:
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
                    total = total + s.lambda_ce * ce
                    out["ce"] = ce.detach()

                # Non-factorised bigram joint NLL — Linear(2·d_latent → K²) on
                # adjacent implied-x1 vectors. Supervision flows back into
                # grad_g (via pred_x1), so the conservative gradient field
                # gets pushed toward digraph-aware structure.
                lam_bj = getattr(s, "lambda_bigram_joint", 0.0)
                if (
                    self.bigram_head is not None
                    and lam_bj > 0.0
                    and pred_x1.shape[1] >= 2
                ):
                    bg_logits = self.bigram_head(pred_x1)  # (M, L-1, K²)
                    bg_log_p = F.log_softmax(bg_logits, dim=-1)
                    bg_target = ids[:, :-1] * self.K + ids[:, 1:]  # (M, L-1)
                    if ce_pad is None:
                        bg_nll = F.nll_loss(
                            bg_log_p.reshape(-1, self.K * self.K),
                            bg_target.reshape(-1),
                        )
                    else:
                        # A bigram position is valid only if BOTH its tokens
                        # are non-padded.
                        valid_pair = (~ce_pad[:, :-1] & ~ce_pad[:, 1:]).reshape(-1).to(
                            bg_log_p.dtype
                        )
                        bg_per = F.nll_loss(
                            bg_log_p.reshape(-1, self.K * self.K),
                            bg_target.reshape(-1),
                            reduction="none",
                        )
                        bg_nll = (bg_per * valid_pair).sum() / valid_pair.sum().clamp(
                            min=1.0
                        )
                    total = total + lam_bj * bg_nll
                    out["bg_joint_nll"] = bg_nll.detach()

        out[TRAINING_LOSS_KEY] = total

        # γ-bucket diagnostics — masked when pad_mask present, otherwise
        # plain MSE. Reuse the same masking helper as flow_loss for
        # bucket-level losses.
        def _bucket_loss(mask_b: torch.Tensor) -> torch.Tensor:
            if pad_mask is None:
                return self.loss_fn(grad_g[mask_b], u_tgt[mask_b])
            v_f = (~pad_mask[mask_b]).to(grad_g.dtype).unsqueeze(-1)
            sq_b = (grad_g[mask_b] - u_tgt[mask_b]).pow(2) * v_f
            return sq_b.sum() / (v_f.sum() * grad_g.shape[-1]).clamp(min=1.0)

        # Per-sample L2 norms over (L, d_latent) for the echo-trap diagnostic.
        # Detached so they don't pollute the training graph.
        if pad_mask is None:
            grad_norms = grad_g.detach().flatten(start_dim=1).norm(dim=-1)
            tgt_norms = u_tgt.detach().flatten(start_dim=1).norm(dim=-1)
        else:
            valid_f = (~pad_mask).to(grad_g.dtype).unsqueeze(-1)
            grad_norms = (
                (grad_g.detach() * valid_f).flatten(start_dim=1).norm(dim=-1)
            )
            tgt_norms = (
                (u_tgt.detach() * valid_f).flatten(start_dim=1).norm(dim=-1)
            )

        for key, mask in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if mask.any():
                out[key] = _bucket_loss(mask)
                # Echo-trap diagnostic: ratio of mean ‖grad_g‖ to mean ‖u_tgt‖
                # in this γ-bin. ≈1 ⇒ real transport, →0 ⇒ flat field (echo
                # trap), ≫1 ⇒ overshoot.
                rkey = key.replace("g<", "r<")
                out[rkey] = grad_norms[mask].mean() / tgt_norms[mask].mean().clamp(
                    min=1e-8
                )

        return out

    def _c_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        s = self.cfg.eqm
        strategy = s.decay_strategy.strip().lower()
        one_minus = 1.0 - gamma
        if strategy == "linear":
            c = one_minus
        elif strategy == "truncated":
            a = torch.as_tensor(s.decay_a, device=gamma.device, dtype=gamma.dtype)
            denom = torch.clamp(1.0 - a, min=1e-7)
            c = torch.where(gamma <= a, torch.ones_like(gamma), one_minus / denom)
        elif strategy == "piecewise":
            a = torch.as_tensor(s.decay_a, device=gamma.device, dtype=gamma.dtype)
            b = torch.as_tensor(s.decay_b, device=gamma.device, dtype=gamma.dtype)
            left = b - ((b - 1.0) / a) * gamma
            right = one_minus / (1.0 - a)
            c = torch.where(gamma <= a, left, right)
        else:
            raise ValueError(f"unknown decay_strategy={s.decay_strategy!r}")
        c = c / torch.as_tensor(s.gradient_lambda, device=gamma.device, dtype=gamma.dtype)
        return c[:, None, None]

    # ----- sampling --------------------------------------------------------- #

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
        """NAG-GD on the conservative gradient in latent space."""
        s = self.cfg.eqm
        chosen = method if method is not None else getattr(s, "sampler", "nag")
        if chosen == "euler":
            nfe = max_steps if max_steps is not None else s.euler_nfe
            return self._sample_euler(B, L, nfe=nfe, x_init=x_init)
        if chosen == "sde":
            nfe = max_steps if max_steps is not None else s.euler_nfe
            return self._sample_sde(B, L, nfe=nfe, x_init=x_init)

        eta = eta if eta is not None else s.sample_eta
        mu = mu if mu is not None else s.sample_mu
        g_min = g_min if g_min is not None else s.sample_g_min
        max_steps = max_steps if max_steps is not None else s.sample_max_steps
        if grad_clip is None:
            grad_clip = s.sample_grad_clip
        if return_best is None:
            return_best = s.sample_return_best

        device = next(self.backbone.parameters()).device

        def _clip(g: torch.Tensor) -> torch.Tensor:
            if grad_clip is None:
                return g
            n = g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return g * (n.clamp(max=grad_clip) / n)

        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = s.source_sigma * torch.randn(B, L, self.d_latent, device=device)

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
            gamma = torch.full((B,), float(s.sample_gamma), device=x.device, dtype=x.dtype)
        else:
            gamma = None
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            energy = (x_req * self.forward(x_req, gamma)).sum()
            grad = torch.autograd.grad(energy, x_req, create_graph=False)[0]
        return grad.detach()

    def _sample_sde(
        self, B: int, L: int, *, nfe: int, x_init: torch.Tensor | None
    ) -> torch.Tensor:
        """SDE / Langevin sampler in latent space.

        Walks γ from 0 → 1 in ``nfe`` Euler steps on the conservative
        gradient ``∇⟨x, f(x; γ)⟩``, with Langevin noise ``sqrt(2·α·h)·ξ``
        injected per step (α = cfg.eqm.sde_alpha). The added stochasticity
        lets the chain cross between basins instead of getting stuck in
        the first one — the standard fix for the "unconditional sampling
        finds spurious basins" failure mode.

        No V_d projection (latent space has no zero-mean constraint).
        """
        from aitchinson_flow.sampling.sde import sde_flow_sample

        s = self.cfg.eqm
        device = next(self.backbone.parameters()).device
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, self.d_latent, device=device)

        time_cond = getattr(s, "time_conditioning", "off") != "off"
        return sde_flow_sample(
            self,
            x,
            n_steps=nfe,
            use_grad=True,  # always conservative gradient (the FM-trained field)
            alpha=float(getattr(s, "sde_alpha", 0.0)),
            time_conditioned=time_cond,
            project_zero_mean=False,  # latent space; no V_d
            grad_clip=getattr(s, "sample_grad_clip", None),
        )

    def _sample_euler(self, B: int, L: int, *, nfe: int, x_init: torch.Tensor | None) -> torch.Tensor:
        s = self.cfg.eqm
        device = next(self.backbone.parameters()).device
        sigma = s.sample_sigma_init if s.sample_sigma_init is not None else s.source_sigma
        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, self.d_latent, device=device)
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


@register("EqMAE")
def build_eqm_ae(cfg: Config) -> EquilibriumFlowMatchingAE:
    return EquilibriumFlowMatchingAE(cfg)
