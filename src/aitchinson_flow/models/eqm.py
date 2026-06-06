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


class BigramHead(nn.Module):
    """Non-factorised bigram joint head — emits K² logits per adjacent pair.

    Unlike the factorised ``log p(a) + log p(b)`` term controlled by
    ``cfg.eqm.lambda_bigram`` (which Phase 5 showed is just re-weighted
    unigram CE — see RESULTS.md §"Phase 5"), this head can in principle
    represent any bigram joint distribution because the K×K logit table is
    not a factorisation of two K-way marginals.

    Memory: K²=729, L−1=39 ⇒ (B, L−1, K²) ≈ 7 MB for B=64. Cheap.

    The head consumes adjacent encoder hidden states ``h[t] ⊕ h[t+1]`` so
    each pair gets its own context-dependent joint, not a global bigram
    table.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.transformer.d_model
        self.K = cfg.text8_dataset.K
        self.proj = nn.Linear(2 * d, self.K * self.K)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d) → flat-K² logits (B, L-1, K*K)."""
        h_pairs = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)  # (B, L-1, 2d)
        return self.proj(h_pairs)


class EquilibriumFlowMatching(nn.Module):
    def __init__(self, cfg: Config, loss_fn: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.backbone = TransformerBackbone(cfg=cfg)
        self.velocity_head = VelocityHead(cfg=cfg)
        # Optional non-factorised bigram joint head (W2). Only instantiated
        # when lambda_bigram_joint > 0 so EqM training without the term
        # produces a checkpoint with the same parameter count as before
        # (state_dict-compatible with existing runs).
        if getattr(cfg.eqm, "lambda_bigram_joint", 0.0) > 0.0:
            self.bigram_head: BigramHead | None = BigramHead(cfg=cfg)
        else:
            self.bigram_head = None

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.velocity_head(self.backbone(x, gamma, h_ctx))

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        h_clean = batch.get("h_clean")
        h_invalid = batch.get("h_invalid")
        out = self._eqm_loss(
            batch["x"], token_ids=batch.get("token_ids"), h_ctx=h_clean
        )
        if (
            getattr(self.cfg.eqm, "lambda_E_hinge", 0.0) > 0.0
            and "x_invalid" in batch
        ):
            aux = self._auditor_hinge(
                batch["x"], batch["x_invalid"],
                h_clean=h_clean, h_invalid=h_invalid,
            )
            out[TRAINING_LOSS_KEY] = out[TRAINING_LOSS_KEY] + (
                self.cfg.eqm.lambda_E_hinge * aux["hinge_loss"]
            )
            for k, v in aux.items():
                if k != "hinge_loss":
                    out[k] = v
            out["hinge_loss"] = aux["hinge_loss"].detach()
        return out

    def eval_step(self, batch: Any) -> LossDict:
        # Eval mirrors training_step structure but skips the FM regression's
        # second-order autograd path when no aux losses need it. We still
        # need autograd for grad-norm computation, so don't wrap in no_grad.
        h_clean = batch.get("h_clean")
        h_invalid = batch.get("h_invalid")
        out = self._eqm_loss(
            batch["x"], token_ids=batch.get("token_ids"), h_ctx=h_clean
        )
        if (
            getattr(self.cfg.eqm, "lambda_E_hinge", 0.0) > 0.0
            and "x_invalid" in batch
        ):
            aux = self._auditor_hinge(
                batch["x"], batch["x_invalid"],
                h_clean=h_clean, h_invalid=h_invalid,
            )
            out["hinge_loss"] = aux["hinge_loss"].detach()
            out["E_grad_clean"] = aux["E_grad_clean"]
            out["E_grad_invalid"] = aux["E_grad_invalid"]
        if self.cfg.training.eval_bpd:
            with torch.no_grad():
                out["bpd"] = self.bpd(
                    batch["token_ids"], max_steps=self.cfg.training.eval_bpd_max_steps
                )
        return out

    @torch.enable_grad()
    def _auditor_hinge(
        self,
        x_clean: torch.Tensor,
        x_invalid: torch.Tensor,
        *,
        h_clean: torch.Tensor | None = None,
        h_invalid: torch.Tensor | None = None,
    ) -> LossDict:
        """Compute the binary-discriminator hinge loss on grad-norm².

        Per-sequence grad-norm² is ``Σ_{l,k} (∇⟨x,f(x;γ_aud,h)⟩)²`` evaluated
        at ``γ = cfg.eqm.auditor_gamma`` (default 1.0, the data-manifold
        endpoint). The hinge biases the trained field so that this
        quantity is small on clean and at least ``margin²`` on invalid.

        The implementation re-uses ``_grad_norm_sq`` for both branches; the
        compute graph is preserved so the loss back-propagates through the
        velocity head and backbone.
        """
        s = self.cfg.eqm
        margin = float(s.margin_energy)
        e_clean = self._grad_norm_sq(
            x_clean, gamma_value=s.auditor_gamma, h_ctx=h_clean
        )
        e_invalid = self._grad_norm_sq(
            x_invalid, gamma_value=s.auditor_gamma, h_ctx=h_invalid
        )
        hinge_clean = e_clean.mean()
        hinge_invalid = torch.relu(margin * margin - e_invalid).mean()
        hinge_total = hinge_clean + hinge_invalid
        return {
            "hinge_loss": hinge_total,
            "E_grad_clean": e_clean.mean().detach(),
            "E_grad_invalid": e_invalid.mean().detach(),
        }

    def _grad_norm_sq(
        self,
        x: torch.Tensor,
        *,
        gamma_value: float,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-sequence ``Σ_{l,k} (∇_x ⟨x, f(x;γ,h)⟩)²`` with create_graph=True
        so the discriminator hinge can back-prop through the gradient.

        Returns (B,) tensor.
        """
        time_cond = getattr(self.cfg.eqm, "time_conditioning", "off")
        B = x.shape[0]
        x_req = x.detach().requires_grad_(True)
        if time_cond != "off":
            gamma = torch.full(
                (B,), float(gamma_value), device=x.device, dtype=x.dtype
            )
        else:
            gamma = None
        v = self.forward(x_req, gamma, h_ctx)
        energy = (x_req * v).sum()
        grad = torch.autograd.grad(
            energy, x_req, create_graph=self.training, retain_graph=True
        )[0]
        return grad.pow(2).sum(dim=(-1, -2))

    @torch.enable_grad()
    def _eqm_loss(
        self,
        x1: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
        h_ctx: torch.Tensor | None = None,
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

        # 3. Target gradient (data-to-noise direction).
        u_tgt = self._c_gamma(gamma) * (x0 - x1)

        # 3b. Optional stochastic-interpolant noise (Albergo et al. 2023). Adds
        # σ(γ)·z to x_γ and σ'(γ)·z to u_tgt. σ(γ)=σ_max·sin(πγ) vanishes at
        # the endpoints so ρ_0, ρ_1 are preserved; peaks at γ=0.5. Provides a
        # per-(x_γ,γ) variance floor in the path interior — an alternative to
        # Dirichlet thickening of x_1.
        sig_max = self.cfg.transformation.sigma_interpolant_max
        if sig_max > 0.0:
            z = torch.randn(B, L, D, device=device, dtype=dt)
            z = z - z.mean(dim=-1, keepdim=True)  # project to V_d
            sched = self.cfg.transformation.sigma_interpolant_schedule
            if sched == "sin":
                sigma_g = sig_max * torch.sin(torch.pi * gamma)
                sigma_prime_g = sig_max * torch.pi * torch.cos(torch.pi * gamma)
            else:
                raise ValueError(
                    f"Unknown transformation.sigma_interpolant_schedule={sched!r}"
                )
            x_gamma = x_gamma + sigma_g[:, None, None] * z
            u_tgt = u_tgt + sigma_prime_g[:, None, None] * z

        x_gamma.requires_grad_(True)

        # 4. Forward pass — pass γ if backbone is time-conditioned, and h_ctx
        # if context-conditioned. When the bigram joint head is active we run
        # backbone+velocity_head separately so we can also feed the encoder
        # hiddens into BigramHead in a single encoder pass.
        if self.bigram_head is not None:
            h_enc = self.backbone(x_gamma, gamma, h_ctx)  # (B, L, d_model)
            v = self.velocity_head(h_enc)
        else:
            h_enc = None
            v = self.forward(x_gamma, gamma, h_ctx)

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
        wants_anchor = (
            s.lambda_ce > 0.0
            or s.lambda_bigram > 0.0
            or s.lambda_trigram > 0.0
            or getattr(s, "lambda_bigram_joint", 0.0) > 0.0
        )
        if wants_anchor and token_ids is not None:
            ce_mask = gamma >= s.ce_min_gamma
            if ce_mask.any():
                lam = s.gradient_lambda
                pred_x1 = x_gamma[ce_mask] - lam * grad_g[ce_mask]
                log_probs = pred_x1 - torch.logsumexp(pred_x1, dim=-1, keepdim=True)
                ids = token_ids[ce_mask].long()  # (M, L)
                if s.lambda_ce > 0.0:
                    aux_kind = getattr(s, "aux_kind", "softmax")
                    if aux_kind == "sparsemax":
                        from aitchinson_flow.losses import sparsemax_loss
                        ce = sparsemax_loss(pred_x1.reshape(-1, D), ids.reshape(-1))
                    else:
                        ce = F.nll_loss(log_probs.reshape(-1, D), ids.reshape(-1))
                    total_loss = total_loss + s.lambda_ce * ce
                    out["ce"] = ce.detach()
                # 7b. Non-factorised bigram NLL (W2). Distinct from the
                # factorised lambda_bigram path above — uses the BigramHead
                # to emit K² joint logits per adjacent position pair from
                # the encoder hiddens (not the velocity output). The Phase
                # 5 negative result documented that the factorised version
                # is a no-op architecturally.
                if (
                    self.bigram_head is not None
                    and getattr(s, "lambda_bigram_joint", 0.0) > 0.0
                    and h_enc is not None
                    and ids.shape[1] >= 2
                ):
                    bg_logits = self.bigram_head(h_enc[ce_mask])  # (M, L-1, K²)
                    bg_log_p = bg_logits.log_softmax(dim=-1)
                    bg_target = ids[:, :-1] * D + ids[:, 1:]
                    bg_joint_nll = F.nll_loss(
                        bg_log_p.reshape(-1, D * D), bg_target.reshape(-1)
                    )
                    total_loss = total_loss + s.lambda_bigram_joint * bg_joint_nll
                    out["bg_joint_nll"] = bg_joint_nll.detach()
                # Bigram log-prob on adjacent positions: log p(a)+log p(b) for the
                # observed digraph. Memory: (M, L-1, D, D) at D=27 = ~3 MB at typical M.
                if s.lambda_bigram > 0.0 and log_probs.shape[1] >= 2:
                    log_p_bi = log_probs[:, :-1, :, None] + log_probs[:, 1:, None, :]
                    bg_target = ids[:, :-1] * D + ids[:, 1:]
                    bg_nll = F.nll_loss(
                        log_p_bi.reshape(-1, D * D), bg_target.reshape(-1)
                    )
                    total_loss = total_loss + s.lambda_bigram * bg_nll
                    out["bg_nll"] = bg_nll.detach()
                # Trigram analogously: (M, L-2, D, D, D) — D³ = 19683. Reshape to
                # (M, L-2, D³) → ~30 MB at typical M; still cheap for L=40, B=64.
                if s.lambda_trigram > 0.0 and log_probs.shape[1] >= 3:
                    log_p_tri = (
                        log_probs[:, :-2, :, None, None]
                        + log_probs[:, 1:-1, None, :, None]
                        + log_probs[:, 2:, None, None, :]
                    )
                    tg_target = ids[:, :-2] * D * D + ids[:, 1:-1] * D + ids[:, 2:]
                    tg_nll = F.nll_loss(
                        log_p_tri.reshape(-1, D * D * D), tg_target.reshape(-1)
                    )
                    total_loss = total_loss + s.lambda_trigram * tg_nll
                    out["tg_nll"] = tg_nll.detach()

        out[TRAINING_LOSS_KEY] = total_loss

        # 8. γ-bucket diagnostics.
        # Per-sample L2 norms over (L, D) — one scalar per batch element.
        grad_norms = grad_g.detach().flatten(start_dim=1).norm(dim=-1)
        tgt_norms = u_tgt.detach().flatten(start_dim=1).norm(dim=-1)
        for key, mask in (
            ("g<.33", gamma < 0.33),
            ("g<.66", (gamma >= 0.33) & (gamma < 0.66)),
            ("g<1", gamma >= 0.66),
        ):
            if mask.any():
                out[key] = self.loss_fn(grad_g[mask], u_tgt[mask])
                # Echo-trap diagnostic: ratio of mean ‖grad_g‖ to mean ‖u_tgt‖
                # in this γ-bin. ≈1 ⇒ field has the right magnitude (real
                # transport). →0 ⇒ field is flat in this bin (echo trap).
                # ≫1 ⇒ overshooting. Detached — diagnostic only.
                rkey = key.replace("g<", "r<")
                out[rkey] = grad_norms[mask].mean() / tgt_norms[mask].mean().clamp(min=1e-8)

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
        method: str | None = None,
        alpha: float | None = None,
        use_grad: bool | None = None,
    ) -> torch.Tensor:
        """Dispatch to NAG-GD or Euler-γ sampler per cfg.eqm.sampler.

        When ``cfg.eqm.sampler == "euler"`` (W1, RESULTS.md §1) ``max_steps`` is
        treated as the number of Euler steps (NFE); the other NAG-specific
        kwargs (eta, mu, g_min, grad_clip, return_best) are ignored. The
        runner's ``_unigram_kl_probe`` always passes ``max_steps=`` so
        switching the sampler only changes inference behaviour.

        Otherwise (default ``"nag"``) iterates
            x ← x − η·∇E(x + μ(x − x_prev))
        until max_batch ||∇E|| < g_min or max_steps is reached.

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
        # Method dispatch: explicit kwarg > cfg.eqm.sampler.
        chosen = method if method is not None else getattr(s, "sampler", "nag")
        if chosen == "sde":
            nfe = max_steps if max_steps is not None else s.euler_nfe
            return self._sample_sde(
                B,
                L,
                nfe=nfe,
                alpha=float(alpha) if alpha is not None else 0.0,
                use_grad=use_grad if use_grad is not None else s.euler_use_grad,
                sigma_init=s.sample_sigma_init,
                x_init=x_init,
            )
        if chosen == "euler":
            nfe = max_steps if max_steps is not None else s.euler_nfe
            return self.sample_euler(
                B,
                L,
                nfe=nfe,
                use_grad=use_grad if use_grad is not None else s.euler_use_grad,
                sigma_init=s.sample_sigma_init,
                x_init=x_init,
            )
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
        """Conservative gradient ∇_x ⟨x, f(x)⟩ — matches the training target.

        When the backbone is time-conditioned, we evaluate at the configured
        ``cfg.eqm.sample_gamma`` (default 0.5). γ=1 is degenerate: c(γ=1)=0
        zeros the FM target and the model learns f(·, γ=1) ≈ 0, giving a
        flat energy field at sample time.
        """
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
        self,
        B: int,
        L: int,
        *,
        nfe: int,
        alpha: float,
        use_grad: bool,
        sigma_init: float | None,
        x_init: torch.Tensor | None,
    ) -> torch.Tensor:
        """SDE sampler entry point — Phase R (CAPSTONE_EXPERIMENTS.md §4)."""
        from aitchinson_flow.sampling.sde import sde_flow_sample

        s = self.cfg.eqm
        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K
        sigma = sigma_init if sigma_init is not None else s.source_sigma

        if x_init is not None:
            x0 = x_init.to(device).detach()
        else:
            x0 = sigma * torch.randn(B, L, K, device=device)
            x0 = x0 - x0.mean(dim=-1, keepdim=True)

        time_cond = getattr(s, "time_conditioning", "off") != "off"
        return sde_flow_sample(
            self,
            x0,
            n_steps=nfe,
            use_grad=use_grad,
            alpha=alpha,
            time_conditioned=time_cond,
            project_zero_mean=True,
            grad_clip=s.sample_grad_clip,
        )

    @torch.no_grad()
    def sample_euler(
        self,
        B: int,
        L: int,
        *,
        nfe: int | None = None,
        use_grad: bool | None = None,
        sigma_init: float | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FM-style Euler integrator over γ on the trained velocity field.

        Walks γ from 0 → 1 in ``nfe`` steps; at each step queries the model at
        the current γ and applies ``x ← x − h·v``. Sign matches the FM target
        ``c(γ)·(x0 − x1)`` (eqm.py:60) which points data→noise, so the data
        direction is the negative of the velocity.

        ``use_grad=False`` (default) uses the raw velocity ``f(x;γ)``; this is
        the path RESULTS.md §1 recommends (the highest-leverage W1 change).
        ``use_grad=True`` runs the same integrator on the conservative
        gradient ``∇⟨x,f⟩`` instead — useful as an ablation that isolates
        "raw f vs grad-of-energy" from "Euler vs NAG".
        """
        s = self.cfg.eqm
        if nfe is None:
            nfe = s.euler_nfe
        if use_grad is None:
            use_grad = s.euler_use_grad
        sigma = sigma_init if sigma_init is not None else s.source_sigma

        device = next(self.parameters()).device
        K = self.cfg.text8_dataset.K

        if x_init is not None:
            x = x_init.to(device).detach()
        else:
            x = sigma * torch.randn(B, L, K, device=device)
            x = x - x.mean(dim=-1, keepdim=True)

        time_cond = getattr(s, "time_conditioning", "off")
        gammas = torch.linspace(0.0, 1.0, nfe + 1, device=device)[:-1]
        h = 1.0 / nfe

        for g in gammas:
            g_b = g.expand(B) if time_cond != "off" else None
            if use_grad:
                # Conservative-gradient path: re-enable autograd locally.
                with torch.enable_grad():
                    x_req = x.detach().requires_grad_(True)
                    energy = (x_req * self.forward(x_req, g_b)).sum()
                    v = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
            else:
                v = self.forward(x, g_b)
            x = x - h * v
            x = x - x.mean(dim=-1, keepdim=True)

        return x

    @torch.no_grad()
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Sequence-level energy ``E(x) = ⟨x, f(x)⟩`` summed over (L, K).

        Returns a (B,) scalar per sequence — the natural EBM readout that the
        OOD harness (W3, scripts/eval_ood.py) compares between clean and
        corrupted inputs. Higher |E| means the field is pushing harder on
        this sequence; the *sign and magnitude* together carry the OOD
        signal (energy is not constrained to be non-negative under the FM
        regression objective).

        Time-conditioned models are queried at ``cfg.eqm.sample_gamma`` to
        stay in the signal regime — γ=1 is degenerate (c(1)=0 zeros the
        velocity, so f(·,γ=1)≈0 and the energy is ≈0 for everything).
        """
        s = self.cfg.eqm
        time_cond = getattr(s, "time_conditioning", "off")
        if time_cond != "off":
            B = x.shape[0]
            gamma = torch.full((B,), float(s.sample_gamma), device=x.device, dtype=x.dtype)
        else:
            gamma = None
        v = self.forward(x, gamma)
        return (x * v).sum(dim=(1, 2))  # (B,)

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
