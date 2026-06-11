"""DirichletFMSvgp — Dirichlet Flow Matching (Stark et al. 2024) +
post-hoc SVGP head over pooled hidden states for OOD detection.

Architecture
------------
                        ┌─────────────────────────┐
                        │ DFMBackbone (existing)  │
                        │   x_t, t → h (B, L, d)  │
                        └─────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
        ┌────────────────┐                ┌────────────────────┐
        │ DFMHead        │                │ SequencePooler     │
        │   h → K logits │                │   h → z (B, e)     │
        │   → DFM CE loss│                └────────────────────┘
        └────────────────┘                         │
                                                   ▼
                                          ┌────────────────────┐
                                          │ _SVGPHead          │
                                          │  q(f) ≈ GP         │
                                          │  → mean + variance │
                                          │  → epistemic UQ    │
                                          └────────────────────┘

Stage 1 (FM training): standard DFM training step (CE on the denoised
posterior). SVGP parameters are frozen; only DFMBackbone + DFMHead train.
Flash attention is enabled by default — DFM has no second-order autograd
requirement.

Stage 2 (SVGP fit): after FM training is done, call ``fit_svgp(pos_loader,
n_neg, ...)``: pool encoder hidden states on positive (real) batches,
generate negatives via configurable strategies, fit the SVGP with
``_SVGPHead.fit``.

Inference (OOD score): ``ood_score(token_ids, t_eval)`` returns the per-batch
SVGP predictive variance — large = epistemically uncertain = likely OOD.
The SVGP latent variance is used directly (not Bernoulli-likelihood mean),
which preserves the OOD signal at saturation.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register
from aitchinson_flow.models.dirichlet_fm import DirichletFlowMatching
from aitchinson_flow.models.dirichlet_fm_auditor import _SVGPHead


class SequencePooler(nn.Module):
    """Pools (B, L, d) hidden states to a per-sample embedding (B, d_embed).

    Modes:
      * "mean"      — length-averaged hidden state, then linear projection
      * "max"       — coord-wise max over positions, then linear projection
      * "attention" — single learned query attends over positions, then
                      linear projection of the attention output

    All modes end with a ``Linear(d_model → d_embed)`` so the SVGP input
    dimensionality is decoupled from ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        d_embed: int = 256,
        mode: str = "mean",
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        if mode not in ("mean", "max", "attention"):
            raise ValueError(f"unknown pooler mode: {mode!r}")
        self.mode = mode
        self.d_model = d_model
        self.d_embed = d_embed
        self.proj = nn.Linear(d_model, d_embed)
        if mode == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(
                d_model, num_heads=attn_heads, batch_first=True
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            pooled = h.mean(dim=1)
        elif self.mode == "max":
            pooled = h.max(dim=1).values
        else:  # "attention"
            B = h.shape[0]
            q = self.query.expand(B, -1, -1)
            pooled, _ = self.attn(q, h, h, need_weights=False)
            pooled = pooled.squeeze(1)
        return self.proj(pooled)


class DirichletFMSvgp(nn.Module):
    """Dirichlet FM + post-hoc SVGP head for OOD detection.

    Stage 1 training (this class's ``training_step``) is identical to
    :class:`DirichletFlowMatching`: cross-entropy denoiser on x_1 given
    a Dirichlet-interpolated x_t. The pooler and SVGP parameters do not
    participate in this stage.

    Stage 2 fitting (``fit_svgp``) extracts pooled embeddings on a loader
    of real (positive) sequences, generates negatives via configurable
    strategies, and trains the SVGP via the existing
    :class:`_SVGPHead.fit` (Adam + VariationalELBO, Bernoulli likelihood,
    RBF kernel, CholeskyVariationalDistribution).

    Inference (``ood_score``) returns the SVGP latent predictive variance
    — the epistemic UQ signal — per batch element.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.K = cfg.text8_dataset.K
        # Reuse the standard DFM (Stark et al. 2024) backbone + head.
        self.dfm = DirichletFlowMatching(cfg=cfg)

        # Pooler + SVGP head sit on top of dfm.backbone's hidden output.
        dfs = cfg.dfm_svgp
        d_model = cfg.transformer.d_model
        self.pooler = SequencePooler(
            d_model=d_model, d_embed=dfs.d_embed, mode=dfs.pooling
        )
        self.svgp = _SVGPHead(
            d_model=dfs.d_embed,
            n_inducing=int(dfs.n_inducing),
            kernel=dfs.kernel,
        )
        # Trainable scalar energy head used by the joint contrastive hinge.
        # Avoids the SVGP kernel-collapse failure mode at init (when
        # lengthscale << ‖z‖, kernel is ≈ 0 everywhere → GP output is just
        # the constant mean → identical for valid and invalid → hinge can't
        # open the gap). The SVGP stays in the model for *post-hoc* OOD UQ
        # via fit_svgp(); during Stage 1 the energy head is the discriminative
        # signal, and the pooler/backbone learn features that separate
        # valid from invalid through it.
        self.energy_head = nn.Linear(dfs.d_embed, 1)
        # Joint contrastive-hinge training (lambda_hinge > 0 + train_svgp_jointly):
        # unfreeze pooler + SVGP gp + SVGP likelihood so the hinge gradient
        # propagates through the encoder. Falls back to the post-hoc fit
        # behaviour when train_svgp_jointly is False.
        pooler_trainable = (
            dfs.train_pooler_with_dfm
            or (dfs.lambda_hinge > 0.0 and dfs.train_svgp_jointly)
        )
        if not pooler_trainable:
            for p in self.pooler.parameters():
                p.requires_grad_(False)
        if dfs.lambda_hinge > 0.0 and dfs.train_svgp_jointly:
            for p in self.svgp.gp.parameters():
                p.requires_grad_(True)
            for p in self.svgp.likelihood.parameters():
                p.requires_grad_(True)

    # ----- Stage 1: FM training -----------------------------------------
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.dfm.forward(x_t, t)

    def training_step(self, batch: Any, step: int) -> LossDict:
        # Stage 1: standard DFM CE training step.
        out = self.dfm.training_step(batch, step)
        dfs = self.cfg.dfm_svgp

        # Optional contrastive-hinge term on the SVGP latent mean.
        # Mirrors BayesianAuditorAE's hinge (clean energy ≈ 0; invalid
        # energy ≥ margin), but here the "energy" is the GP posterior
        # mean of the SVGP head fed pooled hidden states.
        if (
            dfs.lambda_hinge > 0.0
            and "token_ids_invalid" in batch
            and isinstance(batch["token_ids_invalid"], torch.Tensor)
        ):
            total = out[TRAINING_LOSS_KEY]
            token_ids = batch["token_ids"].long()
            token_ids_inv = batch["token_ids_invalid"].long()
            device = token_ids.device
            B = token_ids.shape[0]
            t_h = dfs.hinge_t_eval if dfs.hinge_t_eval is not None else dfs.t_eval
            t = torch.full((B,), float(t_h), device=device, dtype=torch.float32)

            # Pool features at hinge time. We deliberately do NOT detach z —
            # gradients flow back through pooler + backbone.
            z_clean = self.pool_features(token_ids, t)
            z_invalid = self.pool_features(token_ids_inv, t)

            # Energy head — trainable Linear(d_embed → 1). Direct gradient,
            # no kernel-collapse failure mode. The SVGP is reserved for
            # post-hoc OOD UQ (fit_svgp after Stage 1).
            e_clean = self.energy_head(z_clean).squeeze(-1)
            e_invalid = self.energy_head(z_invalid).squeeze(-1)

            hinge_clean = e_clean.pow(2).mean()
            hinge_invalid = F.relu(dfs.margin_energy - e_invalid).mean()
            hinge = hinge_clean + hinge_invalid
            total = total + dfs.lambda_hinge * hinge

            out[TRAINING_LOSS_KEY] = total
            out["hinge_loss"] = hinge.detach()
            out["E_clean"] = e_clean.mean().detach()
            out["E_invalid"] = e_invalid.mean().detach()
        return out

    def eval_step(self, batch: Any) -> LossDict:
        return self.dfm.eval_step(batch)

    @torch.no_grad()
    def sample(self, B: int, L: int, **kwargs: Any) -> torch.Tensor:
        return self.dfm.sample(B, L, **kwargs)

    # ----- Hidden state extraction (Stage 1 → Stage 2 interface) --------
    def get_hidden_states(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """(B, L, K), (B,) → (B, L, d_model). Mirrors dfm.forward but stops
        before the DFMHead, so the pooler sees the same features the
        DFMHead does. Useful for both Stage 2 fit and inference-time UQ."""
        denom = max(self.dfm.t_max - 1.0, 1.0)
        t_norm = ((t - 1.0) / denom).to(dtype=x_t.dtype)
        return self.dfm.backbone(x_t, t_norm)

    def pool_features(
        self,
        token_ids: torch.Tensor,
        t: torch.Tensor,
        *,
        require_grad: bool = False,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Convenience: token_ids (B, L) → pooled features (B, d_embed) at
        a given Dirichlet path time ``t``. Used by :meth:`fit_svgp` to
        build the training set and by :meth:`ood_score` at inference.

        ``deterministic=True`` (default) feeds the *mean* of Dir(beta(t, tokens))
        — beta / sum(beta) — through the backbone instead of a stochastic
        ``Dirichlet(beta).sample()``. The sample injects per-position noise that
        washes out the token identity, collapsing clean vs corrupted features to
        the same noise (the cause of E_clean==E_invalid in the hinge). The mean
        keeps the backbone as a clean text->representation encoder, so corrupted
        tokens produce a genuinely different representation for OOD detection.

        ``require_grad=True`` forces autograd ON regardless of the
        ``train_pooler_with_dfm`` gate — used by :meth:`fit_svgp_hinge`
        when ``train_pooler=True`` so the pooler actually receives gradient
        (otherwise z would be detached and the pooler never updates)."""
        if deterministic:
            B, L = token_ids.shape
            beta = torch.ones(B, L, self.K, device=token_ids.device, dtype=t.dtype)
            beta.scatter_(-1, token_ids.long().unsqueeze(-1),
                          t[:, None, None].expand(B, L, 1).to(beta.dtype))
            x_t = beta / beta.sum(-1, keepdim=True)
        else:
            x_t = self.dfm._sample_xt(token_ids.long(), t)
        grad_on = require_grad or (
            self.training and self.cfg.dfm_svgp.train_pooler_with_dfm
        )
        with torch.set_grad_enabled(grad_on):
            h = self.get_hidden_states(x_t, t)
            z = self.pooler(h)
        return z

    # ----- Negative generators (Stage 2) --------------------------------
    @torch.no_grad()
    def _negatives_random_simplex(
        self, n: int, L: int, t: torch.Tensor
    ) -> torch.Tensor:
        """Strategy: random simplex points (uniform Dirichlet) at the same
        path time. Hidden state → pooled features."""
        device = next(self.parameters()).device
        K = self.K
        x_t = Dirichlet(torch.ones(K, device=device)).sample((n, L))
        t_b = torch.full((n,), float(t.mean().item()) if t.numel() else 4.5,
                         device=device, dtype=x_t.dtype)
        h = self.get_hidden_states(x_t, t_b)
        return self.pooler(h)

    @torch.no_grad()
    def _negatives_scrambled(
        self, token_ids: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Strategy: shuffle each row's tokens independently (preserves
        unigram distribution, destroys bigram structure)."""
        B, L = token_ids.shape
        device = token_ids.device
        perms = torch.argsort(torch.rand(B, L, device=device), dim=-1)
        scrambled = token_ids.gather(1, perms)
        return self.pool_features(scrambled, t)

    # ----- Stage 2: SVGP fit --------------------------------------------
    @torch.no_grad()
    def collect_features(
        self,
        loader: Iterable[dict],
        *,
        t_eval: float | None = None,
        max_pos: int = 4000,
        neg_strategy: str = "scrambled",
        neg_per_pos_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk ``loader`` until ``max_pos`` positive features are collected,
        generate matching negatives, return ``(features, labels)``.

        Labels: 1 = real (positive), 0 = synthetic (negative).
        """
        dfs = self.cfg.dfm_svgp
        if t_eval is None:
            t_eval = dfs.t_eval

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        pos_feats: list[torch.Tensor] = []
        neg_feats: list[torch.Tensor] = []
        n_pos = 0
        for batch in loader:
            tok = batch["token_ids"].to(device).long()
            B = tok.shape[0]
            t = torch.full((B,), float(t_eval), device=device, dtype=torch.float32)
            pos = self.pool_features(tok, t)
            pos_feats.append(pos.detach().cpu())
            # Negatives — same count as positives in this batch (ratio 1.0).
            n_neg = max(1, int(round(B * neg_per_pos_ratio)))
            if neg_strategy == "scrambled":
                neg = self._negatives_scrambled(tok[:n_neg], t[:n_neg])
            elif neg_strategy == "random_simplex":
                neg = self._negatives_random_simplex(n_neg, tok.shape[1], t)
            elif neg_strategy == "mixed":
                half = n_neg // 2
                neg1 = self._negatives_scrambled(tok[:half], t[:half])
                neg2 = self._negatives_random_simplex(n_neg - half, tok.shape[1], t)
                neg = torch.cat([neg1, neg2], dim=0)
            else:
                raise ValueError(f"unknown neg_strategy: {neg_strategy!r}")
            neg_feats.append(neg.detach().cpu())
            n_pos += B
            if n_pos >= max_pos:
                break

        if was_training:
            self.train()

        pos = torch.cat(pos_feats)[:max_pos]
        neg = torch.cat(neg_feats)[:int(max_pos * neg_per_pos_ratio)]
        feats = torch.cat([pos, neg], dim=0)
        labels = torch.cat([
            torch.ones(pos.shape[0]),
            torch.zeros(neg.shape[0]),
        ])
        return feats, labels

    # ----- Stage 2 (hinge variant): train pooler + SVGP via contrastive
    #       energy hinge with the DFM backbone frozen. ---------------------
    def fit_svgp_hinge(
        self,
        loader: Iterable[dict],
        *,
        t_eval: float | None = None,
        n_epochs: int = 5,
        max_steps: int | None = None,
        lr: float = 0.001,
        margin_energy: float | None = None,
        train_pooler: bool = True,
        verbose: bool = True,
        lengthscale_init: float | None = None,
    ) -> dict[str, Any]:
        """Sequential Stage 2: freeze DFM backbone + DFMHead, train pooler
        + SVGP via the contrastive hinge using batch['token_ids_invalid'].
        SVGP latent mean is the energy. Avoids the kernel-collapse failure
        mode by re-seeding inducing points + standardisation buffers from
        positive features and setting a sensible lengthscale before
        gradient training.
        """
        import gpytorch as _gp
        dfs = self.cfg.dfm_svgp
        if t_eval is None:
            t_eval = dfs.t_eval
        if margin_energy is None:
            margin_energy = dfs.margin_energy

        device = next(self.parameters()).device

        # 1) Freeze DFM (backbone + DFMHead). Unfreeze pooler + SVGP.
        for p in self.dfm.parameters():
            p.requires_grad_(False)
        if train_pooler:
            for p in self.pooler.parameters():
                p.requires_grad_(True)
        else:
            for p in self.pooler.parameters():
                p.requires_grad_(False)
        for p in self.svgp.gp.parameters():
            p.requires_grad_(True)
        # Likelihood — not used by hinge loss but unfreeze for completeness.
        for p in self.svgp.likelihood.parameters():
            p.requires_grad_(True)

        self.dfm.eval()  # keep BN/dropout off in backbone
        self.pooler.train()
        self.svgp.gp.train()
        self.svgp.likelihood.train()

        # 2) Re-seed _mu/_sigma + inducing points + lengthscale from a warmup
        # pass of pooled positive features. Avoids the kernel-collapse trap
        # (lengthscale << ||z|| at init → kernel ≈ 0 → all queries give the
        # constant mean → hinge can't open the gap).
        # Materialize the loader once: the warmup pass below and the training
        # loop further down each iterate it. A single-pass generator would be
        # exhausted by the warmup, leaving the training loop with 0 steps.
        batches = list(loader)

        warmup_feats: list[torch.Tensor] = []
        warmup_target = max(self.svgp.n_inducing * 4, 1024)
        with torch.no_grad():
            for batch in batches:
                tok = batch["token_ids"].to(device).long()
                t = torch.full((tok.shape[0],), float(t_eval), device=device)
                z = self.pool_features(tok, t)
                warmup_feats.append(z.detach().cpu())
                if sum(f.shape[0] for f in warmup_feats) >= warmup_target:
                    break
        feats = torch.cat(warmup_feats)[:warmup_target].to(device)
        mu = feats.mean(0)
        sigma = feats.std(0).clamp_min(1e-6)
        Xs = (feats - mu) / sigma
        # Set standardisation buffers so SVGP forward applies the same
        # transform at inference (see _SVGPHead.forward).
        self.svgp._mu = mu.detach()
        self.svgp._sigma = sigma.detach()
        # Re-seed inducing points from standardised features.
        from aitchinson_flow.models.dirichlet_fm_auditor import _build_svgp_module
        perm = torch.randperm(Xs.shape[0], device=device)[: self.svgp.n_inducing]
        ip = Xs[perm].detach().clone()
        self.svgp.gp = _build_svgp_module(ip, kernel=self.svgp.kernel).to(device)
        self.svgp.likelihood = _gp.likelihoods.BernoulliLikelihood().to(device)
        # Lengthscale init: default ℓ = √d_embed keeps r/ℓ ≈ 1 at random
        # pairs (works for OOD via the trained *mean*, but std saturates
        # because all queries are at similar normalised distance).
        # Pass ``lengthscale_init`` (e.g. 1.0 with d_embed=27) to get the
        # "fixed small ℓ" regime: K ≈ 1 for clustered positives near
        # inducing points, K ≈ 0 for OOD queries far from them →
        # input-dependent std for OOD detection.
        import math as _math
        base_k = self.svgp.gp.covar_module.base_kernel
        if lengthscale_init is None:
            base_k.lengthscale = float(_math.sqrt(self.svgp.d_model))
        else:
            base_k.lengthscale = float(lengthscale_init)
        # Also bump the ScaleKernel outputscale a bit so absolute kernel
        # magnitudes start large enough that GP outputs have ~unit variance.
        self.svgp.gp.covar_module.outputscale = 2.0
        # Mark fitted so .forward works at inference.
        self.svgp._fitted = torch.tensor(True)

        # 3) Re-toggle requires_grad on the freshly-built modules.
        for p in self.svgp.gp.parameters():
            p.requires_grad_(True)
        for p in self.svgp.likelihood.parameters():
            p.requires_grad_(True)

        # 4) Optimizer on trainable params only. The discriminative energy is the
        #    linear ``energy_head`` (see _e below), so it MUST be trained here.
        params = (list(self.energy_head.parameters())
                  + list(self.svgp.gp.parameters())
                  + list(self.svgp.likelihood.parameters()))
        if train_pooler:
            params = params + list(self.pooler.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        # 5) Training loop: hinge energy E = the linear ``energy_head`` on the
        #    standardised pooled features — NOT the SVGP latent mean. The GP mean
        #    collapses (kernel ~constant at standardised distances → identical
        #    output for clean and invalid → E_clean==E_invalid, no gap). The
        #    linear head reads the (now deterministic, separable) features
        #    directly — exactly the "backbone + head connector" the energy_head
        #    was added for. The GP still trains for the variance/UQ readout.
        def _e(z: torch.Tensor) -> torch.Tensor:
            zs = (z - self.svgp._mu.to(z.device)) / self.svgp._sigma.to(z.device)
            return self.energy_head(zs).squeeze(-1)

        step = 0
        history: list[dict] = []
        for epoch in range(n_epochs):
            for batch in batches:
                if max_steps is not None and step >= max_steps:
                    break
                tok = batch["token_ids"].to(device).long()
                if "token_ids_invalid" not in batch:
                    continue
                tok_inv = batch["token_ids_invalid"].to(device).long()
                B = tok.shape[0]
                t = torch.full((B,), float(t_eval), device=device)
                opt.zero_grad(set_to_none=True)
                # When training the pooler, force autograd through it
                # explicitly — the train_pooler_with_dfm gate (default False)
                # would otherwise detach z and the pooler would never update.
                z_clean = self.pool_features(tok, t, require_grad=train_pooler)
                z_invalid = self.pool_features(tok_inv, t, require_grad=train_pooler)
                e_clean = _e(z_clean)
                e_invalid = _e(z_invalid)
                hinge_clean = e_clean.pow(2).mean()
                hinge_invalid = F.relu(margin_energy - e_invalid).mean()
                hinge = hinge_clean + hinge_invalid
                hinge.backward()
                opt.step()
                step += 1
                # Guard: after the first step the pooler must have received
                # gradient (otherwise unfreezing it + adding it to the optimizer
                # is a silent no-op — only the GP kernel would update).
                if train_pooler and step == 1:
                    if not any(
                        p.grad is not None for p in self.pooler.parameters()
                    ):
                        raise RuntimeError(
                            "fit_svgp_hinge(train_pooler=True): no pooler "
                            "parameter received gradient after the first step. "
                            "z was likely detached — check pool_features grad gate."
                        )
                if verbose and (step % 20 == 0 or step == 1):
                    print(
                        f"  [hinge] step {step:>4}  hinge={hinge.item():.4f}  "
                        f"E_clean={e_clean.mean().item():+.4f}  "
                        f"E_invalid={e_invalid.mean().item():+.4f}  "
                        f"gap={e_invalid.mean().item() - e_clean.mean().item():+.4f}"
                    )
                history.append({
                    "step": step, "hinge": float(hinge.item()),
                    "E_clean": float(e_clean.mean().item()),
                    "E_invalid": float(e_invalid.mean().item()),
                })
            if max_steps is not None and step >= max_steps:
                break

        # 6) Lock SVGP + pooler back off, eval mode.
        for p in self.svgp.gp.parameters():
            p.requires_grad_(False)
        for p in self.svgp.likelihood.parameters():
            p.requires_grad_(False)
        for p in self.pooler.parameters():
            p.requires_grad_(False)
        self.svgp.gp.eval()
        self.svgp.likelihood.eval()
        self.pooler.eval()
        if step == 0 or not history:
            raise RuntimeError(
                "fit_svgp_hinge ran 0 training steps. The loader yielded no "
                "batch with a 'token_ids_invalid' key (or was empty). Pass a "
                "loader whose batches carry token_ids + token_ids_invalid."
            )
        return {"n_steps": step, "history": history,
                "final_hinge": history[-1]["hinge"]}

    def fit_svgp(
        self,
        loader: Iterable[dict],
        *,
        t_eval: float | None = None,
        max_pos: int = 4000,
        neg_strategy: str = "scrambled",
        neg_per_pos_ratio: float = 1.0,
        n_iters: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """End-to-end Stage 2: collect features, fit SVGP, return summary."""
        feats, labels = self.collect_features(
            loader,
            t_eval=t_eval,
            max_pos=max_pos,
            neg_strategy=neg_strategy,
            neg_per_pos_ratio=neg_per_pos_ratio,
        )
        device = next(self.parameters()).device
        last_loss = self.svgp.fit(
            feats.to(device),
            labels.to(device),
            n_iters=n_iters,
            lr=lr,
            verbose=verbose,
        )
        return {
            "elbo_loss": last_loss,
            "n_pos": int((labels == 1).sum().item()),
            "n_neg": int((labels == 0).sum().item()),
            "t_eval": float(t_eval if t_eval is not None else self.cfg.dfm_svgp.t_eval),
        }

    # ----- Inference: OOD score -----------------------------------------
    @torch.no_grad()
    def ood_score(
        self,
        token_ids: torch.Tensor,
        t_eval: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Per-sample OOD diagnostics. Returns dict with keys:
          * ``prob``:   OOD score = sigmoid(energy_head) — the DISCRIMINATIVE
                        signal the hinge trains (high = OOD). NOT the GP
                        Bernoulli mean, which is uninformative (GP-mean collapse).
          * ``energy``: raw energy_head output (hinge target; high = OOD).
          * ``mean``:   SVGP latent mean (diagnostic).
          * ``std``:    SVGP latent std (epistemic UQ — saturates, see findings).
          * ``prob_gp``: the old GP Bernoulli mean (kept for reference).
        """
        dfs = self.cfg.dfm_svgp
        if t_eval is None:
            t_eval = dfs.t_eval
        device = next(self.parameters()).device
        token_ids = token_ids.to(device).long()
        B = token_ids.shape[0]
        t = torch.full((B,), float(t_eval), device=device, dtype=torch.float32)
        z = self.pool_features(token_ids, t)
        prob_gp, mean, std = self.svgp(z)
        zs = (z - self.svgp._mu.to(z.device)) / self.svgp._sigma.to(z.device)
        energy = self.energy_head(zs).squeeze(-1)
        return {"prob": torch.sigmoid(energy), "energy": energy,
                "mean": mean, "std": std, "prob_gp": prob_gp}


@register("DirichletFMSvgp")
def build_dirichlet_fm_svgp(cfg: Config) -> DirichletFMSvgp:
    return DirichletFMSvgp(cfg)
