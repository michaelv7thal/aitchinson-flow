"""DirichletFM auditor on cached LM features (HaluEval-QA, top-K=32).

The original :class:`DirichletFlowMatching` (text8 generator) consumes
``batch["token_ids"]`` (K=27 character ids) and trains an unconditional
denoiser. This auditor variant:

1. Conditions the denoiser on the LM's per-position last-hidden-state
   ``h_LLM`` (768-dim, GPT-2 small) via a learned projection that gets
   concatenated to the simplex point before the input projection
   (``context_features = "product_concat"``, mirroring the EqM auditor).

2. Targets a *slot-of-next-token* label: position k's denoiser predicts
   which of the LM's top-K vocab slots contains the actual next token at
   position k+1. Positions where the actual token is not in the top-K
   are masked out of the CE.

3. **Trains only on clean rows** of the HaluEval cache. The auditor learns
   the geometry of the LM's next-token distribution under un-perturbed
   prompts; hallucinated rows are reserved entirely for evaluation.

4. Evaluation uses the closed-form mixture-of-Dirichlets density induced
   by the trained denoiser:

       log p_t(x | h) = logsumexp_k [
           log p̂_θ(x_1 = e_k | x_t, h)        (denoiser posterior)
         + log Dir(x | β_k(t))                (path prior)
       ]

   where β_k(t) is the per-class concentration vector. This is the EBM
   the writeup pivots on: a *closed-form* per-position UQ score with no
   Hutchinson estimator and no second-order autograd. The score we report
   is :math:`E_t(x_{LM}) = -\\log p_t(x_{LM} \\mid h)` evaluated at the
   LM's actual top-K softmax distribution at that position.
"""

from __future__ import annotations

from typing import Any

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from aitchinson_flow.config import Config
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register


def _sinusoidal_t(t: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal positional embedding of a scalar t. Same recipe as the
    standard transformer time conditioning. ``t``: (B,) → (B, d)."""
    half = d // 2
    freqs = torch.exp(
        torch.linspace(0.0, -8.0, half, device=t.device, dtype=t.dtype)
    )  # log-spaced
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _AuditorBackbone(nn.Module):
    """Transformer encoder for the auditor.

    Inputs:
      x       : (B, L, K)   simplex point per position
      t       : (B,)        Dirichlet path time in [1, t_max]
      h_ctx   : (B, L, H)   LM hidden state per position; optional

    Three context-conditioning modes (mirror the EqM auditor):

    * ``mode = "off"``           — input is the simplex point only.
                                   Tests whether the geometry of the LM's
                                   next-token simplex distribution alone
                                   carries the discriminative signal.
    * ``mode = "hidden_only"``   — input is h_proj(h_LLM) only; the denoiser
                                   never sees the simplex shape directly.
                                   Functionally a per-position MLP probe on
                                   h_LLM under FM training.
    * ``mode = "product_concat"`` — input is concat(x, h_proj(h_LLM)).

    Output:
      logits  : (B, L, K)   denoiser logits over the K top-K slots
    """

    def __init__(
        self,
        *,
        K: int,
        L: int,
        d_model: int,
        n_layers: int,
        n_head: int,
        dropout: float,
        mode: str = "product_concat",
        ctx_hidden: int = 768,
        ctx_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.K = K
        self.L = L
        self.d_model = d_model
        self.mode = mode
        if mode not in ("off", "hidden_only", "product_concat"):
            raise ValueError(f"Unknown context mode: {mode!r}")
        if mode == "off":
            self.h_proj = None
            in_dim = K
        elif mode == "hidden_only":
            self.h_proj = nn.Linear(ctx_hidden, ctx_proj_dim)
            in_dim = ctx_proj_dim
        else:  # product_concat
            self.h_proj = nn.Linear(ctx_hidden, ctx_proj_dim)
            in_dim = K + ctx_proj_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        self.t_proj = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, K)

    def _build_input(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None,
    ) -> torch.Tensor:
        """Conditioning prep shared across backbones: input/pos/time embedding.

        Returns ``z`` of shape (B, L, d_model). Subclasses run their encoder
        on top of this and apply ``self.head``. Splitting this out makes the
        MLP backbone share the exact same conditioning recipe as the
        transformer for a clean ablation.
        """
        if self.mode == "off":
            inp = x
        elif self.mode == "hidden_only":
            if h_ctx is None:
                raise ValueError("hidden_only mode requires h_ctx to be passed")
            inp = self.h_proj(h_ctx)
        else:
            if h_ctx is None:
                h = x.new_zeros(x.shape[:-1] + (self.h_proj.out_features,))
            else:
                h = self.h_proj(h_ctx)
            inp = torch.cat([x, h], dim=-1)
        z = self.input_proj(inp)
        L = z.shape[1]
        pos = torch.arange(L, device=z.device)
        z = z + self.pos_emb(pos).unsqueeze(0)
        t_emb = self.t_proj(_sinusoidal_t(t.to(z.dtype), self.d_model))
        z = z + t_emb[:, None, :]
        return z

    def _encode_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self._build_input(x, t, h_ctx)
        return self.transformer(z)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self._encode_features(x, t, h_ctx))


class _PerPositionMLPBackbone(nn.Module):
    """Per-position MLP encoder — the cascade-clean alternative to the transformer.

    Same input pipeline (input_proj + pos_emb + t_emb) as
    :class:`_AuditorBackbone`, but the encoder is a stack of per-position
    residual MLP blocks with **no cross-positional information flow**. By
    construction, position k's output depends only on inputs at position k.

    Architecturally each block is::

        z ← z + Dropout(Linear(GELU(Linear(LayerNorm(z)))))

    matching the FFN inside a transformer block but without the attention
    sub-layer. Parameter count per block ≈ 8·d², comparable to a transformer
    block's FFN (4·d² in linear1 + 4·d² in linear2; attention is 4·d²
    extra). For fair comparison set ``num_layers`` to roughly 1.5–2× the
    transformer ``num_layers`` to match total parameter budget.
    """

    def __init__(
        self,
        *,
        K: int,
        L: int,
        d_model: int,
        n_layers: int,
        n_head: int,  # unused; kept for constructor-API parity
        dropout: float,
        mode: str = "product_concat",
        ctx_hidden: int = 768,
        ctx_proj_dim: int = 64,
    ) -> None:
        super().__init__()
        del n_head  # unused in MLP variant
        self.K = K
        self.L = L
        self.d_model = d_model
        self.mode = mode
        if mode not in ("off", "hidden_only", "product_concat"):
            raise ValueError(f"Unknown context mode: {mode!r}")
        if mode == "off":
            self.h_proj = None
            in_dim = K
        elif mode == "hidden_only":
            self.h_proj = nn.Linear(ctx_hidden, ctx_proj_dim)
            in_dim = ctx_proj_dim
        else:
            self.h_proj = nn.Linear(ctx_hidden, ctx_proj_dim)
            in_dim = K + ctx_proj_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        self.t_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, K)

    def _build_input(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "off":
            inp = x
        elif self.mode == "hidden_only":
            if h_ctx is None:
                raise ValueError("hidden_only mode requires h_ctx to be passed")
            inp = self.h_proj(h_ctx)
        else:
            if h_ctx is None:
                h = x.new_zeros(x.shape[:-1] + (self.h_proj.out_features,))
            else:
                h = self.h_proj(h_ctx)
            inp = torch.cat([x, h], dim=-1)
        z = self.input_proj(inp)
        L = z.shape[1]
        pos = torch.arange(L, device=z.device)
        z = z + self.pos_emb(pos).unsqueeze(0)
        t_emb = self.t_proj(_sinusoidal_t(t.to(z.dtype), self.d_model))
        z = z + t_emb[:, None, :]
        return z

    def _encode_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self._build_input(x, t, h_ctx)
        for block in self.blocks:
            z = z + block(z)
        return self.norm(z)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self._encode_features(x, t, h_ctx))


class _SVGPHead(nn.Module):
    """Sparse variational GP head over encoder features.

    First-class sibling of :class:`_AuditorBackbone` and
    :class:`_PerPositionMLPBackbone` — same module status, but used post-hoc
    rather than during the main FM training step. Wraps the same gpytorch
    recipe used by ``scripts/phaseF_uq.train_svgp``: ``ApproximateGP`` with
    a ``CholeskyVariationalDistribution`` + ``VariationalStrategy``,
    ScaleKernel(RBFKernel), Bernoulli likelihood, ``VariationalELBO``.

    Lifecycle:

      * ``__init__`` builds the GP eagerly with random inducing points so
        the state_dict is well-formed from the start. SVGP parameters are
        marked ``requires_grad=False`` until ``fit()`` opts them in, so the
        main optimizer (which sees ``model.parameters()``) does not touch
        them during FM training.
      * ``fit(features, labels, ...)`` re-seeds inducing points from the
        empirical training distribution, toggles grads on, optimises
        ``-ELBO`` for ``n_iters`` steps, then toggles grads off again.
      * ``forward(features)`` returns ``(prob, latent_mean, latent_std)``
        per leading-position; runs under ``no_grad`` and standardises by
        the registered ``_mu``/``_sigma`` buffers from the last fit.

    Inputs to ``forward`` may be (..., d_model); the leading dims are
    flattened internally (so (B, L, d_model) → (B, L)).
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_inducing: int = 128,
        kernel: str = "rbf",
    ) -> None:
        super().__init__()
        if kernel not in ("rbf", "matern52", "matern32"):
            raise ValueError(f"unsupported kernel: {kernel!r}")
        self.d_model = d_model
        self.n_inducing = n_inducing
        self.kernel = kernel
        # Standardisation buffers, populated by fit().
        self.register_buffer("_mu", torch.zeros(d_model))
        self.register_buffer("_sigma", torch.ones(d_model))
        self.register_buffer("_fitted", torch.tensor(False))
        # Build the gpytorch components eagerly so state_dict is stable.
        # Random inducing points are placeholders until fit() reseeds them.
        ip0 = torch.randn(n_inducing, d_model)
        self.gp = _build_svgp_module(ip0, kernel=kernel)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        # Don't include SVGP params in the main FM optimizer's update.
        for p in self.gp.parameters():
            p.requires_grad_(False)
        for p in self.likelihood.parameters():
            p.requires_grad_(False)

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        n_iters: int = 200,
        lr: float = 0.01,
        seed: int = 0,
        verbose: bool = False,
    ) -> float:
        """Train the variational params on ``(N, d_model)`` features.

        ``labels`` is (N,) binary (0/1, float). Returns the final ELBO loss.
        Re-seeds inducing points from a random subset of the standardised
        training set; updates the ``_mu``/``_sigma`` standardisation
        buffers; toggles SVGP param grads on for the duration of the fit
        and back off afterwards (so the main optimizer remains unaware).
        """
        torch.manual_seed(seed)
        device = features.device
        if features.dim() != 2 or features.shape[1] != self.d_model:
            raise ValueError(
                f"expected features (N, {self.d_model}); got {tuple(features.shape)}"
            )
        if labels.dim() != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"labels shape mismatch: {tuple(labels.shape)} vs {features.shape[0]}"
            )

        mu = features.mean(0)
        sigma = features.std(0).clamp_min(1e-6)
        Xs = (features - mu) / sigma
        # Re-seed inducing points from data and rebuild the gp module so
        # variational distribution dims line up.
        perm = torch.randperm(Xs.shape[0], device=device)[: self.n_inducing]
        ip = Xs[perm].detach().clone()
        self.gp = _build_svgp_module(ip, kernel=self.kernel).to(device)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
        self._mu = mu.detach()
        self._sigma = sigma.detach()

        # Opt-in grads for the fit window only.
        for p in self.gp.parameters():
            p.requires_grad_(True)
        for p in self.likelihood.parameters():
            p.requires_grad_(True)

        self.gp.train()
        self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.gp.parameters()) + list(self.likelihood.parameters()),
            lr=lr,
        )
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=Xs.shape[0]
        )
        y = labels.to(device).float()
        last_loss = float("nan")
        for it in range(n_iters):
            opt.zero_grad()
            out = self.gp(Xs)
            loss = -mll(out, y)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            if verbose and (it + 1) % max(1, n_iters // 5) == 0:
                print(f"    [SVGP] iter {it + 1}/{n_iters}  loss={last_loss:.4f}")
        self.gp.eval()
        self.likelihood.eval()

        # Lock grads back off so the main optimizer ignores SVGP params.
        for p in self.gp.parameters():
            p.requires_grad_(False)
        for p in self.likelihood.parameters():
            p.requires_grad_(False)
        self._fitted = torch.tensor(True)
        return last_loss

    @torch.no_grad()
    def forward(
        self,
        features: torch.Tensor,
        *,
        batch_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict per-position posterior on (..., d_model) features.

        Returns ``(prob, mean, std)`` with the same leading shape as input
        (without the trailing d_model). All three are on the same device as
        the input and share dtype with the GP's parameters.
        """
        if not bool(self._fitted.item()):
            raise RuntimeError("_SVGPHead.forward called before fit()")
        orig_shape = features.shape[:-1]
        device = next(self.gp.parameters()).device
        Xs = (features - self._mu.to(features.device)) / self._sigma.to(features.device)
        Xs = Xs.reshape(-1, self.d_model).to(device)
        means, stds, probs = [], [], []
        with gpytorch.settings.fast_pred_var():
            for s in range(0, Xs.shape[0], batch_size):
                e = min(s + batch_size, Xs.shape[0])
                f_dist = self.gp(Xs[s:e])
                means.append(f_dist.mean)
                stds.append(f_dist.variance.clamp_min(1e-12).sqrt())
                probs.append(self.likelihood(f_dist).mean)
        out_dev = features.device
        prob = torch.cat(probs).reshape(*orig_shape).to(out_dev)
        mean = torch.cat(means).reshape(*orig_shape).to(out_dev)
        std = torch.cat(stds).reshape(*orig_shape).to(out_dev)
        return prob, mean, std


def _build_svgp_module(
    inducing_points: torch.Tensor, kernel: str = "rbf"
) -> gpytorch.models.ApproximateGP:
    """Internal: gpytorch SVGP module factory. ScaleKernel + ConstantMean.

    Split out so ``_SVGPHead.fit`` can rebuild the module fresh after
    re-seeding inducing points (``CholeskyVariationalDistribution``
    bakes the inducing-set size into its parameter shapes).

    Kernel choices:
      * ``"rbf"``     — RBFKernel; C∞ samples, fast variance saturation
                        far from inducing points. Original auditor default.
      * ``"matern52"`` — MaternKernel(ν=5/2); C² samples, heavier tails,
                        better-graded OOD variance and more numerically
                        stable. Default in BO literature (Snoek 2012) and
                        in this repo's pure-PyTorch ``_SparseGP``.
      * ``"matern32"`` — MaternKernel(ν=3/2); even rougher (C¹).
    """
    if kernel == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel()
    elif kernel == "matern52":
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel == "matern32":
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
    else:
        raise ValueError(f"unknown kernel: {kernel!r}")

    class _SVGP(gpytorch.models.ApproximateGP):
        def __init__(self, ip: torch.Tensor, base_k: gpytorch.kernels.Kernel) -> None:
            vd = gpytorch.variational.CholeskyVariationalDistribution(ip.size(0))
            vs = gpytorch.variational.VariationalStrategy(
                self, ip, vd, learn_inducing_locations=True
            )
            super().__init__(vs)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(base_k)

        def forward(self, x: torch.Tensor):  # noqa: D401
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return _SVGP(inducing_points, base_kernel)


def _dirichlet_logp_mixture(
    x: torch.Tensor,
    posterior_logits: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Closed-form ``log p_t(x | h)`` for the Dirichlet conditional path.

    Stark et al. parameterise ``p(x_t | x_1 = e_k) = Dir(x_t; β_k(t))`` with
    ``β_k(t)_i = t · 1[i = k] + 1[i ≠ k]``, i.e. the all-ones vector with a
    ``t`` in slot k. Given the posterior ``p̂(x_1 = e_k | x_t, h) =
    softmax(posterior_logits)_k`` and the prior over k that produced it
    (which Bayes-cancels in the mixture below), the marginal density is

        log p_t(x | h) = logsumexp_k [ log p̂_k + log Dir(x | β_k(t)) ]

    Inputs:
      x                  (B, L, K)  point on the simplex (rows sum to 1)
      posterior_logits   (B, L, K)  denoiser logits
      t                  scalar     Dirichlet path time t ≥ 1

    Returns:
      log_p              (B, L)     per-position log density
    """
    B, L, K = x.shape
    log_post = F.log_softmax(posterior_logits, dim=-1)  # (B, L, K)

    # log Dir(x | β_k) for each k. β_k has t in slot k and 1 elsewhere.
    # log B(β) = sum_i lgamma(β_i) - lgamma(sum_i β_i)
    #          = lgamma(t) + (K-1)*lgamma(1) - lgamma(t + K - 1)
    #          = lgamma(t) - lgamma(t + K - 1)
    # log Dir(x | β_k) = (β_k_i - 1) · log(x_i) summed - log B(β_k)
    #                  = (t - 1) · log(x_k)               - log B(β_k)
    # because (1 - 1)·log(x_j) = 0 for j ≠ k.
    log_x = x.clamp_min(1e-12).log()  # (B, L, K)
    log_normaliser = torch.lgamma(
        torch.tensor(t, dtype=x.dtype, device=x.device)
    ) - torch.lgamma(
        torch.tensor(t + K - 1, dtype=x.dtype, device=x.device)
    )
    # log Dir(x | β_k) for class k = (t-1)·log(x_k) - log_normaliser
    log_dir = (t - 1.0) * log_x - log_normaliser  # (B, L, K), one per k
    log_p = torch.logsumexp(log_post + log_dir, dim=-1)  # (B, L)
    return log_p


class DirichletFMAuditor(nn.Module):
    """DirichletFM trained on cached LM features for hallucination UQ."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        K = int(cfg.hallueval_dfm_auditor.batch_size)  # placeholder; set below
        # The K and L for the auditor come from the cache, not from text8_dataset.
        # We accept them as init-time arguments through the meta dict.
        # For factory-built models we read them from the datamodule meta later
        # via `set_dims(K, L)`. To keep the constructor tolerant, fall back to
        # the topk-cache defaults (K=32, L=160) if `set_dims` was not called.
        self._K_default = 32
        self._L_default = 160
        self._build_backbone(self._K_default, self._L_default)

    def _build_backbone(self, K: int, L: int) -> None:
        cfg = self.cfg
        self.K = K
        self.L = L
        mode = cfg.dirichlet_fm.context_features
        kind = cfg.dirichlet_fm.backbone_kind
        backbone_kwargs = dict(
            K=K, L=L,
            d_model=cfg.transformer.d_model,
            n_layers=cfg.transformer.num_layers,
            n_head=cfg.transformer.nhead,
            dropout=cfg.dirichlet_fm.dropout,
            mode=mode,
            ctx_hidden=cfg.dirichlet_fm.ctx_hidden,
            ctx_proj_dim=cfg.dirichlet_fm.ctx_proj_dim,
        )
        if kind == "transformer":
            self.backbone = _AuditorBackbone(**backbone_kwargs)
        elif kind == "mlp":
            self.backbone = _PerPositionMLPBackbone(**backbone_kwargs)
        else:
            raise ValueError(f"Unknown backbone_kind: {kind!r}")
        # Architecture B: optional supervised hallucination head sharing the
        # encoder. Only instantiated when joint_halluc=True so checkpoints
        # remain state_dict-compatible across modes.
        if cfg.dirichlet_fm.joint_halluc:
            d = cfg.transformer.d_model
            self.halluc_head: nn.Module | None = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
            )
        else:
            self.halluc_head = None
        # Architecture A: optional sparse variational GP head over encoder
        # features. Same conditional-instantiation pattern as halluc_head so
        # checkpoints remain state_dict-compatible across the on/off setting.
        # Trained post-hoc via ``fit_svgp(loader)``; queried via
        # ``svgp_score_at_lm(batch)``. Its parameters live in the model but
        # carry ``requires_grad=False`` outside ``fit()``, so the main FM
        # optimizer ignores them.
        if cfg.dirichlet_fm.svgp_head:
            self.svgp_head: _SVGPHead | None = _SVGPHead(
                d_model=cfg.transformer.d_model,
                n_inducing=cfg.dirichlet_fm.svgp_n_inducing,
            )
        else:
            self.svgp_head = None

    def set_dims(self, K: int, L: int) -> None:
        """Re-instantiate the backbone with the actual cache dims.

        Should be called once after the datamodule is built and before any
        training / loading. Modules created by the factory default to
        (K=32, L=160) which matches the existing topk cache; other shapes
        require this call.
        """
        if (K, L) != (self.K, self.L):
            self._build_backbone(K, L)

    @property
    def t_max(self) -> float:
        return float(self.cfg.dirichlet_fm.t_max)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone(x, t, h_ctx)

    def forward_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encoder output (B, L, d_model) before the linear slot head.

        Used by downstream supervised UQ heads (Architecture A — SVGP on
        these features). Both backbones expose ``_encode_features`` with
        the same signature, so this is just a delegate.
        """
        return self.backbone._encode_features(x, t, h_ctx)

    def _sample_xt(self, slot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ Dir(β(t, slot)). slot: (B, L) long, t: (B,) float.

        Positions with slot < 0 (no in-top-K target) are sampled from the
        uniform Dirichlet (β = all-ones) — the per-position CE later masks
        them out anyway.
        """
        B, L = slot.shape
        K = self.K
        beta = torch.ones(B, L, K, device=slot.device, dtype=t.dtype)
        valid = slot >= 0
        slot_safe = slot.clamp_min(0).unsqueeze(-1)
        # scatter t into the slot column where valid
        beta_target = t[:, None, None].expand(B, L, 1).to(beta.dtype)
        beta = beta.scatter(-1, slot_safe, beta_target)
        # Where invalid, restore the all-ones (we set t into slot 0 above
        # because slot_safe was clamped; restore it).
        if (~valid).any():
            inv = (~valid).unsqueeze(-1).expand(B, L, K)
            beta = torch.where(inv, torch.ones_like(beta), beta)
        return Dirichlet(beta).sample()

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        device = batch["topk_logp"].device
        labels = batch["label"].bool()                  # (B,)
        topk_logp_all = batch["topk_logp"]              # (B, L, K)
        next_slot_all = batch["next_slot"]              # (B, L)
        hidden_all = batch["hidden"].float()            # (B, L, H)
        answer_all = batch["answer_mask"]               # (B, L)

        # ---------- slot-prediction CE on CLEAN rows (or all rows) ----------
        # ``train_clean_only=True`` (default) is the one-class density recipe:
        # the slot head only sees label==0 rows, so the EBM models "what
        # normal looks like." Setting it False is the strictly self-
        # supervised variant — slot CE on every row, no label inspection.
        if self.cfg.dirichlet_fm.train_clean_only:
            clean_mask = ~labels
        else:
            clean_mask = torch.ones_like(labels)
        slot_loss = torch.zeros((), device=device, dtype=hidden_all.dtype)
        slot_logits_clean: torch.Tensor | None = None
        z_clean: torch.Tensor | None = None
        t_clean: torch.Tensor | None = None
        if clean_mask.any():
            topk_c = topk_logp_all[clean_mask]
            slot_c = next_slot_all[clean_mask]
            hidden_c = hidden_all[clean_mask]
            answer_c = answer_all[clean_mask]
            B_c, L_c, K_c = topk_c.shape

            t_c = 1.0 + (self.t_max - 1.0) * torch.rand(B_c, device=device)
            x_t = self._sample_xt(slot_c, t_c)
            # We need the encoder features too (for the joint halluc head).
            # `forward_features` returns (B, L, d_model); apply the slot head
            # on top to recover slot logits without a second forward.
            z_c = self.forward_features(x_t, t_c, h_ctx=hidden_c)
            slot_logits = self.backbone.head(z_c)
            valid = (slot_c >= 0) & answer_c
            if valid.any():
                slot_loss = F.cross_entropy(slot_logits[valid], slot_c[valid].long())
            slot_logits_clean = slot_logits
            z_clean = z_c
            t_clean = t_c

        # ---------- hallucination BCE on ALL rows (Architecture B) ----------
        halluc_loss = torch.zeros((), device=device, dtype=hidden_all.dtype)
        n_halluc_valid = 0
        if self.halluc_head is not None:
            B_a, L_a, K_a = topk_logp_all.shape
            t_a = 1.0 + (self.t_max - 1.0) * torch.rand(B_a, device=device)
            # Sample x_t for ALL rows from a "fake" target slot (use top-1 of
            # the LM as a stable anchor; the halluc head gradient does not
            # need a meaningful FM target — it just needs encoder features
            # under the same path family the slot head sees).
            top1 = topk_logp_all.argmax(dim=-1)            # (B, L) — slot-0 ≡ top-1
            # If next_slot is available (not -1) prefer it; else fall back to
            # the LM top-1 slot (which is always 0 in our cache layout).
            slot_anchor = torch.where(next_slot_all >= 0, next_slot_all, top1)
            x_t_a = self._sample_xt(slot_anchor, t_a)
            z_a = self.forward_features(x_t_a, t_a, h_ctx=hidden_all)
            halluc_logits = self.halluc_head(z_a).squeeze(-1)  # (B, L)
            # Per-row label, broadcast over positions; only score answer-mask.
            target_pos = (
                labels.float().unsqueeze(-1).expand(B_a, L_a)
            )
            valid_pos = answer_all
            if valid_pos.any():
                pw = self.cfg.dirichlet_fm.halluc_pos_weight
                if pw > 0:
                    pos_weight = torch.tensor(pw, device=device, dtype=halluc_logits.dtype)
                    halluc_loss = F.binary_cross_entropy_with_logits(
                        halluc_logits[valid_pos],
                        target_pos[valid_pos],
                        pos_weight=pos_weight,
                    )
                else:
                    halluc_loss = F.binary_cross_entropy_with_logits(
                        halluc_logits[valid_pos], target_pos[valid_pos]
                    )
                n_halluc_valid = int(valid_pos.sum().item())

        lam_s = self.cfg.dirichlet_fm.lambda_slot
        lam_h = self.cfg.dirichlet_fm.lambda_halluc
        total = lam_s * slot_loss + lam_h * halluc_loss

        # Edge case: a batch with no clean rows (rare with shuffled data
        # at small batch sizes) and no halluc head leaves both terms as
        # detached scalars; manufacture a grad-bearing zero so the runner's
        # backward() doesn't error. Same pattern as the original training_step.
        if not total.requires_grad:
            param0 = next(self.parameters())
            total = total + (param0.sum() * 0.0)

        out: LossDict = {TRAINING_LOSS_KEY: total, "ce_slot": slot_loss.detach()}
        if self.halluc_head is not None:
            out["bce_halluc"] = halluc_loss.detach()
            out["n_halluc"] = torch.tensor(float(n_halluc_valid), device=device)
        return out

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        # On clean rows, report the same CE for monitoring.
        clean_mask = ~batch["label"].bool()
        if not clean_mask.any():
            zero = torch.zeros((), device=batch["topk_logp"].device)
            return {TRAINING_LOSS_KEY: zero}
        topk_logp = batch["topk_logp"][clean_mask]
        next_slot = batch["next_slot"][clean_mask]
        hidden = batch["hidden"][clean_mask].float()
        answer_mask = batch["answer_mask"][clean_mask]

        B, L, K = topk_logp.shape
        device = topk_logp.device
        t = 1.0 + (self.t_max - 1.0) * torch.rand(B, device=device)
        x_t = self._sample_xt(next_slot, t)
        logits = self.forward(x_t, t, h_ctx=hidden)
        valid = (next_slot >= 0) & answer_mask
        if not valid.any():
            return {TRAINING_LOSS_KEY: torch.zeros((), device=device)}
        loss = F.cross_entropy(logits[valid], next_slot[valid].long())
        return {TRAINING_LOSS_KEY: loss}

    @torch.no_grad()
    def energy_at_lm_distribution(
        self,
        batch: Any,
        *,
        t: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Closed-form EBM evaluated at the LM's actual top-K softmax point.

        Returns per-position scores (and aggregated row-level scores) for the
        whole batch. ``label``, ``answer_mask``, and ``DeltaE`` are passed
        through so the caller can compute AUROC against the per-row label.

        Returns dict:
          ``E_pos``       (B, L) — per-position negative log density
          ``E_pos_clip``  (B, L) — same, with non-answer positions zeroed
          ``E_seq_mean``  (B,)   — mean over answer positions
          ``E_seq_max``   (B,)   — max over answer positions
          ``label``       (B,)
          ``DeltaE_seq``  (B,)   — paper ΔE summed over answer positions
        """
        if t is None:
            t = float(self.cfg.dirichlet_fm.energy_t)

        topk_logp = batch["topk_logp"].float()  # (B, L, K)
        hidden = batch["hidden"].float()        # (B, L, H)
        answer_mask = batch["answer_mask"].bool()
        label = batch["label"].bool()

        # The simplex point we score is the LM's actual top-K softmax. Since
        # topk_logp are *log* probabilities of the top-K slots from the LM,
        # they already sum-exp to <= 1 across the slots; renormalise so each
        # row sums to exactly 1 on the K-simplex (the leftover mass at the
        # bottom of the long tail is dropped — same convention as Phase Q).
        x = topk_logp.exp()
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        B, L, K = x.shape
        device = x.device
        t_batch = torch.full((B,), float(t), device=device, dtype=x.dtype)
        # Run the denoiser at the LM's distribution (not at a sampled x_t).
        logits = self.forward(x, t_batch, h_ctx=hidden)
        log_p = _dirichlet_logp_mixture(x, logits, t=float(t))  # (B, L)

        E_pos = -log_p
        E_pos_clip = torch.where(answer_mask, E_pos, torch.zeros_like(E_pos))
        denom = answer_mask.sum(dim=-1).clamp_min(1).to(E_pos.dtype)
        E_seq_mean = E_pos_clip.sum(dim=-1) / denom
        # Mask out non-answer positions for the max
        very_neg = torch.full_like(E_pos, float("-inf"))
        E_for_max = torch.where(answer_mask, E_pos, very_neg)
        E_seq_max = E_for_max.max(dim=-1).values
        # If a row has no answer tokens, max → -inf; replace with mean for safety.
        no_answer = (answer_mask.sum(dim=-1) == 0)
        E_seq_max = torch.where(no_answer, E_seq_mean, E_seq_max)

        DeltaE = batch.get("DeltaE")
        if DeltaE is not None:
            DeltaE_clip = torch.where(
                answer_mask, DeltaE.float(), torch.zeros_like(DeltaE.float())
            )
            DeltaE_seq = DeltaE_clip.sum(dim=-1)
        else:
            DeltaE_seq = torch.zeros(B, device=device)

        return {
            "E_pos": E_pos,
            "E_pos_clip": E_pos_clip,
            "E_seq_mean": E_seq_mean,
            "E_seq_max": E_seq_max,
            "label": label,
            "DeltaE_seq": DeltaE_seq,
            "answer_mask": answer_mask,
        }


    @torch.no_grad()
    def halluc_score_at_lm(
        self,
        batch: Any,
        *,
        t: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Architecture-B per-position hallucination probability.

        Evaluated at the LM's actual top-K simplex distribution and the
        configured ``energy_t``. Returns the same set of fields as
        ``energy_at_lm_distribution`` so the eval driver can swap signals.

        Requires the model to have been built with
        ``cfg.dirichlet_fm.joint_halluc=True``.
        """
        if self.halluc_head is None:
            raise RuntimeError(
                "halluc_score_at_lm requires joint_halluc=True (Architecture B)"
            )
        if t is None:
            t = float(self.cfg.dirichlet_fm.energy_t)
        topk_logp = batch["topk_logp"].float()
        hidden = batch["hidden"].float()
        answer_mask = batch["answer_mask"].bool()
        label = batch["label"].bool()

        x = topk_logp.exp()
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        B, L, _ = x.shape
        device = x.device
        t_batch = torch.full((B,), float(t), device=device, dtype=x.dtype)
        z = self.forward_features(x, t_batch, h_ctx=hidden)
        logits_pos = self.halluc_head(z).squeeze(-1)  # (B, L)
        prob_pos = torch.sigmoid(logits_pos)
        # Row score = mean over answer mask
        denom = answer_mask.sum(dim=-1).clamp_min(1).to(prob_pos.dtype)
        masked = torch.where(answer_mask, prob_pos, torch.zeros_like(prob_pos))
        prob_seq = masked.sum(dim=-1) / denom
        return {
            "prob_pos": prob_pos,
            "logit_pos": logits_pos,
            "prob_seq": prob_seq,
            "answer_mask": answer_mask,
            "label": label,
        }

    def attach_svgp_head(
        self,
        *,
        n_inducing: int | None = None,
    ) -> None:
        """Bolt an :class:`_SVGPHead` onto the model post-construction.

        Use this when loading a legacy checkpoint that was trained without
        the SVGP head — building it at constructor time and then calling
        ``load_state_dict`` triggers a gpytorch upstream hook that crashes
        on empty submodule dicts. Building it after the load avoids that
        path. After attach, call :meth:`fit_svgp` to populate variational
        params.
        """
        if self.svgp_head is not None:
            return  # already attached; idempotent
        if n_inducing is None:
            n_inducing = self.cfg.dirichlet_fm.svgp_n_inducing
        self.svgp_head = _SVGPHead(
            d_model=self.cfg.transformer.d_model,
            n_inducing=n_inducing,
        ).to(next(self.parameters()).device)

    def fit_svgp(
        self,
        loader: Any,
        *,
        device: torch.device | str | None = None,
        n_iters: int | None = None,
        lr: float | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> float:
        """Fit the SVGP head on encoder features over the given loader.

        Gathers encoder features at the LM's actual top-K simplex (the same
        score-time geometry used by ``energy_at_lm_distribution``) over all
        answer-mask positions in ``loader``, with the per-row label
        broadcast over those positions. Then trains the SVGP variational
        params for ``n_iters`` steps. Defaults read from
        ``cfg.dirichlet_fm.svgp_n_iters`` / ``svgp_lr``.

        Returns the final ELBO loss.
        """
        if self.svgp_head is None:
            raise RuntimeError(
                "fit_svgp requires svgp_head=True in cfg.dirichlet_fm"
            )
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        n_iters = n_iters if n_iters is not None else self.cfg.dirichlet_fm.svgp_n_iters
        lr = lr if lr is not None else self.cfg.dirichlet_fm.svgp_lr

        feats: list[torch.Tensor] = []
        labs: list[torch.Tensor] = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                topk_logp = batch["topk_logp"].float()
                hidden = batch["hidden"].float()
                answer_mask = batch["answer_mask"].bool()
                label = batch["label"].bool()
                x = topk_logp.exp()
                x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                B, L, _ = x.shape
                t_batch = torch.full(
                    (B,), float(self.cfg.dirichlet_fm.energy_t),
                    device=device, dtype=x.dtype,
                )
                z = self.forward_features(x, t_batch, h_ctx=hidden)  # (B, L, d)
                lab_pos = label.unsqueeze(-1).expand(B, L)
                feats.append(z[answer_mask].detach().cpu())
                labs.append(lab_pos[answer_mask].detach().cpu().long())
        if was_training:
            self.train()
        if not feats:
            raise RuntimeError("fit_svgp: no answer-mask positions found in loader")
        X = torch.cat(feats, dim=0).to(device)
        y = torch.cat(labs, dim=0).to(device).float()
        return self.svgp_head.fit(
            X, y, n_iters=n_iters, lr=lr, seed=seed, verbose=verbose,
        )

    @torch.no_grad()
    def svgp_score_at_lm(
        self,
        batch: Any,
        *,
        t: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Architecture-A per-position SVGP posterior at the LM's distribution.

        Evaluated at the same simplex point and time as
        ``energy_at_lm_distribution`` and ``halluc_score_at_lm`` so the
        three readouts can be compared directly. Returns ``prob_pos``
        (Bernoulli posterior mean), ``mean_pos`` and ``std_pos`` (latent
        Gaussian posterior), plus row-level pooled ``prob_seq`` over the
        answer mask. Requires ``svgp_head=True`` and a prior call to
        ``fit_svgp(loader)``.
        """
        if self.svgp_head is None:
            raise RuntimeError(
                "svgp_score_at_lm requires svgp_head=True"
            )
        if t is None:
            t = float(self.cfg.dirichlet_fm.energy_t)
        topk_logp = batch["topk_logp"].float()
        hidden = batch["hidden"].float()
        answer_mask = batch["answer_mask"].bool()
        label = batch["label"].bool()
        x = topk_logp.exp()
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        B, L, _ = x.shape
        device = x.device
        t_batch = torch.full((B,), float(t), device=device, dtype=x.dtype)
        z = self.forward_features(x, t_batch, h_ctx=hidden)  # (B, L, d)
        prob_pos, mean_pos, std_pos = self.svgp_head(z)
        denom = answer_mask.sum(dim=-1).clamp_min(1).to(prob_pos.dtype)
        masked = torch.where(answer_mask, prob_pos, torch.zeros_like(prob_pos))
        prob_seq = masked.sum(dim=-1) / denom
        return {
            "prob_pos": prob_pos,
            "mean_pos": mean_pos,
            "std_pos": std_pos,
            "prob_seq": prob_seq,
            "answer_mask": answer_mask,
            "label": label,
        }

    @torch.no_grad()
    def sample_conditional(
        self,
        h_ctx: torch.Tensor,
        *,
        nfe: int | None = None,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Conditional sampling: integrate the marginal vector field from
        x ~ Dir(1) at t=1 to t=t_max under fixed h_ctx.

        h_ctx: (B, L, H) — held-out hidden states from real LM forward
        passes. The denoiser conditions on these throughout integration.

        Returns argmax slot indices (B, L) — caller maps slot → vocab via
        the chunk's `topk_idx`.

        Math (Stark et al. Theorem 3.1, ``α(t) = t``):
            u_t(x | c) = ċ_t(x_c) · (e_c - x) / (1 - x_c),
            v(x, t)   = Σ_c p̂_c · u_t(x | c)
                      = w − x · Σ_c w_c,    w_c = p̂_c · ċ_t(x_c) / (1 − x_c)
        ``ċ_t(u) = -∂_α I_u(α, K-1) / Beta_pdf(u; α, K-1)`` is computed via
        the same scipy-based central difference used by the unconditional
        DirichletFM sampler.
        """
        # Local imports — keep the module light at import time.
        import numpy as np
        from scipy.special import betainc, gammaln

        if nfe is None:
            nfe = self.cfg.dirichlet_fm.sample_nfe
        device = h_ctx.device
        dtype = h_ctx.dtype
        B, L, _ = h_ctx.shape
        K = self.K

        x = Dirichlet(torch.ones(B, L, K, device=device, dtype=dtype)).sample()
        t_grid = torch.linspace(1.0, self.t_max, nfe + 1, device=device)

        def _c_dot_factor(x_np: "np.ndarray", t: float) -> "np.ndarray":
            a_plus, a_minus = t + eps, max(t - eps, 1e-3)
            F_plus = betainc(a_plus, K - 1, x_np)
            F_minus = betainc(a_minus, K - 1, x_np)
            dF_da = (F_plus - F_minus) / (a_plus - a_minus)
            log_B = gammaln(t) + gammaln(K - 1) - gammaln(t + K - 1)
            log_q = (t - 1.0) * np.log(np.clip(x_np, 1e-10, 1.0 - 1e-10)) + (
                (K - 2) * np.log(np.clip(1.0 - x_np, 1e-10, 1.0 - 1e-10))
            ) - log_B
            log_q = np.clip(log_q, -60.0, 60.0)
            q = np.exp(log_q)
            return -dF_da / np.maximum(q, 1e-12)

        for i in range(nfe):
            t = float(t_grid[i].item())
            dt = float((t_grid[i + 1] - t_grid[i]).item())
            t_batch = torch.full((B,), t, device=device, dtype=dtype)

            logits = self.forward(x, t_batch, h_ctx=h_ctx)
            p1 = logits.softmax(dim=-1)

            x_clamped = x.clamp(min=1e-6, max=1.0 - 1e-6)
            c_dot = torch.from_numpy(
                _c_dot_factor(x_clamped.detach().cpu().double().numpy(), t)
            ).to(device=device, dtype=dtype)

            denom = (1.0 - x_clamped).clamp(min=1e-6)
            w = p1 * c_dot / denom
            sum_w = w.sum(dim=-1, keepdim=True)
            v = w - x * sum_w
            x = x + dt * v
            x = x.clamp(min=1e-6)
            x = x / x.sum(dim=-1, keepdim=True)

        return x.argmax(dim=-1)


@register("DirichletFMAuditor")
def build_dirichlet_fm_auditor(cfg: Config) -> DirichletFMAuditor:
    return DirichletFMAuditor(cfg)
