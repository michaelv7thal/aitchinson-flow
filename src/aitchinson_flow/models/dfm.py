from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from aitchinson_flow.config import Config
from aitchinson_flow.transformer_backbone import DFMBackbone, DFMHead
from aitchinson_flow.models import LossDict, TRAINING_LOSS_KEY, register


class DiscreteFlowMatching(nn.Module):
    """Discrete Flow Matching (Gat et al. 2024) adapted to text8.

    Source distribution: Uniform over K=27 characters.
    Probability path: p_t(x^i | x1) = κ_t·δ(x1^i) + (1-κ_t)·U(K)
    Training loss: cross-entropy E_t[-log p_{1|t}(x1^i | x_t)]
    Sampling: Euler method (Algorithm 1) using probability velocity (Eq. 24)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DFMBackbone(cfg=cfg)
        self.head = DFMHead(cfg=cfg)

    def _kappa(self, t: torch.Tensor) -> torch.Tensor:
        """Noise schedule κ_t ∈ [0, 1]. Quadratic (t²) or linear (t)."""
        if self.cfg.dfm.kappa_schedule == "quadratic":
            return t ** 2
        return t

    def _kappa_dot(self, t: torch.Tensor) -> torch.Tensor:
        """Time derivative κ̇_t."""
        if self.cfg.dfm.kappa_schedule == "quadratic":
            return 2.0 * t
        return torch.ones_like(t)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, L) token IDs (long), t: (B,) float in [0, 1] → logits (B, L, K)."""
        K = self.cfg.text8_dataset.K
        x_onehot = F.one_hot(x_t, K).float()
        h = self.backbone(x_onehot, t)
        return self.head(h)

    def _corrupt(
        self, token_ids: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Corrupt clean token_ids to x_t using the probability path.

        Each position is kept clean w.p. κ_t and replaced by a uniform random
        token w.p. (1 - κ_t), independently across positions and batch items.
        """
        B, L = token_ids.shape
        K = self.cfg.text8_dataset.K
        kappa_t = self._kappa(t)  # (B,)
        keep_mask = torch.bernoulli(kappa_t[:, None].expand(B, L)).bool()
        noise = torch.randint(0, K, (B, L), device=token_ids.device)
        return torch.where(keep_mask, token_ids, noise)

    def training_step(self, batch: Any, step: int) -> LossDict:
        del step
        token_ids = batch["token_ids"].long()  # (B, L)
        B = token_ids.shape[0]
        device = token_ids.device

        t = torch.rand(B, device=device)
        x_t = self._corrupt(token_ids, t)
        logits = self.forward(x_t, t)  # (B, L, K)

        K = self.cfg.text8_dataset.K
        loss = F.cross_entropy(logits.reshape(-1, K), token_ids.reshape(-1))
        return {TRAINING_LOSS_KEY: loss}

    @torch.no_grad()
    def eval_step(self, batch: Any) -> LossDict:
        return self.training_step(batch, 0)

    @torch.no_grad()
    def sample(self, B: int, L: int, *, nfe: int | None = None) -> torch.Tensor:
        """Euler sampling (Algorithm 1 in DFM paper).

        Starts from uniform-random tokens (t=0) and advances to t=1 via
        `nfe` Euler steps using the probability velocity (Eq. 24).

        Returns (B, L) long tensor of sampled token IDs.
        """
        if nfe is None:
            nfe = self.cfg.dfm.sample_nfe

        K = self.cfg.text8_dataset.K
        device = next(self.parameters()).device

        # t=0: fully noisy (uniform over K)
        x = torch.randint(0, K, (B, L), device=device)

        t_vals = torch.linspace(0.0, 1.0, nfe + 1, device=device)
        for i in range(nfe):
            t = t_vals[i]
            h = t_vals[i + 1] - t  # step size = 1/nfe

            kappa_t = self._kappa(t)
            kappa_dot_t = self._kappa_dot(t)

            t_batch = t.expand(B)
            logits = self.forward(x, t_batch)
            p1 = logits.softmax(dim=-1)  # (B, L, K) denoiser predictions

            x_oh = F.one_hot(x, K).float()  # (B, L, K)

            # Probability velocity: u_t = (κ̇_t / (1 - κ_t)) · (p1 - δ(x_t))
            denom = (1.0 - kappa_t).clamp(min=1e-8)
            rate = kappa_dot_t / denom
            velocity = rate * (p1 - x_oh)  # (B, L, K)

            # Euler update: p_{t+h} ≈ δ(x_t) + h · u_t
            p_new = x_oh + h * velocity
            p_new = p_new.clamp(min=0.0)
            p_new = p_new / p_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            x = Categorical(probs=p_new).sample()  # (B, L)

        return x

    def _corrupt_kappa(
        self, token_ids: torch.Tensor, kappa: torch.Tensor
    ) -> torch.Tensor:
        """Corrupt clean tokens at an explicit per-batch keep-probability κ.

        Like ``_corrupt`` but takes κ directly (the ELBO discretizes κ on a
        grid rather than via t)."""
        B, L = token_ids.shape
        K = self.cfg.text8_dataset.K
        keep = torch.bernoulli(kappa[:, None].expand(B, L)).bool()
        noise = torch.randint(0, K, (B, L), device=token_ids.device)
        return torch.where(keep, token_ids, noise)

    @torch.no_grad()
    def elbo_bpc(
        self, token_ids: torch.Tensor, *, n_mc: int = 8, n_steps: int = 1000,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """Variational NLL bound in **bits-per-character** (a genuine upper
        bound on −log p(x)/(L·log2)), so directly comparable to published
        text8 BPC (SEDD 1.32 / D3PM-uniform 1.61 / MDLM ≤1.38).

        This is the standard D3PM discrete-time evidence bound (Austin et al.
        2021) for the **uniform** forward process this model actually uses:
        the corrupted marginal is ``q(x_τ|x₁)=κ·δ_{x₁}+(1−κ)·U(K)`` and the
        bound decomposes into per-step KLs between the closed-form true
        posterior ``q(x_s|x_τ,x₁)`` and the model posterior
        ``p_θ(x_s|x_τ)=Σ_c q(x_s|x_τ,x₁=c)·p_θ(x₁=c|x_τ)``. Estimated by
        Monte-Carlo over ``n_mc`` uniformly-sampled diffusion steps on an
        ``n_steps`` grid (the D3PM/Ho unbiased estimator; larger ``n_steps``
        tightens the discretisation gap, larger ``n_mc`` lowers variance).

        Note the *flow-matching* time convention here: κ(t=1)=1 is clean and
        κ(t=0)=0 is pure noise. The single per-step KL automatically reduces
        to the decoder term −log p_θ(x₁|x_{near-clean}) when κ_s=1 (j=0), and
        the prior term is exactly 0 because κ(0)=0 ⇒ q(·|x₁)=U. Both boundary
        terms are therefore handled by the one formula below.
        """
        K = self.cfg.text8_dataset.K
        device = token_ids.device
        x1_full = token_ids.long()
        B_full, L = x1_full.shape
        if chunk is None:
            # keep B·L·K² transient near the (B=128,L=256) bench footprint
            chunk = max(1, (128 * 256) // max(L, 1))
        eye = torch.eye(K, device=device)
        eps = 1e-8
        N = int(n_steps)
        # grid of flow-times clean(1)→noise(0); κ̃_j decreasing 1→0.
        t_grid = torch.linspace(1.0, 0.0, N + 1, device=device)
        kappa_grid = self._kappa(t_grid).clamp(0.0, 1.0)

        out = []
        for cs in range(0, B_full, chunk):
            x1 = x1_full[cs : cs + chunk]
            B = x1.shape[0]
            seq_nats = torch.zeros(B, device=device)
            for _ in range(n_mc):
                j = torch.randint(0, N, (B,), device=device)  # step in {0..N-1}
                kappa_s = kappa_grid[j].clamp(min=eps)        # keep-prob at the *less* noisy x_s
                kappa_t = kappa_grid[j + 1]                   # keep-prob at the *more* noisy x_τ (≤κ_s)
                t_tau = t_grid[j + 1]                         # flow-time fed to the denoiser at x_τ
                beta = (kappa_t / kappa_s).clamp(0.0, 1.0)    # single-step keep κ_t/κ_s

                x_tau = self._corrupt_kappa(x1, kappa_t)       # (B,L) observed more-noisy state
                p1 = self.forward(x_tau, t_tau).softmax(-1)    # (B,L,K) denoiser p_θ(x₁|x_τ)

                m_oh = eye[x_tau]                              # (B,L,K) observed token one-hot
                # single-step likelihood  q(x_τ=m | x_s=k) = β·[k=m] + (1−β)/K   over k
                q_step = beta[:, None, None] * m_oh + (1.0 - beta)[:, None, None] / K  # (B,L,K_k)
                # clean prior over x_s given x₁=c:  q(x_s=k|x₁=c)=κ_s·[k=c]+(1−κ_s)/K  → (B,K_c,K_k)
                q_clean = (
                    kappa_s[:, None, None] * eye[None]
                    + (1.0 - kappa_s)[:, None, None] / K
                )  # (B,K,K)
                # unnormalised posterior over k for each hypothesised clean c:
                #   U[b,l,c,k] = q_step[b,l,k] · q_clean[b,c,k]
                U = q_step[:, :, None, :] * q_clean[:, None, :, :]      # (B,L,K_c,K_k)
                post = U / U.sum(-1, keepdim=True).clamp(min=eps)      # q(x_s=k|x_τ,x₁=c)
                # true posterior uses the actual clean token c=x1
                q_true = post.gather(
                    2, x1[:, :, None, None].expand(B, L, 1, K)
                ).squeeze(2)                                          # (B,L,K)
                # model posterior marginalises the denoiser over c
                p_model = torch.einsum("blck,blc->blk", post, p1)      # (B,L,K)
                kl = (
                    q_true * (q_true.clamp(min=eps).log() - p_model.clamp(min=eps).log())
                ).sum(-1)                                              # (B,L) per-position term L_j (=L_0 NLL at j=0)
                seq_nats = seq_nats + N * kl.sum(-1)                   # ×N: unbiased over the N step-terms
            # prior term L_N = KL(q(x_noise|x₁)‖U) = 0 since κ(0)=0; omitted.
            out.append((seq_nats / n_mc) / (L * math.log(2)))          # bits per char
        return torch.cat(out).mean()

    @torch.no_grad()
    def denoiser_ce_bpc(
        self, token_ids: torch.Tensor, *, n_mc: int = 8
    ) -> torch.Tensor:
        """Training-objective denoiser cross-entropy in bits/char,
        ``E_t[CE(p_{1|t}(x_t), x₁)]/log2``. This is the *training loss*, NOT a
        likelihood bound — it is uniformly-t-weighted with no schedule
        derivative, so it is **not** comparable to published BPC. Kept as a
        diagnostic; use ``elbo_bpc`` for the peer-comparable number."""
        B, L = token_ids.shape
        K = self.cfg.text8_dataset.K
        total = torch.zeros((), device=token_ids.device)
        for _ in range(n_mc):
            t = torch.rand(B, device=token_ids.device)
            logits = self.forward(self._corrupt(token_ids, t), t)
            total = total + F.cross_entropy(
                logits.reshape(-1, K), token_ids.reshape(-1), reduction="mean"
            )
        return (total / n_mc) / math.log(2)

    @torch.no_grad()
    def bpd(
        self, token_ids: torch.Tensor, *, n_mc: int = 8,
        max_steps: int | None = None,
    ) -> torch.Tensor:
        """Peer-comparable bits-per-char via the variational bound (``elbo_bpc``).

        ``max_steps`` is accepted for cross-arm API compatibility with the
        iterative-recovery ``bpd`` of the EqM / SFLM families (see
        ``scripts/bench_sflm_ebm.py``) and is ignored — DFM's bound is a
        denoising ELBO, not an integration trajectory."""
        del max_steps
        return self.elbo_bpc(token_ids, n_mc=n_mc)


@register("DFM")
def build_dfm(cfg: Config) -> DiscreteFlowMatching:
    return DiscreteFlowMatching(cfg)
