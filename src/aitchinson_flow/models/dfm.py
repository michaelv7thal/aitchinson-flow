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

    @torch.no_grad()
    def bpd(self, token_ids: torch.Tensor, *, n_mc: int = 8) -> torch.Tensor:
        """ELBO-based bits-per-character estimate (Monte Carlo over t).

        Computes E_t[-log p_{1|t}(x1 | x_t)] / log(2) as a BPD lower bound.
        Averaged over n_mc random time samples per batch item.
        """
        B, L = token_ids.shape
        K = self.cfg.text8_dataset.K
        device = token_ids.device

        total_nll = torch.tensor(0.0, device=device)
        for _ in range(n_mc):
            t = torch.rand(B, device=device)
            x_t = self._corrupt(token_ids, t)
            logits = self.forward(x_t, t)
            nll = F.cross_entropy(
                logits.reshape(-1, K), token_ids.reshape(-1), reduction="mean"
            )
            total_nll = total_nll + nll

        return (total_nll / n_mc) / math.log(2)


@register("DFM")
def build_dfm(cfg: Config) -> DiscreteFlowMatching:
    return DiscreteFlowMatching(cfg)
