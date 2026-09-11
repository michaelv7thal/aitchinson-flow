"""Pure-PyTorch sparse variational Gaussian process modules.

``_SparseGP`` — Titsias 2009 / Hensman 2015 non-whitened SVGP, Matern-5/2 kernel,
written in eager PyTorch so the GP posterior mean can be differentiated *twice*
(create_graph=True) when used inside an EBM velocity field. GPyTorch's lazy
KroneckerProductLazyTensor / lazy linear algebra path is not consistently
compatible with second-order autograd; this implementation sidesteps that.

``_ProductSparseGP`` — same SVGP but with a product kernel
``K((z, h), (z', h')) = K_tok(z, z') · K_ctx(h, h')``. Used for the
context-conditioned auditor variants (per-token + external feature h).

Design notes (from a comment in the upstream design):
  • Inducing-point init ~ N(0, 0.3²) — matches the latent-head output scale.
  • Lengthscale init = sqrt(d) · 0.3 → kernel saturates around unit distance.
  • Outputscale init = 2.0 → exceeds the margin in the contrastive hinge, so
    the OOD hinge is informative at random init.
  • Adaptive Cholesky jitter (1e-3, 1e-2, 1e-1, 1.0) avoids gpytorch's
    settings context manager.
  • Non-whitened q(u) = N(var_mean, L_var L_varᵀ) — var_mean gradients
    decouple from K_ZZ, which empirically stabilises training.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class _GPOutput:
    """Predictive mean and diagonal variance returned by SVGP modules."""

    mean: torch.Tensor   # (B,)
    variance: torch.Tensor  # (B,)


class _SparseGP(nn.Module):
    """Pure-PyTorch SVGP: $\\mathbb{R}^d \\to \\mathbb{R}$.

    Kernel:   ``outputscale · Matern-5/2`` (matmul-based; second-order differentiable)
    Mean:     ``LinearMean  m(x) = w · x + b``
    q(u):     ``N(var_mean, L_var L_varᵀ)`` non-whitened
    """

    def __init__(self, d_latent: int, num_inducing: int, base_jitter: float = 1e-3):
        super().__init__()
        d = int(d_latent)
        M = int(num_inducing)
        self.M = M
        self.base_jitter = float(base_jitter)

        self.Z = nn.Parameter(torch.randn(M, d) * 0.3)

        self.var_mean = nn.Parameter(torch.zeros(M))
        self.var_L_raw = nn.Parameter(torch.eye(M) * 0.1)

        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(math.sqrt(d) * 0.3))
        )
        self.log_outputscale = nn.Parameter(torch.tensor(math.log(2.0)))

        self.mean_linear = nn.Linear(d, 1)
        nn.init.zeros_(self.mean_linear.weight)
        nn.init.zeros_(self.mean_linear.bias)

    def _var_L(self) -> torch.Tensor:
        L = self.var_L_raw.tril(-1)
        diag_pos = F.softplus(self.var_L_raw.diagonal()) + 1e-6
        return L + torch.diag(diag_pos)

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ls = self.log_lengthscale.exp()
        os = self.log_outputscale.exp()
        x1_ = x1 / ls
        x2_ = x2 / ls
        sq = (
            (x1_ * x1_).sum(-1, keepdim=True)
            + (x2_ * x2_).sum(-1, keepdim=True).T
            - 2.0 * x1_ @ x2_.T
        ).clamp_min(0.0)
        dist = (sq + 1e-8).sqrt()
        r5 = 5.0**0.5
        return os * (1.0 + r5 * dist + (5.0 / 3.0) * sq) * torch.exp(-r5 * dist)

    def _cholesky(self, K: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        for scale in (1.0, 10.0, 100.0, 1000.0):
            try:
                return torch.linalg.cholesky(K + (self.base_jitter * scale) * eye)
            except torch.linalg.LinAlgError:
                continue
        raise RuntimeError("_SparseGP: K + jitter still not positive definite")

    def forward(self, x: torch.Tensor) -> _GPOutput:
        B = x.shape[0]
        K_ZZ = self._kernel(self.Z, self.Z)
        K_xZ = self._kernel(x, self.Z)
        k_xx = self.log_outputscale.exp().expand(B)

        mean_x = self.mean_linear(x).squeeze(-1)
        mean_Z = self.mean_linear(self.Z).squeeze(-1)

        L_ZZ = self._cholesky(K_ZZ)
        L_var = self._var_L()

        diff = (self.var_mean - mean_Z).unsqueeze(-1)
        v = torch.linalg.solve_triangular(L_ZZ, diff, upper=False)
        alpha = torch.linalg.solve_triangular(L_ZZ.T, v, upper=True).squeeze(-1)

        pred_mean = mean_x + K_xZ @ alpha

        A = torch.linalg.solve_triangular(L_ZZ, K_xZ.T, upper=False)
        C = torch.linalg.solve_triangular(L_ZZ, L_var, upper=False)

        pred_var = (
            k_xx - (A * A).sum(0) + (C.T @ A).pow(2).sum(0)
        ).clamp_min(1e-6)

        return _GPOutput(mean=pred_mean, variance=pred_var)

    def kl_divergence(self) -> torch.Tensor:
        mean_Z = self.mean_linear(self.Z).squeeze(-1)
        K_ZZ = self._kernel(self.Z, self.Z)
        L_ZZ = self._cholesky(K_ZZ)
        L_var = self._var_L()
        C = torch.linalg.solve_triangular(L_ZZ, L_var, upper=False)
        v = torch.linalg.solve_triangular(
            L_ZZ, (self.var_mean - mean_Z).unsqueeze(-1), upper=False
        ).squeeze(-1)
        log_det_K = 2.0 * L_ZZ.diagonal().log().sum()
        log_det_S = 2.0 * L_var.diagonal().log().sum()
        return 0.5 * (
            (C * C).sum() + (v * v).sum() - self.M + log_det_K - log_det_S
        )


class _ProductSparseGP(nn.Module):
    """SVGP with product kernel ``K((z,h),(z',h')) = K_tok(z,z') · K_ctx(h,h')``.

    Both factors are Matern-5/2 with independent hyperparameters. The product of
    two PSD kernels is PSD, so all Titsias-2009 math applies unchanged; K_ZZ
    and K_xZ are the Hadamard products of the two single-factor matrices
    evaluated at shared inducing locations ``(Z_tok, Z_ctx)``.

    Used by the context-conditioned (per-token) auditor variant where
        z — per-token latent from the model's own encoder
        h — per-token external context (e.g. top-K next-token logits, GPT-2
            last-layer hidden state, or any side-channel feature)

    Initialised so each factor's outputscale = sqrt(2.0), giving a combined
    outputscale ≈ 2.0 at zero distance (matches ``_SparseGP`` default and
    keeps the contrastive hinge informative).
    """

    def __init__(
        self,
        d_tok: int,
        d_ctx: int,
        num_inducing: int,
        base_jitter: float = 1e-3,
    ):
        super().__init__()
        self.M = int(num_inducing)
        self.base_jitter = float(base_jitter)

        self.Z_tok = nn.Parameter(torch.randn(self.M, d_tok) * 0.3)
        self.Z_ctx = nn.Parameter(torch.randn(self.M, d_ctx) * 0.3)

        self.var_mean = nn.Parameter(torch.zeros(self.M))
        self.var_L_raw = nn.Parameter(torch.eye(self.M) * 0.1)

        self.log_lengthscale_tok = nn.Parameter(
            torch.tensor(math.log(math.sqrt(d_tok) * 0.3))
        )
        self.log_outputscale_tok = nn.Parameter(torch.tensor(math.log(math.sqrt(2.0))))
        self.log_lengthscale_ctx = nn.Parameter(
            torch.tensor(math.log(math.sqrt(d_ctx) * 0.3))
        )
        self.log_outputscale_ctx = nn.Parameter(torch.tensor(math.log(math.sqrt(2.0))))

        self.mean_tok = nn.Linear(d_tok, 1)
        self.mean_ctx = nn.Linear(d_ctx, 1)
        nn.init.zeros_(self.mean_tok.weight)
        nn.init.zeros_(self.mean_tok.bias)
        nn.init.zeros_(self.mean_ctx.weight)
        nn.init.zeros_(self.mean_ctx.bias)

    def _var_L(self) -> torch.Tensor:
        L = self.var_L_raw.tril(-1)
        diag_pos = F.softplus(self.var_L_raw.diagonal()) + 1e-6
        return L + torch.diag(diag_pos)

    @staticmethod
    def _matern52(
        x1: torch.Tensor,
        x2: torch.Tensor,
        log_ls: torch.Tensor,
        log_os: torch.Tensor,
    ) -> torch.Tensor:
        ls = log_ls.exp()
        os = log_os.exp()
        x1_ = x1 / ls
        x2_ = x2 / ls
        sq = (
            (x1_ * x1_).sum(-1, keepdim=True)
            + (x2_ * x2_).sum(-1, keepdim=True).T
            - 2.0 * x1_ @ x2_.T
        ).clamp_min(0.0)
        dist = (sq + 1e-8).sqrt()
        r5 = 5.0**0.5
        return os * (1.0 + r5 * dist + (5.0 / 3.0) * sq) * torch.exp(-r5 * dist)

    def _kernel(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> torch.Tensor:
        K_tok = self._matern52(
            z1, z2, self.log_lengthscale_tok, self.log_outputscale_tok
        )
        K_ctx = self._matern52(
            h1, h2, self.log_lengthscale_ctx, self.log_outputscale_ctx
        )
        return K_tok * K_ctx

    def _cholesky(self, K: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        for scale in (1.0, 10.0, 100.0, 1000.0):
            try:
                return torch.linalg.cholesky(K + (self.base_jitter * scale) * eye)
            except torch.linalg.LinAlgError:
                continue
        raise RuntimeError("_ProductSparseGP: K + jitter still not positive definite")

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> _GPOutput:
        B = z.shape[0]
        K_ZZ = self._kernel(self.Z_tok, self.Z_tok, self.Z_ctx, self.Z_ctx)
        K_xZ = self._kernel(z, self.Z_tok, h, self.Z_ctx)
        k_xx = (
            self.log_outputscale_tok.exp() * self.log_outputscale_ctx.exp()
        ).expand(B)

        mean_x = (self.mean_tok(z) + self.mean_ctx(h)).squeeze(-1)
        mean_Z = (self.mean_tok(self.Z_tok) + self.mean_ctx(self.Z_ctx)).squeeze(-1)

        L_ZZ = self._cholesky(K_ZZ)
        L_var = self._var_L()
        diff = (self.var_mean - mean_Z).unsqueeze(-1)
        v = torch.linalg.solve_triangular(L_ZZ, diff, upper=False)
        alpha = torch.linalg.solve_triangular(L_ZZ.T, v, upper=True).squeeze(-1)

        pred_mean = mean_x + K_xZ @ alpha

        A = torch.linalg.solve_triangular(L_ZZ, K_xZ.T, upper=False)
        C = torch.linalg.solve_triangular(L_ZZ, L_var, upper=False)
        pred_var = (
            k_xx - (A * A).sum(0) + (C.T @ A).pow(2).sum(0)
        ).clamp_min(1e-6)

        return _GPOutput(mean=pred_mean, variance=pred_var)

    def kl_divergence(self) -> torch.Tensor:
        mean_Z = (
            self.mean_tok(self.Z_tok) + self.mean_ctx(self.Z_ctx)
        ).squeeze(-1)
        K_ZZ = self._kernel(self.Z_tok, self.Z_tok, self.Z_ctx, self.Z_ctx)
        L_ZZ = self._cholesky(K_ZZ)
        L_var = self._var_L()
        C = torch.linalg.solve_triangular(L_ZZ, L_var, upper=False)
        v = torch.linalg.solve_triangular(
            L_ZZ, (self.var_mean - mean_Z).unsqueeze(-1), upper=False
        ).squeeze(-1)
        log_det_K = 2.0 * L_ZZ.diagonal().log().sum()
        log_det_S = 2.0 * L_var.diagonal().log().sum()
        return 0.5 * (
            (C * C).sum() + (v * v).sum() - self.M + log_det_K - log_det_S
        )


__all__ = ["_GPOutput", "_SparseGP", "_ProductSparseGP"]
