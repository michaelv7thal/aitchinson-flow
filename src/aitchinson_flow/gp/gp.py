"""Sparse variational GP building blocks for Bayesian models.

Exposes predictive outputs via ``GPOutput``, a single-kernel SVGP in pure
PyTorch (Matérn 5/2, Titsias-style inducing variables), and a product-kernel
SVGP for paired token and context latents. Written for second-order autograd
and adaptive Cholesky jitter rather than tying training to an external GP
library's global settings.
"""

from __future__ import annotations
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ._svgp_algebra import svgp_kl, svgp_predictive
from ._svgp_kernels import matern52, var_L_from_raw


@dataclass
class GPOutput:
    """Predictive statistics returned by a sparse GP ``forward`` call.

    Attributes:
        mean: Predictive mean of the latent function, shape ``(B,)``.
        variance: Epistemic (diagonal) predictive variance, shape ``(B,)``.
        noise_var: Learned homoscedastic aleatoric variance as a scalar tensor
            (broadcast to the batch), or ``None`` if not used.
    """

    mean: torch.Tensor
    variance: torch.Tensor
    noise_var: torch.Tensor | None = None


class SparseGP(nn.Module):
    """Sparse variational GP from ``d``-dimensional inputs to scalar latents.

    Isotropic Matérn 5/2 covariance, linear mean, and a Gaussian variational
    distribution over ``M`` inducing variables in the same input space. The
    inducing Gram matrix is Cholesky-factored with adaptive diagonal jitter.
    """

    def __init__(self, cfg: Config) -> None:
        """Create inducing locations, variational parameters, and hyperparameters.

        Args:
            cfg: Includes latent dimension ``d_latent`` and inducing count
                ``num_inducing``.
        """
        super().__init__()
        d, M = cfg.transformer.d_latent, cfg.gp.num_inducing
        self.M = M
        self.base_jitter = 1e-3

        self.Z = nn.Parameter(torch.randn(M, cfg.transformer.d_latent) * 0.3)
        self.var_mean = nn.Parameter(torch.zeros(M))
        self.var_L_raw = nn.Parameter(torch.eye(M) * 0.1)

        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(math.sqrt(d) * 0.3)))
        self.log_outputscale = nn.Parameter(torch.tensor(math.log(2.0)))

        self.mean_linear = nn.Linear(d, 1)
        nn.init.zeros_(self.mean_linear.weight)
        nn.init.zeros_(self.mean_linear.bias)

        # Homoscedastic aleatoric noise σ²_noise = softplus(log_noise_var)
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(0.1)))

    @property
    def noise_var(self) -> torch.Tensor:
        """Scalar aleatoric variance (softplus of ``log_noise_var``)."""
        return F.softplus(self.log_noise_var)

    def _var_L(self) -> torch.Tensor:
        """Lower Cholesky factor of the variational covariance over inducing values."""
        return var_L_from_raw(self.var_L_raw, diag_floor=1e-4)

    def _kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Full Matérn 5/2 covariance between rows of ``x1`` and rows of ``x2``."""
        return matern52(x1, x2, self.log_lengthscale, self.log_outputscale)

    def forward(self, x: torch.Tensor) -> GPOutput:
        """Predictive mean, epistemic variance, and aleatoric noise at ``x``.

        Args:
            x: Batch of input vectors, shape ``(B, d)``.

        Returns:
            ``GPOutput`` with ``mean`` and ``variance`` of shape ``(B,)`` and
            ``noise_var`` the shared scalar aleatoric variance.
        """
        B = x.shape[0]
        K_ZZ = self._kernel(self.Z, self.Z)
        K_xZ = self._kernel(x, self.Z)
        K_xx = self.log_outputscale.exp().expand(B)
        mean_x = self.mean_linear(x).squeeze(-1)
        mean_Z = self.mean_linear(self.Z).squeeze(-1)
        pred_mean, pred_var = svgp_predictive(
            K_ZZ,
            K_xZ,
            K_xx,
            mean_x,
            mean_Z,
            self.var_mean,
            self._var_L(),
            self.base_jitter,
            name="SparseGP",
        )
        return GPOutput(mean=pred_mean, variance=pred_var, noise_var=self.noise_var)

    def kl_divergence(self) -> torch.Tensor:
        """Analytic KL from the variational distribution on ``u`` to the SVGP prior.

        Returns:
            Scalar tensor, the KL term for the ELBO.
        """
        mean_Z = self.mean_linear(self.Z).squeeze(-1)
        K_ZZ = self._kernel(self.Z, self.Z)
        return svgp_kl(
            self.M,
            K_ZZ,
            mean_Z,
            self.var_mean,
            self._var_L(),
            self.base_jitter,
            name="SparseGP",
        )


class ProductSparseGP(nn.Module):
    """SVGP with a product of two Matérn 5/2 kernels on token and context parts.

    Inputs are pairs ``(z, h)``; covariance is the elementwise product of a
    kernel on ``z`` and a kernel on ``h``, each with its own length scale and
    output scale. Inducing variables use paired locations ``(Z_tok, Z_ctx)``.
    Typical use: per-token latents from a projection of logits together with
    per-token context vectors from hidden states.
    """

    def __init__(self, d_tok: int, d_ctx: int, M: int, base_jitter: float = 1e-3) -> None:
        """Create inducing sites and parameters for both kernel factors.

        Args:
            d_tok: Dimension of token latents ``z``.
            d_ctx: Dimension of context latents ``h``.
            M: Number of inducing variables.
            base_jitter: Initial diagonal jitter for Cholesky of the inducing
                Gram matrix.
        """
        super().__init__()
        self.M = M
        self.base_jitter = base_jitter

        # Inducing locations for token-space latent
        self.Z_tok = nn.Parameter(torch.randn(M, d_tok) * 0.3)
        # Inducing locations for context-space latent
        self.Z_ctx = nn.Parameter(torch.randn(M, d_ctx) * 0.3)

        # Variational distribution: q(u) = N(var_mean, L_var L_varᵀ)
        self.var_mean = nn.Parameter(torch.zeros(M))
        self.var_L_raw = nn.Parameter(torch.eye(M) * 0.1)

        # Kernel hyperparameters (log-space)
        self.log_lengthscale_tok = nn.Parameter(torch.tensor(math.log(math.sqrt(d_tok) * 0.3)))
        self.log_outputscale_tok = nn.Parameter(torch.tensor(math.log(math.sqrt(2.0))))
        self.log_lengthscale_ctx = nn.Parameter(torch.tensor(math.log(math.sqrt(d_ctx) * 0.3)))
        self.log_outputscale_ctx = nn.Parameter(torch.tensor(math.log(math.sqrt(2.0))))

        # Linear mean
        self.mean_tok = nn.Linear(d_tok, 1)
        self.mean_ctx = nn.Linear(d_ctx, 1)
        nn.init.zeros_(self.mean_tok.weight)
        nn.init.zeros_(self.mean_tok.bias)
        nn.init.zeros_(self.mean_ctx.weight)
        nn.init.zeros_(self.mean_ctx.bias)

        # Homoscedastic aleatoric noise σ²_noise = softplus(log_noise_var)
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(0.1)))

    @property
    def noise_var(self) -> torch.Tensor:
        """Scalar aleatoric variance (softplus of ``log_noise_var``)."""
        return F.softplus(self.log_noise_var)

    def _var_L(self) -> torch.Tensor:
        """Lower Cholesky factor of the variational covariance over inducing values."""
        return var_L_from_raw(self.var_L_raw, diag_floor=1e-6)

    def _kernel(
        self, z1: torch.Tensor, z2: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor
    ) -> torch.Tensor:
        """Product of token and context Matérn 5/2 matrices (same row pairing)."""
        K_tok = matern52(z1, z2, self.log_lengthscale_tok, self.log_outputscale_tok)
        K_ctx = matern52(h1, h2, self.log_lengthscale_ctx, self.log_outputscale_ctx)
        return K_tok * K_ctx

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> GPOutput:
        """Predictive mean, epistemic variance, and aleatoric noise at ``(z, h)``.

        Args:
            z: Token latents, shape ``(B, d_tok)``.
            h: Context latents, shape ``(B, d_ctx)``, aligned row-wise with ``z``.

        Returns:
            ``GPOutput`` with ``mean`` and ``variance`` of shape ``(B,)`` and
            ``noise_var`` the shared scalar aleatoric variance.
        """
        B = z.shape[0]
        K_ZZ = self._kernel(self.Z_tok, self.Z_tok, self.Z_ctx, self.Z_ctx)
        K_xZ = self._kernel(z, self.Z_tok, h, self.Z_ctx)
        K_xx = (self.log_outputscale_tok.exp() * self.log_outputscale_ctx.exp()).expand(B)
        mean_x = (self.mean_tok(z) + self.mean_ctx(h)).squeeze(-1)
        mean_Z = (self.mean_tok(self.Z_tok) + self.mean_ctx(self.Z_ctx)).squeeze(-1)
        pred_mean, pred_var = svgp_predictive(
            K_ZZ,
            K_xZ,
            K_xx,
            mean_x,
            mean_Z,
            self.var_mean,
            self._var_L(),
            self.base_jitter,
            name="ProductSparseGP",
        )
        return GPOutput(mean=pred_mean, variance=pred_var, noise_var=self.noise_var)

    def kl_divergence(self) -> torch.Tensor:
        """Analytic KL from the variational distribution on ``u`` to the SVGP prior.

        Returns:
            Scalar tensor, the KL term for the ELBO.
        """
        mean_Z = (self.mean_tok(self.Z_tok) + self.mean_ctx(self.Z_ctx)).squeeze(-1)
        K_ZZ = self._kernel(self.Z_tok, self.Z_tok, self.Z_ctx, self.Z_ctx)
        return svgp_kl(
            self.M,
            K_ZZ,
            mean_Z,
            self.var_mean,
            self._var_L(),
            self.base_jitter,
            name="ProductSparseGP",
        )
