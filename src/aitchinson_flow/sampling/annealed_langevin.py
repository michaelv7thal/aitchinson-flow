"""Annealed-Langevin sampler for noise-conditional score models.

NCSN-style (Song & Ermon 2019): integrate the reverse-time Langevin SDE
along a geometric σ-ladder σ_max → σ_min. At each level σ_i, run K
Langevin steps with step size α_i = ε · (σ_i / σ_min)²:

    x_{k+1} = x_k + (α_i / 2) · s_θ(x_k, σ_i) + sqrt(α_i) · ξ_k,   ξ ~ N(0, I).

The (σ_i / σ_min)² scaling keeps the per-step SNR constant across the
ladder, which is what makes the chain stable as σ → 0 (Song & Ermon §3.2).

Unconditional sampling. Initialise x ~ N(0, σ_max² I) (the score is well
approximated by −x/σ² at large σ, so the chain effectively starts as
prior samples) and run all n_sigma levels.

Recovery sampling. Initialise x at the perturbed input and start the
ladder at the noise level matching the perturbation magnitude.  Only the
σ-levels below that point are visited; the chain "anneals back down" to
the data manifold.

The score function ``s_θ(x, σ)`` is supplied by the model via a
``score(x, sigma)`` method. Both ``ScoreDSM`` and ``EqMDSM`` implement
this — the former returns the network output directly, the latter
returns ``−∇_x ⟨x, f_θ(x, σ)⟩`` (the conservative-energy gradient).
"""

from __future__ import annotations

from typing import Callable

import torch


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(x, sigma) -> score`` where sigma is broadcastable per-sample."""


def make_sigma_ladder(
    sigma_min: float, sigma_max: float, n_sigma: int, device: torch.device | None = None
) -> torch.Tensor:
    """Geometric σ-ladder σ_max ≥ … ≥ σ_min, length ``n_sigma``."""
    return torch.logspace(
        start=float(torch.log10(torch.tensor(sigma_max))),
        end=float(torch.log10(torch.tensor(sigma_min))),
        steps=int(n_sigma),
        device=device,
    )


def annealed_langevin_sample(
    score_fn: ScoreFn,
    shape: tuple[int, ...],
    *,
    sigma_min: float,
    sigma_max: float,
    n_sigma: int,
    steps_per_sigma: int,
    eps: float = 1e-5,
    x_init: torch.Tensor | None = None,
    start_sigma: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """NCSN-style annealed Langevin sampler.

    Args:
        score_fn: ``s_θ(x, σ)``; σ is a (B,) tensor.
        shape: output shape, e.g. (B, L, d_latent).
        sigma_min, sigma_max, n_sigma, steps_per_sigma, eps: schedule.
        x_init: if provided, used as the starting state (must match
            ``shape``). Otherwise initialise from ``N(0, σ_max² I)``.
        start_sigma: if provided, the σ-ladder is restricted to values
            ≤ start_sigma. Useful for recovery: perturb by σ_p, then
            anneal from σ_p down to σ_min.
        device, dtype: target for the returned tensor.
        generator: optional torch.Generator for reproducibility.

    Returns:
        Final iterate, shape ``shape``.
    """
    if device is None:
        device = (
            x_init.device if x_init is not None else torch.device("cpu")
        )
    if dtype is None:
        dtype = x_init.dtype if x_init is not None else torch.float32

    sigmas = make_sigma_ladder(sigma_min, sigma_max, n_sigma, device=device)
    if start_sigma is not None:
        sigmas = sigmas[sigmas <= start_sigma]
        if sigmas.numel() == 0:
            sigmas = torch.tensor([sigma_min], device=device, dtype=sigmas.dtype)

    if x_init is None:
        if generator is not None:
            x = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            x = torch.randn(shape, device=device, dtype=dtype)
        x = x * float(sigmas[0])
    else:
        x = x_init.detach().to(device=device, dtype=dtype).clone()

    B = shape[0]
    sigma_min_t = torch.tensor(float(sigma_min), device=device, dtype=dtype)

    for sigma_i in sigmas:
        alpha_i = eps * (sigma_i / sigma_min_t) ** 2
        sigma_b = sigma_i.expand(B).to(dtype)
        for _ in range(int(steps_per_sigma)):
            s = score_fn(x, sigma_b)
            if generator is not None:
                noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            else:
                noise = torch.randn_like(x)
            x = x + 0.5 * alpha_i * s + alpha_i.sqrt() * noise

    return x


def langevin_sample_single_sigma(
    score_fn: ScoreFn,
    x_init: torch.Tensor,
    *,
    sigma: float,
    n_steps: int,
    eps: float = 1e-5,
    sigma_min: float = 0.05,
) -> torch.Tensor:
    """Single-σ Langevin chain at a fixed noise level. Useful for
    diagnostics — fix σ and observe the equilibration of the chain on the
    noise-conditional energy landscape."""
    device = x_init.device
    dtype = x_init.dtype
    sigma_t = torch.tensor(float(sigma), device=device, dtype=dtype)
    sigma_min_t = torch.tensor(float(sigma_min), device=device, dtype=dtype)
    alpha = eps * (sigma_t / sigma_min_t) ** 2
    B = x_init.shape[0]
    sigma_b = sigma_t.expand(B)
    x = x_init.detach().clone()
    for _ in range(int(n_steps)):
        s = score_fn(x, sigma_b)
        x = x + 0.5 * alpha * s + alpha.sqrt() * torch.randn_like(x)
    return x
