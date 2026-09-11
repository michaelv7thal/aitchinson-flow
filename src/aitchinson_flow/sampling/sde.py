"""Stochastic-flow Euler-γ sampler for continuous-on-simplex flow models.

Phase R (CAPSTONE_EXPERIMENTS.md §4) — Langevin-style noise injection at
every integration step. Walks γ from 0 → 1 in ``n_steps`` Euler steps.
Update rule per step (with linear γ schedule, h = 1/n_steps):

    v   = ∇⟨x, f(x; γ)⟩          if use_grad else  f(x; γ)
    x  ← x − h·v + sqrt(2·α·h)·ξ,    ξ ~ N(0, I)

Sign matches the FM target ``c(γ)·(x0 − x1)`` which points data → noise,
so the data direction is the negative of the velocity (consistent with
``EqM.sample_euler``).

Re-centers x to the V_d subspace (zero-mean across the K axis) at every
step — same projection EqM/FMonCLR use for their deterministic Euler
sampler. The CLR features live in an unconstrained R^{K-1} embedded as
zero-mean R^K, so noise injection is in-manifold after the projection.

The α=0 path reproduces deterministic Euler and is used as the within-sweep
control cell (see Phase R sweep cell ``eqm_sde_a0p00_grad``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def sde_flow_sample(
    model: nn.Module,
    x0: torch.Tensor,
    *,
    n_steps: int,
    use_grad: bool = False,
    alpha: float = 0.0,
    time_conditioned: bool = True,
    project_zero_mean: bool = True,
    jitter_seed: int | None = None,
    grad_clip: float | None = None,
) -> torch.Tensor:
    """Stochastic Euler-γ sampler.

    Args:
        model: callable ``model(x, gamma)`` returning a velocity-shaped tensor.
        x0: (B, L, K) initial state in CLR (or any unconstrained-R^K space).
        n_steps: number of Euler steps. h = 1/n_steps.
        use_grad: if True, the velocity used for the Euler step is the
            conservative gradient ``∇_x ⟨x, model(x; γ)⟩`` — requires the
            model to support second-order autograd. If False, use raw
            ``model(x; γ)`` (the FM-style sampler). EqM-style fields with
            ``use_grad=True`` correspond to ``EqM.sample_euler(use_grad=True)``.
        alpha: Langevin diffusion coefficient. The noise scale per step is
            ``sqrt(2·α·h)`` so the SDE limit is the Langevin SDE
            ``dx = −v dt + sqrt(2α) dW``. α=0 reduces to the deterministic
            Euler integrator.
        time_conditioned: if True, the model receives a (B,) γ tensor at
            each step. EqM defaults to ``time_conditioning="off"``; pass
            False to skip the γ argument.
        project_zero_mean: re-project to V_d (zero-mean across the last
            axis) after each step. True for CLR (EqM, FMonCLR); False for
            ambient logit space (LogitKLFlow).
        jitter_seed: optional seed for the per-step Gaussian noise. None
            uses the global generator.
        grad_clip: if not None, clip the per-position L2 norm of the
            velocity ``v`` to at most ``grad_clip`` before each Euler step
            (rows whose ‖·‖ over the last axis exceed the cap are scaled
            down to it). Mirrors the NAG sampler's ``sample_grad_clip`` cold-
            start spike fix. None (default) preserves current behaviour.

    Returns:
        (B, L, K) final iterate.
    """
    device = x0.device
    dtype = x0.dtype
    B = x0.shape[0]
    h = 1.0 / float(n_steps)
    noise_scale = float((2.0 * alpha * h) ** 0.5)

    if jitter_seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(jitter_seed))
    else:
        gen = None

    x = x0.clone()
    if project_zero_mean:
        x = x - x.mean(dim=-1, keepdim=True)

    gammas = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)[:-1]

    for g in gammas:
        g_b = g.expand(B) if time_conditioned else None

        if use_grad:
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                v = model(x_req, g_b) if time_conditioned else model(x_req)
                energy = (x_req * v).sum()
                v = torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()
        else:
            v = model(x, g_b) if time_conditioned else model(x)

        if grad_clip is not None:
            n = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            v = v * (n.clamp(max=grad_clip) / n)

        x = x - h * v
        if noise_scale > 0.0:
            if gen is not None:
                xi = torch.randn(x.shape, generator=gen, device=device, dtype=dtype)
            else:
                xi = torch.randn_like(x)
            x = x + noise_scale * xi
        if project_zero_mean:
            x = x - x.mean(dim=-1, keepdim=True)

    return x
