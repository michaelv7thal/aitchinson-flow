"""Peer-comparable BPC for Dirichlet FM via a variational-dequantization ELBO.

Implements docs/dirichlet_fm_bpc.md:  treat the simplex point x as a latent, the
token c as observed, and bound the discrete NLL by

    log P(c) >= E_{x~q(.|c)}[ log p_flow(x) + log p_dec(c|x) - log q(x|c) ]

with q = Dir(beta(t_max,c)), p_dec = denoiser softmax at t_max, and p_flow the
continuous-flow (CNF) density of the SAME marginal velocity field that
DirichletFM.sample() integrates. BPC = -ELBO/(L ln2) is a valid UPPER bound.

The CNF divergence is a central-finite-difference Hutchinson estimate in the
first K-1 simplex coordinates (the c_t factor is scipy-based / non-diff). Run
`--self-test` first: it gates the divergence+ODE machinery on a linear tangent
field with analytic trace, no model needed.

Usage:
    uv run python scripts/eval_dirichletfm_elbo_bpc.py --self-test
    uv run python scripts/eval_dirichletfm_elbo_bpc.py --ckpt .../DirichletFM/epoch_final.pt \
        --n 64 --nfe 60 --m-probes 2 --n-mc 1
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.distributions import Dirichlet  # noqa: E402


# ----------------------------------------------------------------------------- #
# Reduced-coordinate finite-difference Hutchinson divergence + backward CNF ODE
# ----------------------------------------------------------------------------- #
def _reduced_perturb(eps_red: torch.Tensor) -> torch.Tensor:
    """(...,K-1) reduced perturbation -> (...,K) tangent perturbation (sum 0)."""
    last = -eps_red.sum(-1, keepdim=True)
    return torch.cat([eps_red, last], dim=-1)


def fd_divergence(v_fn, x, t, m_probes, delta, generator=None):
    """E[Tr(dg/dx~)] via central-FD Hutchinson in the K-1 free coords.

    v_fn(x,t) -> (...,K) tangent velocity.  x: (...,K) on the simplex.
    Returns (...,) divergence estimate (sum over the K-1 reduced dims).
    """
    Km1 = x.shape[-1] - 1
    # Total Liouville trace per leading-batch element: a single Hutchinson probe
    # spans ALL latent dims (for a sequence: positions x coords) jointly, and
    # eps^T J eps must be summed over every non-batch dim (cross-position coupling
    # means only the total trace is meaningful). Reduce all dims except dim 0.
    red_dims = tuple(range(1, x.dim()))  # (coords,) for 2D; (positions, coords) for 3D
    acc = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    for _ in range(m_probes):
        eps_red = torch.randn(*x.shape[:-1], Km1, dtype=x.dtype,
                              device=x.device, generator=generator)
        du = _reduced_perturb(eps_red)
        vp = v_fn(x + delta * du, t)
        vm = v_fn(x - delta * du, t)
        jac_eps_red = (vp[..., :Km1] - vm[..., :Km1]) / (2.0 * delta)  # J_red eps
        acc = acc + (eps_red * jac_eps_red).sum(red_dims)
    return acc / m_probes


def cnf_logp(v_fn, x_tmax, t_max, t_min, nfe, m_probes, delta, log_prior_fn,
             generator=None, project=False):
    """log p_flow(x_tmax) by integrating the ODE backward t_max -> t_min.

    Accumulates  -∫ div(v) dt  and adds log p_prior(x_{t_min}).
    Heun (trapezoid) steps for the state; midpoint divergence per step.
    """
    x = x_tmax.clone()
    div_integral = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    ts = torch.linspace(t_max, t_min, nfe + 1, dtype=x.dtype, device=x.device)
    for i in range(nfe):
        t0 = float(ts[i])
        t1 = float(ts[i + 1])
        dt = t0 - t1  # >0
        v0 = v_fn(x, t0)
        x_pred = x - dt * v0                                   # backward Euler step
        v1 = v_fn(x_pred, t1)
        x_new = x - dt * 0.5 * (v0 + v1)                       # Heun
        # divergence at the step midpoint (trapezoid in t)
        d0 = fd_divergence(v_fn, x, t0, m_probes, delta, generator)
        d1 = fd_divergence(v_fn, x_new, t1, m_probes, delta, generator)
        div_integral = div_integral + dt * 0.5 * (d0 + d1)
        x = x_new
        if project:
            x = x.clamp(min=1e-6)
            x = x / x.sum(-1, keepdim=True)
    return log_prior_fn(x) - div_integral, x


# ----------------------------------------------------------------------------- #
# Self-test: linear tangent field with analytic divergence
# ----------------------------------------------------------------------------- #
def self_test() -> int:
    torch.manual_seed(0)
    dtype = torch.float64
    K = 6
    Km1 = K - 1
    # random reduced-linear field g(x~) = R x~ + b ; full v = [g ; -sum g]
    R = torch.randn(Km1, Km1, dtype=dtype)
    b = torch.randn(Km1, dtype=dtype)
    tr_analytic = float(R.trace())

    def v_fn(x, t):
        del t
        xr = x[..., :Km1]
        g = xr @ R.T + b
        return torch.cat([g, -g.sum(-1, keepdim=True)], dim=-1)

    # divergence at random simplex points
    x = Dirichlet(torch.ones(K, dtype=dtype)).sample((4096,))
    g = torch.Generator().manual_seed(1)
    for m, delta in [(1, 1e-3), (4, 1e-3), (16, 1e-3)]:
        div = fd_divergence(v_fn, x, 0.0, m, delta, g)
        est = float(div.mean())
        err = abs(est - tr_analytic) / (abs(tr_analytic) + 1e-9)
        print(f"  FD-Hutchinson Tr: M={m:>2} est={est:+.5f} "
              f"analytic={tr_analytic:+.5f} relerr={err:.4%}")
    # ODE divergence integral over t in [1, t_max]: should be Tr(R)*(t_max-1)
    t_max, t_min = 8.0, 1.0
    x1 = Dirichlet(torch.ones(K, dtype=dtype)).sample((2048,))
    _, _ = cnf_logp(v_fn, x1, t_max, t_min, nfe=20, m_probes=8, delta=1e-3,
                    log_prior_fn=lambda z: torch.zeros(z.shape[:-1], dtype=z.dtype),
                    generator=g)
    # recompute just the integral analytically vs estimate
    div = fd_divergence(v_fn, x1, 0.0, 16, 1e-3, g).mean()
    integral_est = float(div) * (t_max - t_min)
    integral_an = tr_analytic * (t_max - t_min)
    err = abs(integral_est - integral_an) / (abs(integral_an) + 1e-9)
    print(f"  ∫div dt over [1,8]: est={integral_est:+.4f} "
          f"analytic={integral_an:+.4f} relerr={err:.4%}")
    ok = err < 0.05
    print("  SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- #
# Real model
# ----------------------------------------------------------------------------- #
def run_model(args) -> int:
    from scripts.ood_variance_perpos import _load_dirichletfm, _perpos_feats  # noqa
    from aitchinson_flow.training import build_training_datamodule
    from aitchinson_flow.models.dirichlet_fm import _conditional_velocity_factor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    inner = model.dfm if hasattr(model, "dfm") else model
    K = cfg.text8_dataset.K
    t_max = float(cfg.dirichlet_fm.t_max)
    dtype = torch.float64

    @torch.no_grad()
    def v_fn(x, t):
        """Marginal velocity field, exactly as DirichletFM.sample() builds it."""
        xc = x.clamp(min=1e-6, max=1.0 - 1e-6).to(torch.float32)
        tb = torch.full((x.shape[0],), float(t), device=x.device, dtype=torch.float32)
        p1 = inner.forward(xc, tb).softmax(-1).to(dtype)
        c_dot = _conditional_velocity_factor(xc, t, K).to(dtype)
        denom = (1.0 - xc.to(dtype)).clamp(min=1e-6)
        w = p1 * c_dot / denom
        return w - x * w.sum(-1, keepdim=True)

    dm, _ = build_training_datamodule(cfg)
    vl = dm.val_dataloader() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.n:
            break
    ids = torch.cat(seqs)[: args.n].to(device)
    B, L = ids.shape
    print(f"[elbo-bpc] ckpt={Path(args.ckpt).name} K={K} t_max={t_max} "
          f"eval={B}x{L} nfe={args.nfe} M={args.m_probes} n_mc={args.n_mc}")
    print(f"  (peers: SEDD 1.32 / MDLM<=1.38 / D3PM-uniform 1.61; unigram {math.log2(K):.2f})")

    ones = torch.ones(K, dtype=dtype, device=device)
    log_prior = Dirichlet(ones)

    def log_prior_fn(x):  # joint over positions = sum over L of Dir(1) log-prob
        return log_prior.log_prob(x.clamp(min=1e-7)).sum(-1)

    gseed = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)  # reproducible q-samples (paired nfe-sweeps)
    per_seq = []
    torch.set_grad_enabled(False)
    chunk = args.chunk
    for cs in range(0, B, chunk):                             # batch sequences
        c = ids[cs : cs + chunk]                             # (b,L)
        b = c.shape[0]
        beta = torch.ones(b, L, K, dtype=dtype, device=device)
        beta.scatter_(-1, c.unsqueeze(-1), float(t_max))
        q = Dirichlet(beta)
        tt = torch.full((b,), t_max, device=device)
        elbo_acc = torch.zeros(b, dtype=dtype, device=device)
        for _ in range(args.n_mc):
            x = q.sample().to(dtype)                          # (b,L,K) ~ q(.|c)
            x = x.clamp(min=1e-6)
            x = x / x.sum(-1, keepdim=True)
            logq = q.log_prob(x.clamp(min=1e-7)).sum(-1)      # (b,)
            logits = inner.forward(x.to(torch.float32), tt).to(dtype)
            logpdec = logits.log_softmax(-1).gather(-1, c.unsqueeze(-1)).squeeze(-1).sum(-1)
            logpflow, _ = cnf_logp(v_fn, x, t_max, 1.0, args.nfe, args.m_probes,
                                   args.delta, log_prior_fn, gseed, project=True)
            elbo_acc = elbo_acc + (logpflow + logpdec - logq)
        bpc_chunk = -(elbo_acc / args.n_mc) / (L * math.log(2))
        per_seq.extend(bpc_chunk.tolist())
        run = torch.tensor(per_seq)
        print(f"  seq {len(per_seq):>3}/{B}  BPC running mean={run.mean():.4f} "
              f"+/- {run.std()/math.sqrt(len(run)):.4f}")

    bpc = torch.tensor(per_seq)
    print(f"\nDirichletFM ELBO-BPC = {bpc.mean():.4f} "
          f"+/- {bpc.std()/math.sqrt(len(bpc)):.4f} (sem over {B} seqs)")
    print("  = variational UPPER bound on bits/char (valid peer-comparable object).")
    print(f"  sanity: 0 < {bpc.mean():.3f} < unigram {math.log2(K):.2f}; "
          f"far above the 0.05 identity artifact.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nfe", type=int, default=60)
    ap.add_argument("--m-probes", type=int, default=2)
    ap.add_argument("--n-mc", type=int, default=1)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=16, help="# sequences per batch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.ckpt:
        ap.error("provide --ckpt or --self-test")
    return run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
