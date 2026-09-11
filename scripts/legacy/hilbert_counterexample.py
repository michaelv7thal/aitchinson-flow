"""Hilbert-works counter-example — interior-of-simplex compositional data.

Goal: show that the Hilbert (variation-norm) metric *does* outperform Euclidean
when the data lives in the *interior* of the simplex and inter-class structure
is genuinely about ratios of components — i.e., the original Aitchison
compositional-data setting that Hilbert geometry was designed for. This
distinguishes "Hilbert geometry is wrong" (overclaim) from "Hilbert geometry
is wrong for one-hot text" (the correct claim from the main capstone).

Synthetic data: a Dirichlet mixture in S_5 with three latent recipe types:
  * Type A: Dir(2, 8, 1, 1, 1) — peaked on component 2
  * Type B: Dir(1, 1, 5, 5, 1) — balanced between 3 and 4
  * Type C: Dir(3, 3, 3, 3, 3) — roughly uniform mixtures

Every sample is in the interior (xi > 0). Inter-class differences are
ratios, not vertex coordinates — exactly where Hilbert pays off.

Two FM flow models trained on this data, both small MLPs over CLR
coordinates:
  * Flow A: Euclidean (MSE) on the conservative-gradient FM target
  * Flow B: Soft-Hilbert (LSE-smoothed variation seminorm)

Headline metric: KL(p_gen || p_data) on the implied probability vector
p = softmax(CLR(z)). p_data is the analytic mixture density.

Usage:
    python scripts/hilbert_counterexample.py --out runs/hilbert_counter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap()

from aitchinson_flow.losses import SoftHilbertLoss  # noqa: E402
from aitchinson_flow.geometry import hilbert_distance  # noqa: E402


# ─── Synthetic data ─────────────────────────────────────────────────────────

DIRICHLET_TYPES = (
    torch.tensor([2.0, 8.0, 1.0, 1.0, 1.0]),
    torch.tensor([1.0, 1.0, 5.0, 5.0, 1.0]),
    torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0]),
)
TYPE_PRIORS = torch.tensor([1.0, 1.0, 1.0]) / 3.0


def sample_mixture(n: int, *, K: int = 5, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw ``n`` points from the 3-Dirichlet mixture. Returns (n, K) probability vectors."""
    z = torch.multinomial(TYPE_PRIORS, n, replacement=True, generator=generator)
    out = torch.empty(n, K)
    for k, alpha in enumerate(DIRICHLET_TYPES):
        mask = z == k
        m = int(mask.sum().item())
        if m == 0:
            continue
        out[mask] = torch.distributions.Dirichlet(alpha).sample((m,))
    return out


def mixture_log_prob(p: torch.Tensor) -> torch.Tensor:
    """Analytic log p_data(p) under the 3-Dirichlet mixture. (n,)."""
    log_terms: list[torch.Tensor] = []
    for k, alpha in enumerate(DIRICHLET_TYPES):
        d = torch.distributions.Dirichlet(alpha)
        log_terms.append(d.log_prob(p) + math.log(float(TYPE_PRIORS[k])))
    return torch.logsumexp(torch.stack(log_terms, dim=-1), dim=-1)


def to_clr(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """probability vector → CLR coordinates (zero-mean log)."""
    log_p = p.clamp(min=eps).log()
    return log_p - log_p.mean(dim=-1, keepdim=True)


def from_clr(z: torch.Tensor) -> torch.Tensor:
    """CLR → probability via softmax (which == exp/Z given zero-mean log)."""
    return F.softmax(z, dim=-1)


# ─── Flow model ─────────────────────────────────────────────────────────────


class TinyVelocity(nn.Module):
    """Small MLP velocity field f: R^K → R^K. The model output is f(x); the
    conservative-gradient FM target is ∇⟨x, f(x)⟩."""

    def __init__(self, K: int, d: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K + 1, d),  # +1 for γ time conditioning
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, K),
        )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # x: (B, K), gamma: (B,)
        h = torch.cat([x, gamma.unsqueeze(-1)], dim=-1)
        v = self.net(h)
        return v - v.mean(dim=-1, keepdim=True)  # stay in V_d


def fm_loss(
    model: TinyVelocity,
    x1: torch.Tensor,
    *,
    loss_kind: str,
    sigma: float,
    gamma_power: float,
    soft_alpha: float,
) -> torch.Tensor:
    B, K = x1.shape
    device = x1.device
    x0 = sigma * torch.randn(B, K, device=device)
    x0 = x0 - x0.mean(dim=-1, keepdim=True)
    gamma = torch.rand(B, device=device).pow(gamma_power)
    x_g = (1 - gamma[:, None]) * x0 + gamma[:, None] * x1
    x_g.requires_grad_(True)
    u_tgt = (1.0 - gamma)[:, None] * (x0 - x1)

    v = model(x_g, gamma)
    energy = (x_g * v).sum()
    grad_g = torch.autograd.grad(energy, x_g, create_graph=True)[0]

    if loss_kind == "mse":
        return F.mse_loss(grad_g, u_tgt)
    elif loss_kind == "hilbert_soft":
        # Reuse the project's SoftHilbertLoss for parity.
        loss_fn = SoftHilbertLoss(alpha=soft_alpha)
        return loss_fn(grad_g, u_tgt)
    else:
        raise ValueError(f"unknown loss_kind={loss_kind!r}")


@torch.no_grad()
def sample_nag(
    model: TinyVelocity,
    n: int,
    K: int,
    *,
    sigma: float = 0.1,
    eta: float = 0.1,
    mu: float = 0.9,
    max_steps: int = 200,
) -> torch.Tensor:
    device = next(model.parameters()).device
    x = sigma * torch.randn(n, K, device=device)
    x = x - x.mean(dim=-1, keepdim=True)

    def grad_fn(x_in: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_req = x_in.detach().requires_grad_(True)
            gamma_one = torch.ones(x_req.shape[0], device=device)
            energy = (x_req * model(x_req, gamma_one)).sum()
            return torch.autograd.grad(energy, x_req, create_graph=False)[0].detach()

    x_last = x.clone()
    g = grad_fn(x)
    for _ in range(max_steps):
        x_last = x
        x = x - eta * g
        g = grad_fn(x + mu * (x - x_last))
    return x


# ─── Evaluation ─────────────────────────────────────────────────────────────


def kl_to_data(samples_p: torch.Tensor, *, K: int, n_grid: int = 200000) -> float:
    """KL(p_gen || p_data), Monte-Carlo estimated.

    KL(q||p) ≈ E_{x~q}[log q(x) - log p(x)]. We need a density estimate of q
    (the model's output), so we use a kernel density on the simplex via a
    Dirichlet kernel. For K=5 this is tractable.

    To avoid having to fit a density, we instead report the *reverse* KL,
    KL(p_data || p_gen) is hard since we'd need q at p_data points — but
    forward KL(q||p) is easy because we have p_data analytically and just
    need samples from q.
    """
    # Forward KL(q || p): E_{x~q}[log q(x) - log p(x)].
    # Estimate log q(x) at sampled points via a leave-one-out KDE on the
    # simplex (Dirichlet kernel, concentration κ).
    n = samples_p.shape[0]
    log_p_data = mixture_log_prob(samples_p)
    # KDE: q(x) ≈ (1/(n-1)) Σ_{j ≠ i} Dir(x; κ·s_j)
    kappa = 80.0
    alpha = kappa * samples_p  # (n, K)
    # log Dirichlet density:  Σ (α-1) log x_i  +  log Γ(Σα) - Σ log Γ(α)
    log_pdf = torch.zeros(n, n)
    for i in range(n):
        # density at samples_p[i] under each kernel j
        a = alpha               # (n, K)
        x_i = samples_p[i]      # (K,)
        log_norm = (
            torch.lgamma(a.sum(dim=-1)) - torch.lgamma(a).sum(dim=-1)
        )                       # (n,)
        log_pdf[i] = log_norm + ((a - 1.0) * x_i.clamp(min=1e-8).log()).sum(dim=-1)
    # Leave-one-out: subtract diagonal contribution.
    eye = torch.eye(n, dtype=torch.bool)
    log_pdf = log_pdf.masked_fill(eye, float("-inf"))
    log_q = torch.logsumexp(log_pdf, dim=-1) - math.log(max(n - 1, 1))
    return float((log_q - log_p_data).mean())


def hilbert_kl_proxy(samples_p: torch.Tensor, ref_p: torch.Tensor) -> float:
    """Mean Hilbert distance from each sampled point to the *nearest* reference
    point. Lower = generation matches data manifold under the variation
    seminorm. A Hilbert-side "matching" metric, complementary to KL."""
    # Add tiny eps to avoid log(0); use CLR for numerics.
    z_s = to_clr(samples_p)
    z_r = to_clr(ref_p)
    # variation_norm(zs - zr) for all pairs → min over r.
    diff = z_s.unsqueeze(1) - z_r.unsqueeze(0)
    h = diff.amax(dim=-1) - diff.amin(dim=-1)
    return float(h.min(dim=-1).values.mean())


# ─── Training driver ────────────────────────────────────────────────────────


def train_one(
    loss_kind: str,
    *,
    K: int,
    n_train: int,
    n_eval: int,
    n_steps: int,
    batch: int,
    lr: float,
    sigma: float,
    soft_alpha: float,
    gamma_power: float,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    g_train = torch.Generator().manual_seed(seed + 1)
    g_eval = torch.Generator().manual_seed(seed + 2)

    train_p = sample_mixture(n_train, K=K, generator=g_train).to(device)
    eval_p = sample_mixture(n_eval, K=K, generator=g_eval).to(device)
    train_z = to_clr(train_p)

    model = TinyVelocity(K=K, d=128).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    losses: list[float] = []
    for step in range(n_steps):
        idx = torch.randint(0, n_train, (batch,), device=device)
        x1 = train_z[idx]
        loss = fm_loss(
            model, x1, loss_kind=loss_kind, sigma=sigma,
            gamma_power=gamma_power, soft_alpha=soft_alpha,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if (step + 1) % 200 == 0:
            losses.append(float(loss.detach()))

    # Sample and evaluate.
    z_gen = sample_nag(model, n_eval, K)
    p_gen = from_clr(z_gen)

    forward_kl = kl_to_data(p_gen, K=K)
    h_proxy = hilbert_kl_proxy(p_gen, eval_p)
    return {
        "loss_kind": loss_kind,
        "final_loss": losses[-1] if losses else float("nan"),
        "n_steps": n_steps,
        "forward_kl": forward_kl,
        "hilbert_match_min": h_proxy,
        "samples_first5": p_gen[:5].cpu().tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="runs/hilbert_counter")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--soft-alpha", type=float, default=2.0)
    ap.add_argument("--gamma-power", type=float, default=1.0)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows: list[dict] = []
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for kind in ("mse", "hilbert_soft"):
        for seed in seeds:
            print(f"[{kind} seed={seed}] training…")
            r = train_one(
                kind, K=args.K, n_train=args.n_train, n_eval=args.n_eval,
                n_steps=args.n_steps, batch=args.batch, lr=args.lr,
                sigma=args.sigma, soft_alpha=args.soft_alpha,
                gamma_power=args.gamma_power, seed=seed, device=device,
            )
            r["seed"] = seed
            print(f"  forward_KL={r['forward_kl']:.4f}  "
                  f"hilbert_match={r['hilbert_match_min']:.4f}  "
                  f"final_loss={r['final_loss']:.4f}")
            rows.append(r)

    # Summarise across seeds.
    summary: dict = {}
    for kind in ("mse", "hilbert_soft"):
        kls = [r["forward_kl"] for r in rows if r["loss_kind"] == kind]
        hs = [r["hilbert_match_min"] for r in rows if r["loss_kind"] == kind]
        summary[kind] = {
            "forward_kl_mean": sum(kls) / len(kls),
            "forward_kl_std": (sum((k - sum(kls) / len(kls)) ** 2 for k in kls)
                                / max(len(kls) - 1, 1)) ** 0.5,
            "hilbert_match_mean": sum(hs) / len(hs),
            "n_seeds": len(kls),
        }

    print()
    print("=== Summary ===")
    for kind, s in summary.items():
        print(f"  {kind:>14s}: KL={s['forward_kl_mean']:.4f}±{s['forward_kl_std']:.4f}  "
              f"H_match={s['hilbert_match_mean']:.4f}  n={s['n_seeds']}")

    (out / "results.json").write_text(json.dumps(
        {"summary": summary, "rows": rows, "args": vars(args)}, indent=2))
    print(f"\nWrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
