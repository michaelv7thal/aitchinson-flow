"""compositional_eqm_demo.py — Minimal self-contained demo of EqM-on-text.

Shows, on a tiny K=10 / L=12 character-like task, the four chained insights:

  (1) The Dirac-FM-on-discrete-data failure mode:
      Deterministic CLR ⇒ flow_loss → 0 ⇒ ∇E ≈ 0 everywhere ⇒ sampler is a no-op.

  (2) The variance-floor fix:
      Dirichlet-thickened CLR ⇒ flow_loss plateaus at  c̄²·Tr(Σ_x₁)  ⇒
      ∇E ≠ 0 in a neighbourhood of each token ⇒ sampler does real work.

  (3) Per-γ structure of the floor:
      floor(γ) = c(γ)² · Tr(Σ_x₁) ⇒ floor(γ=1) = 0 even with thickening.
      The model can still place ∇E ≈ 0 at the data manifold (correct EBM).

  (4) Recovery / denoising as the actually-useful EqM mode:
      perturb a clean test sequence, run NAG-GD, decode, compare to source.
      Deterministic-trained model: recovery acc ≈ perturbed acc (sampler no-op).
      Dirichlet-trained model: recovery acc > perturbed acc at moderate α.

Compositional EqM = (Dirichlet-thickened CLR data) + (Hilbert/MSE loss) +
                    (NAG-GD on the conservative gradient of the energy field).

Self-contained: no project imports. Run:
    python scripts/compositional_eqm_demo.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
DEVICE = "cpu"

# ─── Setup ─────────────────────────────────────────────────────────────────
K = 10        # vocab size  ("characters"; small enough to be fast)
L = 12        # sequence length
SIGMA_SOURCE = 0.1
GRAD_LAMBDA = 1.0
N_TRAIN_STEPS = 1500
BATCH = 64


# ─── Data: simple bigram-style language ────────────────────────────────────
def make_data(n: int) -> torch.Tensor:
    """Synthetic 'language': each next token is (prev_token + offset) mod K
    with offset rotating per position. Gives a deterministic bigram pattern
    that the model has to learn the joint structure for. Shape (n, L)."""
    seqs = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        cur = torch.randint(0, K, (1,)).item()
        for l in range(L):
            seqs[i, l] = cur
            cur = (cur + 1 + (l % 3)) % K  # rotating-offset bigram
    return seqs


# ─── CLR / inverse-CLR ─────────────────────────────────────────────────────
def clr(p: torch.Tensor) -> torch.Tensor:
    """Simplex → CLR coordinates (zero-mean log-ratios)."""
    log_p = p.clamp(min=1e-12).log()
    return log_p - log_p.mean(dim=-1, keepdim=True)


# ─── Two formulations of x₁ ────────────────────────────────────────────────
def encode_deterministic(token_ids: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """One-hot + label-smoothing → CLR. The Dirac-data formulation."""
    oh = F.one_hot(token_ids, num_classes=K).float()
    p = (1 - eps) * oh + eps / K
    return clr(p)


def encode_dirichlet(
    token_ids: torch.Tensor, alpha_peak: float = 10.0, alpha_base: float = 0.1
) -> torch.Tensor:
    """Sample Dir(α_base + α_peak·e_t) → CLR. Per-call fresh draw → variance."""
    oh = F.one_hot(token_ids, num_classes=K).float()
    alpha = alpha_base + alpha_peak * oh
    gam = torch._standard_gamma(alpha)
    p = gam / gam.sum(dim=-1, keepdim=True)
    return clr(p)


# ─── Empirical Tr(Σ_x₁): measure data variance ──────────────────────────────
def measure_x1_variance(encode_fn, n_per_token: int = 500) -> float:
    """For each token id, draw n_per_token samples; estimate per-token Tr(Cov)
    and average. Tells us the variance floor scale."""
    var_per_token = []
    for t in range(K):
        tids = torch.full((n_per_token, 1), t, dtype=torch.long)
        samples = encode_fn(tids)[:, 0, :]  # (n, K)
        # Trace of covariance = sum of per-coordinate variances
        var_per_token.append(samples.var(dim=0).sum().item())
    return float(sum(var_per_token) / K)


# ─── Tiny velocity field: 2-layer MLP with γ-conditioning ──────────────────
class TinyEqMField(nn.Module):
    def __init__(self, K: int, L: int, hidden: int = 64):
        super().__init__()
        self.K = K
        self.L = L
        # γ embedded via sinusoidal then projected
        self.gamma_proj = nn.Linear(8, hidden)
        self.in_proj = nn.Linear(K, hidden)
        self.pos_emb = nn.Embedding(L, hidden)
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden, K)

    def gamma_embed(self, gamma):
        # gamma: (B,) in [0, 1] → (B, 8) sinusoidal embedding
        freqs = torch.arange(4, device=gamma.device, dtype=gamma.dtype) + 1
        ang = gamma[:, None] * freqs[None, :] * math.pi
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # x: (B, L, K), gamma: (B,)
        B, L_, K_ = x.shape
        h = self.in_proj(x)
        pos = torch.arange(L_, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        g_h = self.gamma_proj(self.gamma_embed(gamma))
        h = h + g_h.unsqueeze(1)
        h = h + self.layers(h)
        v = self.out(h)
        # Project to V_d (zero-mean across K)
        return v - v.mean(dim=-1, keepdim=True)


# ─── Loss functions ────────────────────────────────────────────────────────
def mse_loss(grad_g: torch.Tensor, u_tgt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(grad_g, u_tgt)


def hilbert_soft_loss(grad_g, u_tgt, alpha: float = 1.0) -> torch.Tensor:
    """Soft variation seminorm, smooth approximation of max−min on residual."""
    diff = grad_g - u_tgt
    return ((torch.logsumexp(alpha * diff, dim=-1)
             + torch.logsumexp(-alpha * diff, dim=-1)) / alpha).mean()


# ─── Training step: FM with conservative gradient ───────────────────────────
def fm_step(model, x1, loss_fn):
    B, L_, K_ = x1.shape
    device = x1.device
    x0 = SIGMA_SOURCE * torch.randn(B, L_, K_, device=device)
    x0 = x0 - x0.mean(dim=-1, keepdim=True)  # zero-mean across K
    gamma = torch.rand(B, device=device)
    c_gamma = (1 - gamma) / GRAD_LAMBDA
    x_g = (1 - gamma).view(B, 1, 1) * x0 + gamma.view(B, 1, 1) * x1
    x_g.requires_grad_(True)
    u_tgt = c_gamma.view(B, 1, 1) * (x0 - x1)

    f_out = model(x_g, gamma)
    energy = (x_g * f_out).sum()
    grad_g = torch.autograd.grad(energy, x_g, create_graph=True, retain_graph=True)[0]
    return loss_fn(grad_g, u_tgt)


# ─── NAG-GD sampler ─────────────────────────────────────────────────────────
@torch.no_grad()
def sample_nag(model, x_init, n_steps=200, eta=0.1, mu=0.9, gamma_at_sample=0.5):
    x = x_init.clone()
    x_last = x.clone()
    B = x.shape[0]
    gamma = torch.full((B,), gamma_at_sample, device=x.device)

    def grad_fn(x_in):
        with torch.enable_grad():
            x_req = x_in.detach().requires_grad_(True)
            energy = (x_req * model(x_req, gamma)).sum()
            return torch.autograd.grad(energy, x_req)[0].detach()

    g = grad_fn(x)
    for _ in range(n_steps):
        x_last = x
        x = x - eta * g
        g = grad_fn(x + mu * (x - x_last))
    return x


# ─── Per-γ floor diagnostic: Var(u_tgt | γ) at multiple γ values ────────────
def per_gamma_floor(encode_fn, n_per_gamma: int = 1000) -> dict[float, float]:
    """Sample (x_0, x_1) pairs at fixed γ, compute Var(c(γ)·(x_0-x_1)).
    Empirically estimates the irreducible-noise floor at each γ."""
    out = {}
    for g in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
        c_g = (1 - g) / GRAD_LAMBDA
        # Random tokens & noise
        tids = torch.randint(0, K, (n_per_gamma, 1))
        x0 = SIGMA_SOURCE * torch.randn(n_per_gamma, 1, K)
        x0 = x0 - x0.mean(dim=-1, keepdim=True)
        x1 = encode_fn(tids)
        u_tgt = c_g * (x0 - x1)
        # Trace of empirical covariance (= sum of per-coord variances)
        out[g] = float(u_tgt.var(dim=0).sum())
    return out


# ─── Train + recovery diagnostic ───────────────────────────────────────────
def run_config(name: str, encode_fn, loss_fn, train_data, test_data):
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"  Tr(Σ_x₁) ≈ {measure_x1_variance(encode_fn):.3f}    "
          f"(0 = Dirac, > 0 = thickened)")
    print(f"  per-γ MSE floor (Tr(Var(u_tgt))):")
    for g, fl in per_gamma_floor(encode_fn).items():
        print(f"    γ={g:.2f}   floor ≈ {fl:.4f}")

    model = TinyEqMField(K, L)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print("\n  Training (loss every 250 steps):")
    for step in range(N_TRAIN_STEPS):
        idx = torch.randint(0, train_data.shape[0], (BATCH,))
        x1 = encode_fn(train_data[idx])
        loss = fm_step(model, x1, loss_fn=loss_fn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"    step {step + 1:4d}: flow_loss = {float(loss):.4f}")

    # Recovery
    print("\n  Recovery test on held-out data:")
    test = test_data[:64]
    x_clean = encode_fn(test)
    embed_norm = x_clean.norm(dim=-1).mean().item()

    print(f"    embed_norm = {embed_norm:.3f}")
    print(f"    {'α':>6} {'σ_pert':>8} {'acc_pert':>10} {'acc_recov':>10} "
          f"{'Δacc':>8} {'sampler did?':>14}")

    for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00):
        sigma = alpha * embed_norm
        torch.manual_seed(123 + int(alpha * 100))
        x_pert = x_clean + sigma * torch.randn_like(x_clean)
        # Decode perturbed input
        ids_pert = x_pert.argmax(-1)
        # Sample → decode
        x_recov = sample_nag(model, x_pert, n_steps=200)
        ids_recov = x_recov.argmax(-1)
        acc_p = (ids_pert == test).float().mean().item()
        acc_r = (ids_recov == test).float().mean().item()
        verdict = ("yes" if acc_r > acc_p + 0.02
                   else "no-op" if abs(acc_r - acc_p) < 0.02
                   else "harmful")
        print(f"    {alpha:>6.2f} {sigma:>8.3f} {acc_p:>10.3f} {acc_r:>10.3f} "
              f"{acc_r - acc_p:>+8.3f} {verdict:>14}")


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    train_data = make_data(2000)
    test_data = make_data(200)

    print(f"K={K} (vocab), L={L} (seq length)")
    print(f"σ_source={SIGMA_SOURCE}, λ={GRAD_LAMBDA}, "
          f"steps={N_TRAIN_STEPS}, batch={BATCH}")

    configs = [
        ("Deterministic CLR + MSE",
         encode_deterministic, mse_loss),
        ("Deterministic CLR + Hilbert (soft, α=1)",
         encode_deterministic, hilbert_soft_loss),
        ("Dirichlet-thickened CLR + MSE",
         encode_dirichlet, mse_loss),
        ("Dirichlet-thickened CLR + Hilbert (soft, α=1)",
         encode_dirichlet, hilbert_soft_loss),
    ]
    for name, enc, loss_fn in configs:
        run_config(name, enc, loss_fn, train_data, test_data)

    print()
    print("=" * 72)
    print("Reading the table:")
    print("  Tr(Σ_x₁) > 0  ⇒  variance floor at γ < 1, sampler can do real work.")
    print("  floor(γ=1) ≈ 0  ⇒  data manifold is the energy minimum.")
    print("  Δacc > 0       ⇒  sampler genuinely denoises the perturbed input.")
    print("  Deterministic recipes show acc_recov ≈ acc_pert at all α (no-op).")
    print("  Dirichlet recipes show Δacc > 0 in the moderate-α regime.")
    print("=" * 72)


if __name__ == "__main__":
    main()
