"""
S-FLM-as-EBM viability probe.

Question under test: can the time-free hyperspherical flow model
(S-FLM, arXiv:2605.11125) be turned into a usable energy-based model
(EqM-style, arXiv:2510.02300) on a toy sequence task, or does it
collapse to the trivial solution E[x1] (centroid) under a deterministic
SLERP interpolant?

We sweep the three EqM anti-collapse mechanisms that the notebook
harness is missing (see eqm.py / DFM_SVGP_FINDINGS.md):

  S  alpha (noise) schedule:
       "uniform"  : alpha ~ U(alpha_lo, alpha_hi)        (toy default)
       "trunc"    : alpha ~ U(0.5, alpha_hi)             (drop pure noise)
       "import"   : alpha = alpha_hi * U(0,1)**0.5       (EqM gamma-power,
                                                          mass toward 1)
  H  contrastive energy hinge weight (lambda_hinge):
       relu(margin + E_clean - E_neg).mean()  with negatives =
       {uniform-sphere sequences, the centroid sequence (the failure
       mode itself as a hard negative)}.  This is the toy analogue of
       eqm.py:_auditor_hinge.

Primary metric (a *fair* version of notebook Diagnostic 1): the energy
of REAL structured data sequences must sit below the centroid sequence
and below random.  The notebook's original codeword diagnostic feeds
single-token-repeated sequences through a position-embedded model that
only ever saw structured "a + b = c" sequences, so it is OOD and
misleading; we keep it as a secondary readout only.

Usage:  python scripts/sflm_ebm_probe.py [--steps N] [--quick]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------
# Spherical primitives (verbatim from testing.ipynb)
# --------------------------------------------------------------------------
def normalize(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def sample_uniform_sphere(shape, device, dtype=torch.float32):
    return normalize(torch.randn(shape, device=device, dtype=dtype))


def geodesic_distance(p, q):
    dot = (p * q).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.arccos(dot)


def slerp(p, q, alpha):
    omega = geodesic_distance(p, q).unsqueeze(-1)
    sin_omega = torch.sin(omega).clamp(min=1e-7)
    a = alpha.unsqueeze(-1) if alpha.dim() == p.dim() - 1 else alpha
    coef_p = torch.sin((1.0 - a) * omega) / sin_omega
    coef_q = torch.sin(a * omega) / sin_omega
    return coef_p * p + coef_q * q


def exp_map(p, v):
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.cos(v_norm) * p + torch.sin(v_norm) * (v / v_norm)


def project_tangent(p, v):
    return v - (p * v).sum(dim=-1, keepdim=True) * p


# --------------------------------------------------------------------------
# Toy task + model (from testing.ipynb, with config knobs added)
# --------------------------------------------------------------------------
@dataclass
class ToyConfig:
    P: int = 7
    seq_len: int = 5
    embed_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    batch_size: int = 256
    n_steps: int = 3000
    lr: float = 3e-4
    tau: float = 0.1
    alpha_lo: float = 0.0
    alpha_hi: float = 0.95
    # --- new EqM-style knobs ---
    alpha_sched: str = "uniform"  # uniform | trunc | import
    lambda_hinge: float = 0.0  # contrastive energy hinge weight
    hinge_margin: float = 0.5
    ce_min_alpha: float = 0.0  # only apply CE where alpha >= this
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModularArithmeticDataset(Dataset):
    def __init__(self, cfg, n_examples):
        P = cfg.P
        self.cfg = cfg
        self.vocab_size = P + 2
        self.PLUS = P
        self.EQ = P + 1
        rng = torch.Generator().manual_seed(0)
        a = torch.randint(0, P, (n_examples,), generator=rng)
        b = torch.randint(0, P, (n_examples,), generator=rng)
        c = (a + b) % P
        plus = torch.full_like(a, self.PLUS)
        eq = torch.full_like(a, self.EQ)
        self.data = torch.stack([a, plus, b, eq, c], dim=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class TinyDenoiser(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.codebook = nn.Parameter(torch.randn(vocab_size, cfg.embed_dim) * 0.1)
        self.pos_embed = nn.Parameter(torch.zeros(cfg.seq_len, cfg.hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.in_proj = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.hidden_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)

    def codebook_normalized(self):
        return normalize(self.codebook)

    def features(self, z):
        h = self.in_proj(z) + self.pos_embed.unsqueeze(0)
        h = self.transformer(h)
        return self.out_proj(h)

    def logits(self, z):
        return (self.features(z) @ self.codebook_normalized().T) / self.cfg.tau

    def log_posterior(self, z):
        return F.log_softmax(self.logits(z), dim=-1)

    def energy(self, z, tau=None):
        h = self.features(z)
        e_hat = self.codebook_normalized()
        eff_tau = self.cfg.tau if tau is None else tau
        scores = (h @ e_hat.T) / eff_tau
        return -eff_tau * torch.logsumexp(scores, dim=-1)

    def total_energy(self, z, tau=None):
        return self.energy(z, tau=tau).sum(dim=-1)


def encode_tokens(tokens, model):
    return model.codebook_normalized()[tokens]


def sample_alpha(cfg, B, device):
    """Per-sequence alpha following the configured schedule."""
    u = torch.rand(B, 1, device=device)
    if cfg.alpha_sched == "uniform":
        return cfg.alpha_lo + (cfg.alpha_hi - cfg.alpha_lo) * u
    if cfg.alpha_sched == "trunc":
        return 0.5 + (cfg.alpha_hi - 0.5) * u
    if cfg.alpha_sched == "import":
        # EqM gamma-power: U**0.5 biases mass toward 1 (the signal regime).
        return cfg.alpha_hi * u.pow(0.5)
    raise ValueError(cfg.alpha_sched)


def hinge_loss(model, z_pos, cfg):
    """Contrastive energy hinge: data energy must sit `margin` below the
    energy of (uniform-sphere, centroid) negatives.  Toy analogue of
    eqm.py:_auditor_hinge / the DFM-SVGP OOD hinge."""
    B, L, d = z_pos.shape
    e_pos = model.total_energy(z_pos)  # (B,)
    z_unif = sample_uniform_sphere((B, L, d), z_pos.device)
    centroid = normalize(model.codebook_normalized().mean(dim=0))
    z_cent = centroid.expand(B, L, -1)
    e_neg = torch.cat(
        [model.total_energy(z_unif), model.total_energy(z_cent)]
    )  # (2B,)
    e_pos_rep = e_pos.repeat(2)
    return torch.relu(cfg.hinge_margin + e_pos_rep - e_neg).mean()


def train(cfg):
    ds = ModularArithmeticDataset(cfg, n_examples=8192)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    model = TinyDenoiser(cfg, ds.vocab_size).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step = 0
    model.train()
    while step < cfg.n_steps:
        for tokens in loader:
            tokens = tokens.to(cfg.device)
            z1 = encode_tokens(tokens, model)
            B, L, d = z1.shape
            z0 = sample_uniform_sphere(z1.shape, cfg.device)
            alpha = sample_alpha(cfg, B, cfg.device)  # (B,1)
            z_a = slerp(z0, z1, alpha.expand(B, L))

            log_p = model.log_posterior(z_a)  # (B,L,V)
            ce_mask = (alpha.squeeze(-1) >= cfg.ce_min_alpha)
            if ce_mask.any():
                lp = log_p[ce_mask].reshape(-1, log_p.shape[-1])
                ce = F.nll_loss(lp, tokens[ce_mask].reshape(-1))
            else:
                ce = torch.zeros((), device=cfg.device)
            loss = ce
            if cfg.lambda_hinge > 0.0:
                loss = loss + cfg.lambda_hinge * hinge_loss(model, z1, cfg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= cfg.n_steps:
                break
    model.eval()
    return model, ds


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
@torch.no_grad()
def energy_landscape(model, ds, cfg):
    """Fair Diagnostic 1: energy on REAL structured sequences vs centroid
    vs random vs SLERP-noised.  Plus the notebook's original (OOD)
    codeword-vs-centroid readout for continuity."""
    # Real data sequences (the proper positive set).
    idx = torch.randint(0, len(ds), (256,))
    z_data = encode_tokens(ds.data[idx].to(cfg.device), model)
    E_data = model.total_energy(z_data).mean().item()

    e_hat = model.codebook_normalized()
    centroid = normalize(e_hat.mean(dim=0))
    E_cent = model.total_energy(centroid.expand(256, cfg.seq_len, -1)).mean().item()
    E_rand = model.total_energy(
        sample_uniform_sphere((256, cfg.seq_len, cfg.embed_dim), cfg.device)
    ).mean().item()

    # Notebook's original codeword check (OOD: constant sequences).
    z_codes = e_hat.unsqueeze(1).expand(-1, cfg.seq_len, -1)
    E_codes = model.energy(z_codes).mean().item()

    return {
        "E_data": E_data, "E_centroid": E_cent, "E_random": E_rand,
        "E_codeword_ood": E_codes,
        "data<centroid": E_data < E_cent,        # PRIMARY: must be True
        "data<random": E_data < E_rand,          # PRIMARY: must be True
        "codeword<centroid_ood": E_codes < E_cent,  # notebook's metric
    }


def sample_adaptive(model, cfg, n_seqs, n_steps, target=0.1):
    z = sample_uniform_sphere((n_seqs, cfg.seq_len, cfg.embed_dim), cfg.device)
    for _ in range(n_steps):
        z = z.detach().requires_grad_(True)
        g, = torch.autograd.grad(model.total_energy(z).sum(), z)
        g = project_tangent(z.detach(), g)
        gn = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        eta = (target / gn).clamp(max=1.0)
        z = normalize(exp_map(z.detach(), -eta * g))
    return z.detach()


@torch.no_grad()
def score_samples(model, ds, cfg, z):
    preds = model.logits(z).argmax(-1)
    plus_ok = preds[:, 1] == ds.PLUS
    eq_ok = preds[:, 3] == ds.EQ
    digits_ok = (preds[:, [0, 2, 4]] < ds.cfg.P).all(dim=-1)
    structural = plus_ok & eq_ok & digits_ok
    arith = (preds[:, 0] + preds[:, 2]) % ds.cfg.P == preds[:, 4]
    # Diversity: if "stuck at E[x1]" the samples collapse to ~1 sequence.
    uniq = len({tuple(r) for r in preds.cpu().tolist()})
    # Mean per-position token entropy (nats); 0 == fully collapsed.
    ent = 0.0
    for pos in range(preds.shape[1]):
        p = torch.bincount(preds[:, pos], minlength=ds.vocab_size).float()
        p = p / p.sum()
        ent += -(p[p > 0] * p[p > 0].log()).sum().item()
    return {
        "structural": structural.float().mean().item(),
        "arith": (structural & arith).float().mean().item(),
        "unique_frac": uniq / preds.shape[0],
        "pos_entropy": ent / preds.shape[1],
    }


def recover_from_data(model, ds, cfg, n=256, alpha=0.85, n_steps=80):
    """Local-recovery test: init near real data (SLERP noise at `alpha`)
    and run the same GD.  Decouples 'is the energy basin correct?' from
    'can the global sampler traverse uniform-noise -> data?'."""
    idx = torch.randint(0, len(ds), (n,))
    z1 = encode_tokens(ds.data[idx].to(cfg.device), model)
    z0 = sample_uniform_sphere(z1.shape, cfg.device)
    a = torch.full((n, cfg.seq_len), alpha, device=cfg.device)
    z = slerp(z0, z1, a)
    for _ in range(n_steps):
        z = z.detach().requires_grad_(True)
        g, = torch.autograd.grad(model.total_energy(z).sum(), z)
        g = project_tangent(z.detach(), g)
        gn = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z = normalize(exp_map(z.detach(), -(0.1 / gn).clamp(max=1.0) * g))
    return score_samples(model, ds, cfg, z.detach())["arith"]


def run_config(name, cfg):
    torch.manual_seed(42)
    model, ds = train(cfg)
    land = energy_landscape(model, ds, cfg)
    z = sample_adaptive(model, cfg, 256, 150)
    s = score_samples(model, ds, cfg, z)
    rec = recover_from_data(model, ds, cfg)
    return {"name": name, **land, **s,
            "recover_arith": rec, "arith_chance": 1.0 / cfg.P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--quick", action="store_true",
                    help="1200 steps, fewer configs")
    args = ap.parse_args()
    steps = 1200 if args.quick else args.steps

    base = ToyConfig(n_steps=steps)
    configs = {
        # 1. reproduce the failure (notebook default)
        "baseline_uniform": replace(base),
        # 2. drop pure-noise region (alpha >= 0.5)
        "alpha_trunc": replace(base, alpha_sched="trunc"),
        # 3. EqM gamma-power importance schedule
        "import_sched": replace(base, alpha_sched="import"),
        # 4. baseline + contrastive energy hinge (the decisive term)
        "hinge_only": replace(base, lambda_hinge=1.0, hinge_margin=0.5),
        # 5. full EqM-style: importance schedule + CE-masked + hinge
        "eqm_style": replace(base, alpha_sched="import", ce_min_alpha=0.3,
                             lambda_hinge=1.0, hinge_margin=0.5),
    }
    if args.quick:
        configs = {k: configs[k] for k in
                   ["baseline_uniform", "hinge_only", "eqm_style"]}

    print(f"device={base.device}  steps={steps}  configs={list(configs)}\n")
    rows = []
    for name, cfg in configs.items():
        r = run_config(name, cfg)
        rows.append(r)
        print(
            f"[{name:18s}] E_data={r['E_data']:+.2f} "
            f"E_cent={r['E_centroid']:+.2f} | data<cent={str(r['data<centroid']):5s} "
            f"| sample arith={r['arith']:.3f} uniq={r['unique_frac']:.2f} "
            f"H={r['pos_entropy']:.2f} | recover={r['recover_arith']:.3f} "
            f"(chance {r['arith_chance']:.3f})"
        )

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("  energy basin correct?  -> data<centroid")
    print("  sampler reaches it?    -> sample arith vs chance")
    print("  basin correct locally? -> recover arith (init near data)")
    print("  stuck at E[x1]?        -> uniq~0 / pos_entropy~0")
    print("=" * 88)
    print(f"{'config':18s} {'E_data':>7s} {'E_cent':>7s} {'d<c':>5s} "
          f"{'samp_ar':>7s} {'uniq':>5s} {'H':>5s} {'recov':>6s}")
    for r in rows:
        print(f"{r['name']:18s} {r['E_data']:+7.2f} {r['E_centroid']:+7.2f} "
              f"{str(r['data<centroid']):>5s} {r['arith']:7.3f} "
              f"{r['unique_frac']:5.2f} {r['pos_entropy']:5.2f} "
              f"{r['recover_arith']:6.3f}")
    print(f"\n(arith chance = {rows[0]['arith_chance']:.3f}; "
          f"max pos_entropy = {math.log(rows[0]['arith_chance'] ** -1 + 2):.2f})")


if __name__ == "__main__":
    main()
