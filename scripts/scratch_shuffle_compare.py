"""Scratch: non-AR detector stack on the SUBSTITUTION vs SHUFFLE axes.

The thesis-relevant question: substitution is near-linearly separable in
DirichletFM feature space (linear head ~= GP), so the GP only earns the UQ
channel. SHUFFLE (order corruption) is non-local — the token is valid, only its
context is wrong — and may be non-linearly separable, where the GP energy could
beat the linear head on DETECTION. Each detector is trained with MATCHED
negatives (the same corruption type it's tested on); one-class is valid-only.

Detectors (all non-autoregressive, on frozen DirichletFM features):
  linear        : discriminative linear hinge head
  gp_energy     : discriminative SVGP energy (Matern)
  oneclass_var  : valid-only SVGP variance
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import torch
import torch.nn as nn

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src")); sys.path.append(str(repo))
from scripts.ood_variance_perpos import _load_dirichletfm, _perpos_feats  # noqa: E402
from scripts.scratch_gp_oneclass_sweep import auroc, fit_oneclass, eval_ev  # noqa: E402
from aitchinson_flow.models.sparse_gp import _SparseGP  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids, partially_shuffle_token_ids,
)

CKPT = "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_continue/epoch_final.pt"
FIT, HELD, SEED = 256, 64, 42


def make_corrupt(kind):
    if kind == "substitution":
        return lambda tok, K, rate, seed: corrupt_token_ids(
            tok.clone(), vocab_size=K, corrupt_rate=rate, seed=seed)
    return lambda tok, K, rate, seed: partially_shuffle_token_ids(
        tok.clone(), shuffle_rate=rate, seed=seed)


def train_linear(zpos, zneg, d, device, steps=600, lr=5e-2, margin=4.0):
    head = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = head(zpos).squeeze(-1).pow(2).mean() + \
            torch.relu(margin - head(zneg).squeeze(-1)).mean()
        loss.backward(); opt.step()
    head.eval()
    return lambda z: head(z).squeeze(-1).detach().cpu()


def train_gp_energy(zpos, zneg, dl, device, M=256, steps=1500, lr=1e-2, batch=4096,
                    ls_scale=0.1, margin=4.0, seed=SEED):
    gp = _SparseGP(dl, M).to(device)
    with torch.no_grad():
        gp.log_lengthscale.copy_(torch.tensor(math.log(math.sqrt(dl) * ls_scale)))
        gp.Z.copy_(zpos[torch.randperm(zpos.shape[0])[:M]])
    gp.log_lengthscale.requires_grad_(False)
    opt = torch.optim.Adam([p for p in gp.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    npos, nneg = zpos.shape[0], zneg.shape[0]
    for _ in range(steps):
        opt.zero_grad()
        dp = gp(zpos[torch.randint(0, npos, (batch,), generator=g).to(device)])
        dn = gp(zneg[torch.randint(0, nneg, (batch,), generator=g).to(device)])
        loss = dp.mean.pow(2).mean() + torch.relu(margin - dn.mean).mean() \
            + 0.01 * gp.kl_divergence() / batch
        loss.backward(); opt.step()
    gp.eval()

    @torch.no_grad()
    def score(z):
        out = []
        for i in range(0, z.shape[0], 16384):
            out.append(gp(z[i:i + 16384]).mean.cpu())
        return torch.cat(out)
    return score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    model, cfg = _load_dirichletfm(CKPT, device)
    K = cfg.text8_dataset.K
    t_eval = float(cfg.dfm_svgp.t_eval)
    dm, _ = build_training_datamodule(cfg)
    vl = dm.val_dataloader() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= FIT + HELD:
            break
    seqs = torch.cat(seqs)[:FIT + HELD]
    fit_tok, held_tok = seqs[:FIT], seqs[FIT:]

    fc = _perpos_feats(model, fit_tok, t_eval, device)
    d = fc.shape[-1]
    mu = fc.reshape(-1, d).mean(0); sg = fc.reshape(-1, d).std(0).clamp_min(1e-6)
    std = lambda h: ((h - mu) / sg)
    zc = std(fc.reshape(-1, d)).to(device)

    # one-class GP variance (corruption-agnostic; trained on clean only)
    gp1 = fit_oneclass(zc, d, 512, 0.1, False, 0.05, 800, 1e-2, 4096, 0.01, device, SEED)

    print(f"[data] fit={FIT} held={HELD} t_eval={t_eval} d={d}")
    print(f"\n{'axis':>13} {'linear':>8} {'gp_energy':>10} {'oneclass_var':>13}")
    for kind in ("substitution", "shuffle"):
        cf = make_corrupt(kind)
        # discriminative training negatives (matched corruption, rate 0.3)
        corr_fit = cf(fit_tok, K, 0.3, SEED)
        fk = _perpos_feats(model, corr_fit, t_eval, device)
        ylab = (corr_fit != fit_tok).reshape(-1)
        zk = std(fk.reshape(-1, d)).to(device)
        zpos = torch.cat([zc, zk[~ylab.to(device)]], 0)
        zneg = zk[ylab.to(device)]
        s_lin = train_linear(zpos, zneg, d, device)
        s_gpe = train_gp_energy(zpos, zneg, d, device, seed=SEED)
        # held eval (rate 0.15)
        corr_held = cf(held_tok, K, 0.15, SEED + 7)
        fh = _perpos_feats(model, corr_held, t_eval, device)
        bh, Lh = corr_held.shape
        zh = std(fh.reshape(-1, d)).to(device)
        yh = (corr_held != held_tok)
        S_lin = s_lin(zh).reshape(bh, Lh)
        S_gpe = s_gpe(zh).reshape(bh, Lh)
        _, Vh = eval_ev(gp1, zh, bh, Lh)
        au_lin = auroc(S_lin[yh], S_lin[~yh])
        au_gpe = auroc(S_gpe[yh], S_gpe[~yh])
        au_var = auroc(Vh[yh], Vh[~yh])
        print(f"{kind:>13} {au_lin:>8.4f} {au_gpe:>10.4f} {au_var:>13.4f}"
              f"   (corrupt_frac={yh.float().mean():.3f})")


if __name__ == "__main__":
    main()
