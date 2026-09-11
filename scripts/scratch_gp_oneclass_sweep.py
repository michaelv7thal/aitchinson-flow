"""Scratch: tune the one-class SVGP localizer's variance channel.

Caches frozen DirichletFM per-token features once, then fits a one-class SVGP
(regression to y=0) at a grid of lengthscale / inducing / noise settings and
reports held-out per-token AUROC for the predictive VARIANCE (and the mean, for
reference). Cheap proxy for the full heal A/B: the localizer's per-token AUROC
is what drives loc precision/recall and therefore fix/net.
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import torch
import torch.nn as nn

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src")); sys.path.append(str(repo))

from scripts.ood_variance_perpos import _load_dirichletfm, _perpos_feats  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402
from aitchinson_flow.models.sparse_gp import _SparseGP  # noqa: E402


def auroc(pos, neg):
    pos, neg = pos.flatten().float(), neg.flatten().float()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comb = torch.cat([pos, neg]); order = comb.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, comb.numel() + 1, dtype=torch.float)
    return float((ranks[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2)
                 / (pos.numel() * neg.numel()))


def fit_oneclass(zc, dl, num_inducing, ls_scale, learn_ls, noise_init, steps, lr,
                 batch, lambda_kl, device, seed):
    gp = _SparseGP(dl, num_inducing).to(device)
    with torch.no_grad():
        gp.log_lengthscale.copy_(torch.tensor(math.log(math.sqrt(dl) * ls_scale)))
        idx = torch.randperm(zc.shape[0])[:num_inducing]
        gp.Z.copy_(zc[idx])
    gp.log_lengthscale.requires_grad_(learn_ls)
    log_noise = nn.Parameter(torch.tensor(math.log(noise_init), device=device))
    params = [p for p in gp.parameters() if p.requires_grad] + [log_noise]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    ntot = zc.shape[0]
    for step in range(steps):
        opt.zero_grad()
        ib = torch.randint(0, ntot, (batch,), generator=g).to(device)
        o = gp(zc[ib]); sig2 = log_noise.exp()
        nll = 0.5 * (math.log(2 * math.pi) + log_noise + (o.mean.pow(2) + o.variance) / sig2)
        loss = nll.mean() + lambda_kl * gp.kl_divergence() / ntot
        loss.backward(); opt.step()
    gp.eval()
    return gp


@torch.no_grad()
def eval_ev(gp, z, b, L):
    Es, Vs = [], []
    for i in range(0, z.shape[0], 16384):
        o = gp(z[i:i + 16384]); Es.append(o.mean.cpu()); Vs.append(o.variance.cpu())
    return torch.cat(Es).reshape(b, L), torch.cat(Vs).reshape(b, L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_continue/epoch_final.pt")
    ap.add_argument("--fit-seqs", type=int, default=160)
    ap.add_argument("--held-seqs", type=int, default=48)
    ap.add_argument("--corrupt-rate", type=float, default=0.15)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K
    t_eval = float(cfg.dfm_svgp.t_eval)

    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        blob = torch.load(cache)
        zc, zh, yh, d = blob["zc"], blob["zh"], blob["yh"], blob["d"]
        bh, Lh = blob["bh"], blob["Lh"]
        print(f"[cache] loaded {cache}  zc={tuple(zc.shape)} zh={tuple(zh.shape)}")
    else:
        dm, _ = build_training_datamodule(cfg)
        vl = dm.val_dataloader() or dm.train_dataloader()
        need = args.fit_seqs + args.held_seqs
        seqs = []
        for b in vl:
            seqs.append(b["token_ids"].long())
            if sum(s.shape[0] for s in seqs) >= need:
                break
        seqs = torch.cat(seqs)[:need]
        fit_tok = seqs[:args.fit_seqs]
        held_tok = seqs[args.fit_seqs:]
        fc = _perpos_feats(model, fit_tok, t_eval, device)
        d = fc.shape[-1]
        mu = fc.reshape(-1, d).mean(0); sigma = fc.reshape(-1, d).std(0).clamp_min(1e-6)
        zc = ((fc.reshape(-1, d) - mu) / sigma)
        held_corr = corrupt_token_ids(held_tok.clone(), vocab_size=K,
                                      corrupt_rate=args.corrupt_rate, seed=args.seed + 7)
        fh = _perpos_feats(model, held_corr, t_eval, device)
        bh, Lh = held_corr.shape
        zh = ((fh.reshape(-1, d) - mu) / sigma)
        yh = (held_corr != held_tok)
        if cache:
            torch.save({"zc": zc, "zh": zh, "yh": yh, "d": d, "bh": bh, "Lh": Lh}, cache)
            print(f"[cache] wrote {cache}")

    zc = zc.to(device); zh = zh.to(device)
    print(f"[data] zc={tuple(zc.shape)} zh={tuple(zh.shape)} d={d} corrupt_frac={yh.float().mean():.3f}")
    print(f"{'ls':>5} {'M':>4} {'noise':>6} {'lrnLS':>5}  {'V_clean':>8} {'ls_fit':>7}  {'AUROC_var':>9} {'AUROC_mean':>10}")
    grid = []
    for ls_scale in (0.05, 0.1, 0.15, 0.3):
        for M in (256, 512):
            for noise in (0.05, 0.2):
                gp = fit_oneclass(zc, d, M, ls_scale, False, noise, args.steps, 1e-2,
                                  4096, 0.01, device, args.seed)
                Eh, Vh = eval_ev(gp, zh, bh, Lh)
                with torch.no_grad():
                    oc = gp(zc[:8192]); vcl = oc.variance.mean().item()
                au_v = auroc(Vh[yh], Vh[~yh]); au_m = auroc(Eh[yh], Eh[~yh])
                lsf = gp.log_lengthscale.exp().item()
                print(f"{ls_scale:>5.2f} {M:>4} {noise:>6.2f} {'F':>5}  {vcl:>8.3f} {lsf:>7.2f}"
                      f"  {au_v:>9.4f} {au_m:>10.4f}")
                grid.append((ls_scale, M, noise, au_v, au_m))
    best = max(grid, key=lambda r: r[3])
    print(f"\nBEST variance-AUROC: ls_scale={best[0]} M={best[1]} noise={best[2]} "
          f"-> AUROC_var={best[3]:.4f} (mean={best[4]:.4f})")


if __name__ == "__main__":
    main()
