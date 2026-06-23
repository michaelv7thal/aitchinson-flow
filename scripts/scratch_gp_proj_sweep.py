"""Scratch: does a low-dim projection (DKL) before the one-class SVGP help OOD?

backbone features (1280) -> Linear(1280->dl) -> SVGP(regression to y=0) -> variance.
Tests learned vs frozen-random projection at several dl, vs raw 1280-d baseline.
Reuses the cached features from scratch_gp_oneclass_sweep (gp_feat_cache.pt).
Metric: held-out per-token variance AUROC (corrupt vs clean).
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import torch
import torch.nn as nn

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src")); sys.path.append(str(repo))
from aitchinson_flow.models.sparse_gp import _SparseGP  # noqa: E402
from scripts.scratch_gp_oneclass_sweep import auroc  # noqa: E402


def fit_proj_oneclass(zc, d, dl, M, ls_scale, noise_init, steps, lr, batch,
                      lambda_kl, device, seed, proj_kind):
    torch.manual_seed(seed)
    if proj_kind == "none":
        proj = None
    else:
        proj = nn.Linear(d, dl).to(device)
        if proj_kind == "random":
            for p in proj.parameters():
                p.requires_grad_(False)

    def phi(z):
        return z if proj is None else proj(z)

    gp = _SparseGP(dl, M).to(device)
    with torch.no_grad():
        gp.log_lengthscale.copy_(torch.tensor(math.log(math.sqrt(dl) * ls_scale)))
        idx = torch.randperm(zc.shape[0])[:M]
        gp.Z.copy_(phi(zc[idx]))
    gp.log_lengthscale.requires_grad_(False)
    log_noise = nn.Parameter(torch.tensor(math.log(noise_init), device=device))
    params = [p for p in gp.parameters() if p.requires_grad] + [log_noise]
    if proj is not None and proj_kind == "learned":
        params += list(proj.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    ntot = zc.shape[0]
    for _ in range(steps):
        opt.zero_grad()
        ib = torch.randint(0, ntot, (batch,), generator=g).to(device)
        o = gp(phi(zc[ib])); sig2 = log_noise.exp()
        nll = 0.5 * (math.log(2 * math.pi) + log_noise + (o.mean.pow(2) + o.variance) / sig2)
        (nll.mean() + lambda_kl * gp.kl_divergence() / ntot).backward()
        opt.step()
    gp.eval()
    return gp, proj


@torch.no_grad()
def var_auroc(gp, proj, z, yh, b, L):
    def phi(z):
        return z if proj is None else proj(z)
    Vs = []
    for i in range(0, z.shape[0], 16384):
        Vs.append(gp(phi(z[i:i + 16384])).variance.cpu())
    V = torch.cat(Vs).reshape(b, L)
    return auroc(V[yh], V[~yh])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.cache)
    zc, zh, yh, d = blob["zc"].to(device), blob["zh"].to(device), blob["yh"], blob["d"]
    bh, Lh = blob["bh"], blob["Lh"]
    print(f"[data] zc={tuple(zc.shape)} zh={tuple(zh.shape)} d={d}")

    # raw baseline (no projection)
    gp, pr = fit_proj_oneclass(zc, d, d, 512, 0.1, 0.05, args.steps, 1e-2, 4096, 0.01,
                               device, args.seed, "none")
    print(f"{'proj':>8} {'dl':>5} {'M':>4}  AUROC_var")
    print(f"{'raw':>8} {d:>5} {512:>4}  {var_auroc(gp, pr, zh, yh, bh, Lh):.4f}")
    for kind in ("learned", "random"):
        for dl in (32, 64, 128, 256):
            gp, pr = fit_proj_oneclass(zc, d, dl, 512, 0.1, 0.05, args.steps, 1e-2, 4096,
                                       0.01, device, args.seed, kind)
            au = var_auroc(gp, pr, zh, yh, bh, Lh)
            print(f"{kind:>8} {dl:>5} {512:>4}  {au:.4f}")


if __name__ == "__main__":
    main()
