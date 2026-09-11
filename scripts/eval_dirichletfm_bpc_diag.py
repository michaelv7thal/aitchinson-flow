"""Honest BPC characterization of a DirichletFM (Dirichlet FM) checkpoint.

DirichletFM has NO peer-comparable likelihood: its forward process is a
continuous Dirichlet path (not a discrete D3PM ELBO like the DFM arm), so a
genuine bits-per-char bound would need a continuous-flow (prob-flow ODE +
Hutchinson trace) likelihood, which is not implemented. What IS computable is
the *denoiser cross-entropy* in bits/char, swept over the path-time t at which
x_t ~ Dir(beta(t, tokens)) is drawn:

    BPC(t) = E_x[ -log2 p_theta(x1 | x_t) ]

This is a DIAGNOSTIC, NOT a likelihood (no schedule-derivative weighting, no
normalization to a joint). It makes the landscape explicit:
  * t -> t_max (near-clean): x_t basically reveals the token -> BPC -> 0. This is
    the identity artifact that model.bpd(t_frac=0.95) reports; a <0.5-BPC value
    means "the readout saw the answer", NOT a good model.
  * t -> 1 (uniform prior): x_t carries no token info -> BPC -> unigram entropy.
Neither end, nor any single t, is comparable to published text8 BPC
(SEDD 1.32 / D3PM-uniform 1.61 / MDLM <=1.38). Use the DFM (Discrete-FM) arm's
elbo_bpc for a peer number.
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributions import Dirichlet  # noqa: E402
from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402


@torch.no_grad()
def ce_bpc_at_t(model, ids, t, K, device, n_mc, chunk=16):
    """E_x[-log2 p(x1|x_t)] with x_t ~ Dir(beta(t)), averaged over n_mc draws."""
    tot, cnt = 0.0, 0
    for i in range(0, ids.shape[0], chunk):
        tb = ids[i:i + chunk].to(device).long()
        B, L = tb.shape
        beta = torch.ones(B, L, K, device=device)
        beta.scatter_(-1, tb.unsqueeze(-1), float(t))
        tt = torch.full((B,), float(t), device=device)
        for _ in range(n_mc):
            x_t = Dirichlet(beta).sample()
            ce = F.cross_entropy(
                model.forward(x_t, tt).reshape(-1, K), tb.reshape(-1),
                reduction="sum")
            tot += float(ce)
            cnt += B * L
    return tot / cnt / math.log(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256, help="# eval sequences")
    ap.add_argument("--n-mc", type=int, default=4, help="# Dirichlet draws per t")
    ap.add_argument("--ts", type=str, default="1.01,2,3,4,5,6,7,7.6")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    inner = model.dfm if hasattr(model, "dfm") else model  # plain DirichletFM for bpd()
    K = cfg.text8_dataset.K
    t_max = float(cfg.dirichlet_fm.t_max)
    dm, _ = build_training_datamodule(cfg)
    vl = dm.val_dataloader() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.n:
            break
    ids = torch.cat(seqs)[: args.n]
    print(f"[bpc-diag] ckpt={Path(args.ckpt).name} t_max={t_max} K={K} "
          f"eval={ids.shape[0]}x{ids.shape[1]} n_mc={args.n_mc}")
    print(f"  (peers: SEDD 1.32 / MDLM <=1.38 / D3PM-uniform 1.61; "
          f"unigram floor ~log2(27)={math.log2(K):.2f})\n")

    print(f"{'t':>6} {'t/t_max':>8} {'denoiser CE BPC':>16}   regime")
    best = (None, 1e9)
    for t in [float(x) for x in args.ts.split(",")]:
        t = min(max(t, 1.0), t_max)
        bpc = ce_bpc_at_t(model, ids, t, K, device, args.n_mc)
        reg = ("~uniform prior (no token info)" if t <= 1.5 else
               "IDENTITY ARTIFACT (sees token)" if t >= 0.9 * t_max else
               "context+partial-token")
        print(f"{t:>6.2f} {t / t_max:>8.2f} {bpc:>16.4f}   {reg}")
        if bpc < best[1]:
            best = (t, bpc)

    bpd = float(inner.bpd(ids.to(device)))  # the model's own (t_frac=0.95) number
    print(f"\nmodel.bpd() [t_frac=0.95, the recorded route] = {bpd:.4f} BPC "
          f"{'<-- <0.5 identity artifact, NOT a likelihood' if bpd < 0.5 else ''}")
    print(f"min over swept t: BPC={best[1]:.4f} at t={best[0]:.2f}")
    print("\nNOTE: none of these is peer-comparable. DirichletFM has no honest "
          "BPC; for a published-comparable number train+eval the DFM arm "
          "(elbo_bpc) or implement a continuous-flow ODE likelihood.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
