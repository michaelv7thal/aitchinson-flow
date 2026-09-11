"""Scratch: does t_eval lift the one-class SVGP variance OOD signal?

Re-extracts frozen DirichletFM per-token features at several path-times t_eval,
fits the best one-class SVGP config (small fixed lengthscale), and reports
held-out per-token variance AUROC. Tests whether a better feature-readout time
can close the gap to the discriminative linear head (~0.85).
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import torch
import torch.nn as nn

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src")); sys.path.append(str(repo))

from scripts.ood_variance_perpos import _load_dirichletfm, _perpos_feats  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402
from aitchinson_flow.models.sparse_gp import _SparseGP  # noqa: E402
from scripts.scratch_gp_oneclass_sweep import auroc, fit_oneclass, eval_ev  # noqa: E402

CKPT = "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_continue/epoch_final.pt"
FIT, HELD, RATE, SEED = 160, 48, 0.15, 42


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    model, cfg = _load_dirichletfm(CKPT, device)
    K = cfg.text8_dataset.K
    t_max = float(model.dfm.t_max)

    dm, _ = build_training_datamodule(cfg)
    vl = dm.val_dataloader() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= FIT + HELD:
            break
    seqs = torch.cat(seqs)[:FIT + HELD]
    fit_tok, held_tok = seqs[:FIT], seqs[FIT:]
    held_corr = corrupt_token_ids(held_tok.clone(), vocab_size=K, corrupt_rate=RATE, seed=SEED + 7)
    yh = (held_corr != held_tok)

    print(f"t_max={t_max}  corrupt_frac={yh.float().mean():.3f}")
    print(f"{'t_eval':>7} {'V_clean':>8}  {'AUROC_var':>9} {'AUROC_mean':>10}")
    results = []
    for t_eval in (1.0, 2.0, 3.0, 4.5, 6.0, 7.0, 7.8):
        fc = _perpos_feats(model, fit_tok, t_eval, device)
        d = fc.shape[-1]
        mu = fc.reshape(-1, d).mean(0); sg = fc.reshape(-1, d).std(0).clamp_min(1e-6)
        zc = ((fc.reshape(-1, d) - mu) / sg).to(device)
        fh = _perpos_feats(model, held_corr, t_eval, device)
        bh, Lh = held_corr.shape
        zh = ((fh.reshape(-1, d) - mu) / sg).to(device)
        gp = fit_oneclass(zc, d, 512, 0.1, False, 0.05, 800, 1e-2, 4096, 0.01, device, SEED)
        Eh, Vh = eval_ev(gp, zh, bh, Lh)
        with torch.no_grad():
            vcl = gp(zc[:8192]).variance.mean().item()
        au_v = auroc(Vh[yh], Vh[~yh]); au_m = auroc(Eh[yh], Eh[~yh])
        print(f"{t_eval:>7.1f} {vcl:>8.3f}  {au_v:>9.4f} {au_m:>10.4f}")
        results.append((t_eval, au_v, au_m))
    best = max(results, key=lambda r: r[1])
    print(f"\nBEST: t_eval={best[0]} -> AUROC_var={best[1]:.4f} (mean={best[2]:.4f})")


if __name__ == "__main__":
    main()
