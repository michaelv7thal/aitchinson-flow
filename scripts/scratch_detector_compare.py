"""Scratch: head-to-head per-token AUROC of the three heal localizers, through
the EXACT production code paths in heal_dirichlet.py.

  linear        : discriminative linear hinge head (current heal localizer)
  gp/contrastive: discriminative SVGP energy (Matern)  -> candidate to beat linear
  gp/oneclass   : valid-only SVGP variance (the UQ channel)

Reports AUROC on a held-out corrupt set so we can pick the localizer and confirm
the UQ channel before committing to full inpaint runs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src")); sys.path.append(str(repo))

from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from scripts.heal_dirichlet import train_localizer, train_gp_localizer, _auroc  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402

CKPT = "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_continue/epoch_final.pt"
FIT, HELD, RATE, SEED = 256, 64, 0.15, 42


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
    held_corr = corrupt_token_ids(held_tok.clone(), vocab_size=K, corrupt_rate=RATE, seed=SEED + 7)
    yh = (held_corr != held_tok)
    print(f"[data] fit={FIT} held={HELD} t_eval={t_eval} corrupt_frac={yh.float().mean():.3f}")

    def au(score):
        S = score(held_corr)
        return _auroc(S[yh], S[~yh])

    print("\n=== training localizers ===")
    s_lin, _ = train_localizer(model, fit_tok, t_eval, device, train_rate=0.3,
                               margin=4.0, steps=600, lr=5e-2, seed=SEED)
    au_lin = au(s_lin)

    s_gpc, _ = train_gp_localizer(model, fit_tok, t_eval, device, mode="contrastive",
                                  train_rate=0.3, margin=4.0, steps=1500, lr=1e-2, seed=SEED,
                                  d_latent=0, num_inducing=256, margin_var=1.0, lambda_var=0.5,
                                  lambda_kl=0.01, batch=4096, var_weights=[0.0],
                                  lengthscale_scale=0.1, learn_lengthscale=False, noise_init=0.1)
    au_gpc = au(s_gpc)

    s_gp1, _ = train_gp_localizer(model, fit_tok, t_eval, device, mode="oneclass",
                                  train_rate=0.15, margin=4.0, steps=800, lr=1e-2, seed=SEED,
                                  d_latent=0, num_inducing=512, margin_var=1.0, lambda_var=0.5,
                                  lambda_kl=0.01, batch=4096, var_weights=[0.0],
                                  lengthscale_scale=0.1, learn_lengthscale=False, noise_init=0.05)
    au_gp1 = au(s_gp1)

    print("\n=== held-out per-token AUROC (substitution localization) ===")
    print(f"  linear (discriminative)      : {au_lin:.4f}")
    print(f"  gp/contrastive energy (Matern): {au_gpc:.4f}")
    print(f"  gp/oneclass variance (UQ)    : {au_gp1:.4f}")
    best = max([("linear", au_lin), ("gp-energy", au_gpc)], key=lambda r: r[1])
    print(f"\n  -> best localizer: {best[0]} ({best[1]:.4f}); "
          f"gp-energy beats linear: {au_gpc > au_lin}  (margin {au_gpc - au_lin:+.4f})")


if __name__ == "__main__":
    main()
