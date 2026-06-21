"""Diagnostic: why does the BayesLinHead 'variance' anti-correlate with corruption?

Loads the DirichletFM ckpt, builds CLEAN per-token features + REPLACE-corrupt
features, forms Phi = (1/N) sum z z^T on clean tokens, and compares for clean vs
corrupt tokens:
  * ||z||              (standardised feature norm)
  * z^T (Phi+jit I)^-1 z   (the 'variance' = Mahalanobis-in-Phi the head reports)
  * energy of z in the TOP-k vs TAIL principal directions of the clean cloud

If corrupt tokens have SMALLER var + smaller tail-energy, the 'variance' is a
predictive-variance form that rewards lying in the dense core -> corrupt features
collapse inward -> anti-correlation. That is geometry, not an arithmetic bug.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from scripts.ood_variance_perpos import _load_dirichletfm, _perpos_feats  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import corrupt_token_ids  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_final.pt"
model, cfg = _load_dirichletfm(ckpt, device)
K = cfg.text8_dataset.K
t_eval = float(cfg.dfm_svgp.t_eval)
dm, _ = build_training_datamodule(cfg)
vl = dm.val_dataloader() or dm.train_dataloader()

seqs = []
for b in vl:
    seqs.append(b["token_ids"].long())
    if sum(s.shape[0] for s in seqs) >= 192:
        break
seqs = torch.cat(seqs)
fit_tok, eval_tok = seqs[:128], seqs[128:192]

fc = _perpos_feats(model, fit_tok, t_eval, device)
d = fc.shape[-1]
mu = fc.reshape(-1, d).mean(0)
sigma = fc.reshape(-1, d).std(0).clamp_min(1e-6)


def proj(h):
    return ((h.reshape(-1, d) - mu) / sigma)


zc_fit = proj(fc).double()
Phi = zc_fit.T @ zc_fit / zc_fit.shape[0]
jit = 0.1 * Phi.trace() / d
Sig_inv = Phi + jit * torch.eye(d, dtype=Phi.dtype)
L = torch.linalg.cholesky(Sig_inv)

# eigenspectrum of the clean cloud
evals, evecs = torch.linalg.eigh(Phi)          # ascending
top = evecs[:, -20:]                            # 20 dominant directions
tail = evecs[:, :-20]                           # rest


def stats(tok, corr_rate):
    ec = _perpos_feats(model, tok, t_eval, device)
    if corr_rate > 0:
        ctok = corrupt_token_ids(tok.clone(), vocab_size=K, corrupt_rate=corr_rate, seed=1)
        ec = _perpos_feats(model, ctok, t_eval, device)
        mask = (ctok != tok).reshape(-1)
    else:
        mask = torch.ones(ec.shape[0] * ec.shape[1], dtype=torch.bool)
    z = proj(ec).double()
    z = z[mask]                                  # only changed tokens when corrupt
    w = torch.linalg.solve_triangular(L, z.T, upper=False)
    var = (w * w).sum(0)                         # z^T Sig_inv^-1 z
    nrm = (z * z).sum(1)                         # ||z||^2
    top_e = (z @ top).pow(2).sum(1)
    tail_e = (z @ tail).pow(2).sum(1)
    return var.mean().item(), nrm.mean().item(), top_e.mean().item(), tail_e.mean().item()


print(f"d={d}  Phi eig: min={evals[0]:.3g} max={evals[-1]:.3g} "
      f"top20/total={evals[-20:].sum()/evals.sum():.3f}")
print(f"{'set':>14} {'mean Var':>10} {'mean||z||^2':>12} {'top20 E':>9} {'tail E':>9}")
for tag, rate in [("clean", 0.0), ("corrupt@0.3", 0.3), ("corrupt@0.5", 0.5)]:
    v, n, te, tl = stats(eval_tok, rate)
    print(f"{tag:>14} {v:>10.2f} {n:>12.2f} {te:>9.2f} {tl:>9.2f}")
