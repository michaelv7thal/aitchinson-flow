"""Decisive sweep: can an UNSUPERVISED variance/density score detect corruption
on DirichletFM features, at ANY path-time t — and does the denoiser NLL work?

For each t in a sweep over [1, t_max]:
  * deterministic Dirichlet-mean features z (standardised by CLEAN fit stats)
  * Phi = (1/N) sum z z^T on clean fit tokens; Var(z)=z^T(Phi+jit I)^-1 z
  * denoiser NLL = -log softmax(forward(x_t, t))[observed token]
Reports clean-vs-corrupt AUROC for BOTH scores, at sequence and per-token
granularity, plus the ||z||^2 clean/corrupt collapse gap.

AUROC > 0.5 ⇒ score is higher for corrupt (works). < 0.5 ⇒ inverted.
If Var AUROC never clears ~0.5 at any t but NLL does, the unsupervised
variance route is dead and the denoiser-likelihood route is the fix.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scripts.ood_variance_perpos import _load_dirichletfm  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_final.pt"
model, cfg = _load_dirichletfm(ckpt, device)
K = cfg.text8_dataset.K
t_max = float(cfg.dirichlet_fm.t_max)
dm, _ = build_training_datamodule(cfg)
vl = dm.val_dataloader() or dm.train_dataloader()

seqs = []
for b in vl:
    seqs.append(b["token_ids"].long())
    if sum(s.shape[0] for s in seqs) >= 224:
        break
seqs = torch.cat(seqs)
fit_tok, eval_tok = seqs[:160], seqs[160:224]


@torch.no_grad()
def feats_nll(tok, t, chunk=16):
    """tok (B,L) -> hidden (B,L,d), nll (B,L) at path-time t (deterministic mean)."""
    hs, nlls = [], []
    for i in range(0, tok.shape[0], chunk):
        tb = tok[i:i + chunk].to(device).long()
        B, L = tb.shape
        tt = torch.full((B,), float(t), device=device)
        beta = torch.ones(B, L, K, device=device)
        beta.scatter_(-1, tb.unsqueeze(-1), float(t))
        x_t = beta / beta.sum(-1, keepdim=True)
        h = model.get_hidden_states(x_t, tt)
        logits = model.forward(x_t, tt)
        nll = -F.log_softmax(logits, dim=-1).gather(-1, tb.unsqueeze(-1)).squeeze(-1)
        hs.append(h.cpu())
        nlls.append(nll.cpu())
    return torch.cat(hs), torch.cat(nlls)


def auroc(score, label):
    if (label == 1).sum() < 2 or (label == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def corrupt(tok, scheme, rate, seed=1):
    if scheme == "replace":
        return corrupt_token_ids(tok.clone(), vocab_size=K, corrupt_rate=rate, seed=seed)
    return partially_shuffle_token_ids(tok.clone(), shuffle_rate=rate, seed=seed)


t_sweep = [1.25, 2.0, 3.0, 4.5, 6.0, 7.5]
rate = 0.3
print(f"t_max={t_max}  K={K}  fit={fit_tok.shape[0]} eval={eval_tok.shape[0]}  "
      f"corruption=replace/shuffle @ {rate}")
hdr = (f"{'t':>5} {'scheme':>8} {'|z|2_cln':>9} {'|z|2_cor':>9} "
       f"{'Var_AUseq':>10} {'Var_AUtok':>10} {'NLL_AUseq':>10} {'NLL_AUtok':>10}")

for t in t_sweep:
    hf, _ = feats_nll(fit_tok, t)
    d = hf.shape[-1]
    mu = hf.reshape(-1, d).mean(0)
    sigma = hf.reshape(-1, d).std(0).clamp_min(1e-6)
    zc = ((hf.reshape(-1, d) - mu) / sigma).double()
    Phi = zc.T @ zc / zc.shape[0]
    jit = 0.1 * Phi.trace() / d
    Lc = torch.linalg.cholesky(Phi + jit * torch.eye(d, dtype=Phi.dtype))

    def var_of(h):  # (B,L,d)->(B,L)
        B, L, _ = h.shape
        z = ((h.reshape(-1, d) - mu) / sigma).double()
        w = torch.linalg.solve_triangular(Lc, z.T, upper=False)
        return (w * w).sum(0).reshape(B, L), (z * z).sum(1).mean().item()

    hcl, ncl = feats_nll(eval_tok, t)
    vcl, n2cl = var_of(hcl)
    if t == t_sweep[0]:
        print(hdr)
    for scheme in ["replace", "shuffle"]:
        ot = corrupt(eval_tok, scheme, rate)
        hco, nco = feats_nll(ot, t)
        vco, n2co = var_of(hco)
        changed = (ot != eval_tok)
        # sequence-level: clean vs corrupt (mean over tokens)
        lab_s = np.r_[np.zeros(len(eval_tok)), np.ones(len(eval_tok))]
        var_seq = auroc(np.r_[vcl.mean(1).numpy(), vco.mean(1).numpy()], lab_s)
        nll_seq = auroc(np.r_[ncl.mean(1).numpy(), nco.mean(1).numpy()], lab_s)
        # token-level: within corrupt seqs, changed vs unchanged positions
        cm = changed.reshape(-1).numpy().astype(int)
        var_tok = auroc(vco.reshape(-1).numpy(), cm)
        nll_tok = auroc(nco.reshape(-1).numpy(), cm)
        print(f"{t:>5.2f} {scheme:>8} {n2cl:>9.1f} {n2co:>9.1f} "
              f"{var_seq:>10.3f} {var_tok:>10.3f} {nll_seq:>10.3f} {nll_tok:>10.3f}")
