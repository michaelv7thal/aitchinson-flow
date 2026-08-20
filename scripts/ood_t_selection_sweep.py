#!/usr/bin/env python3
"""Re-run and RECORD the path-time selection sweep behind the paper's detectors.

appendix tab:config states the read-out times (NLL t=3.0, variance/BGMM t=7.5)
were "selected on the validation split by a sweep over t in {1.25, 2, 3, 4.5,
6, 7.5} ... and then held fixed for the test split". The original sweep
(scripts/legacy/diag_var_tsweep.py) printed a table and wrote nothing, and ran
on a checkpoint that is not the pinned model. This script re-runs the same
design on the pinned checkpoint and writes the table to a JSON artifact so the
selection procedure is checkable (cleanup plan item F1, repro-audit gap R4).

    .venv/bin/python scripts/ood_t_selection_sweep.py \
        [--ckpt runs/.../DirichletFM_converge/epoch_final.pt] \
        [--out bench_ood_final/tsweep/t_selection.json]

Design, unchanged from the legacy sweep: validation split, first 160 sequences
fit the standardisation and the Laplace precision, the next 64 are scored;
replace and shuffle corruption at rate 0.3, corruption seed 1; deterministic
Dirichlet-mean features; unsupervised variance score z^T(Phi+jit I)^{-1}z and
the denoiser NLL, AUROC at sequence and token granularity. GPU, ~minutes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.append(str(_ROOT))

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


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _auroc(score, label) -> float:
    if (label == 1).sum() < 2 or (label == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(_ROOT / "runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_converge/epoch_final.pt"))
    ap.add_argument("--out", type=Path, default=_ROOT / "bench_ood_final/tsweep/t_selection.json")
    ap.add_argument("--t-grid", default="1.25,2,3,4.5,6,7.5")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--fit-seqs", type=int, default=160)
    ap.add_argument("--eval-seqs", type=int, default=64)
    ap.add_argument("--corrupt-seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K
    dm, _ = build_training_datamodule(cfg)
    vl = dm.val_dataloader() or dm.train_dataloader()

    n_need = args.fit_seqs + args.eval_seqs
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= n_need:
            break
    seqs = torch.cat(seqs)[:n_need]
    fit_tok, eval_tok = seqs[: args.fit_seqs], seqs[args.fit_seqs:]

    @torch.no_grad()
    def feats_nll(tok, t):
        hs, nlls = [], []
        for i in range(0, tok.shape[0], args.chunk):
            tb = tok[i:i + args.chunk].to(device).long()
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

    def corrupt(tok, scheme):
        if scheme == "replace":
            return corrupt_token_ids(tok.clone(), vocab_size=K,
                                     corrupt_rate=args.rate, seed=args.corrupt_seed)
        return partially_shuffle_token_ids(tok.clone(), shuffle_rate=args.rate,
                                           seed=args.corrupt_seed)

    t_grid = [float(t) for t in args.t_grid.split(",")]
    rows = []
    for t in t_grid:
        hf, _ = feats_nll(fit_tok, t)
        d = hf.shape[-1]
        flat = hf.reshape(-1, d)
        mu, sigma = flat.mean(0), flat.std(0).clamp_min(1e-6)
        zc = ((flat - mu) / sigma).double()
        Phi = zc.T @ zc / zc.shape[0]
        jit = 0.1 * Phi.trace() / d
        Lc = torch.linalg.cholesky(Phi + jit * torch.eye(d, dtype=Phi.dtype))

        def var_of(h):
            B, L, _ = h.shape
            z = ((h.reshape(-1, d) - mu) / sigma).double()
            w = torch.linalg.solve_triangular(Lc, z.T, upper=False)
            return (w * w).sum(0).reshape(B, L)

        hcl, ncl = feats_nll(eval_tok, t)
        vcl = var_of(hcl)
        for scheme in ("replace", "shuffle"):
            ot = corrupt(eval_tok, scheme)
            hco, nco = feats_nll(ot, t)
            vco = var_of(hco)
            changed = (ot != eval_tok).reshape(-1).numpy().astype(int)
            lab_s = np.r_[np.zeros(len(eval_tok)), np.ones(len(eval_tok))]
            rows.append({
                "t": t, "scheme": scheme,
                "var_auroc_seq": _auroc(np.r_[vcl.mean(1).numpy(), vco.mean(1).numpy()], lab_s),
                "var_auroc_tok": _auroc(vco.reshape(-1).numpy(), changed),
                "nll_auroc_seq": _auroc(np.r_[ncl.mean(1).numpy(), nco.mean(1).numpy()], lab_s),
                "nll_auroc_tok": _auroc(nco.reshape(-1).numpy(), changed),
            })
            print(f"t={t:>5.2f} {scheme:>8} "
                  f"var seq/tok {rows[-1]['var_auroc_seq']:.3f}/{rows[-1]['var_auroc_tok']:.3f}  "
                  f"nll seq/tok {rows[-1]['nll_auroc_seq']:.3f}/{rows[-1]['nll_auroc_tok']:.3f}")

    def best(metric):
        agg = {t: float(np.mean([r[metric] for r in rows if r["t"] == t])) for t in t_grid}
        return max(agg, key=agg.get), agg

    nll_best, nll_by_t = best("nll_auroc_tok")
    var_best, var_by_t = best("var_auroc_tok")
    out = {
        "purpose": "path-time selection sweep for the detector read-out times (appendix tab:config)",
        "produced_by": "scripts/ood_t_selection_sweep.py",
        "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ckpt": str(Path(args.ckpt).resolve().relative_to(_ROOT)),
        "ckpt_md5": _md5(Path(args.ckpt)),
        "split": "validation",
        "t_grid": t_grid, "rate": args.rate, "schemes": ["replace", "shuffle"],
        "fit_seqs": args.fit_seqs, "eval_seqs": args.eval_seqs,
        "corrupt_seed": args.corrupt_seed,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "rows": rows,
        "selected": {
            "nll_t_best_by_token_auroc": nll_best,
            "nll_token_auroc_by_t": nll_by_t,
            "var_t_best_by_token_auroc": var_best,
            "var_token_auroc_by_t": var_by_t,
            "note": "the paper's settings are t_nll=3.0 and t_var=7.5 (tab:config); "
                    "this re-run records the sweep on the pinned checkpoint",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"selected: nll t={nll_best}  var t={var_best}")


if __name__ == "__main__":
    main()
