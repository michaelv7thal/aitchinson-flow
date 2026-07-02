"""Denoiser-NLL OOD detector ("DenoiserNLL") on DirichletFM (Dirichlet FM).

The model's OWN per-token surprise is a training-free OOD score:

    NLL_t = -log softmax( f(x_t, t) )[ observed_token_t ]

read at a LOW-ish path time (t_nll ~ 3) where the deterministic Dirichlet-mean
input is near-uniform, so the denoiser must predict each token from CONTEXT — a
corrupted / out-of-place token then gets a low predicted probability => high NLL.
No head to train, no fit set; works for BOTH substitution (replace) and order
(shuffle) corruption, and localizes per position.

Alongside it we carry the Bayesian-linear VARIANCE baseline, but read at a HIGH
path time (t_var ~ 7.5):  Var(z)=z^T (Phi+jit I)^-1 z over clean per-token feats.
At low/mid t the feature collapses toward the ID core and Var INVERTS (corrupt =>
LOW var); only near t_max does it recover as a (sequence-only) detector. See
scripts/diag_var_tsweep.py for the t-sweep that motivates t_nll / t_var.

Reports, per corruption cell, SEQUENCE and PER-TOKEN AUROC for both scores, plus
per-token examples for heatmaps. DirichletFM only (guards against Discrete-FM
'DFM'). Model-agnostic w.r.t. the backbone: any checkpoint exposing
get_hidden_states + forward (logits) works — drop a new-backbone ckpt straight in.

Usage:
    python scripts/ood_denoiser_nll.py \
        --ckpt runs/.../DirichletFM/epoch_final.pt \
        --out  ood_out/nll/denoiser_nll_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.ood_variance_perpos import _load_dirichletfm, _auroc  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 (best-effort for example dump)


def _det_mean_xt(tok: torch.Tensor, t: float, K: int, device: str) -> torch.Tensor:
    """Deterministic Dirichlet mean beta/sum(beta) for token ids at path-time t."""
    B, L = tok.shape
    beta = torch.ones(B, L, K, device=device)
    beta.scatter_(-1, tok.to(device).long().unsqueeze(-1), float(t))
    return beta / beta.sum(-1, keepdim=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-nll", type=float, default=3.0,
                    help="path time for the denoiser-NLL readout (context regime)")
    ap.add_argument("--t-var", type=float, default=7.5,
                    help="path time for the variance baseline (peaky regime where "
                         "Var is no longer inverted); set <=0 to skip the baseline")
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="# in-distribution seqs for the variance Phi (NLL needs none)")
    ap.add_argument("--n", type=int, default=256, help="# eval sequences per split")
    ap.add_argument("--ridge", type=float, default=0.1,
                    help="Laplace prior precision: Sigma_w=(Phi + ridge*tr(Phi)/d I)^-1")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="dataset split for fit/eval sequences "
                         "(test = held-out last-5M text8 split)")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-pos", type=int, default=120,
                    help="positions shown in the per-token heatmaps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K
    t_max = float(cfg.dirichlet_fm.t_max)
    use_var = args.t_var and args.t_var > 0
    for nm, tv in [("t_nll", args.t_nll), ("t_var", args.t_var if use_var else 1.0)]:
        if not (1.0 <= float(tv) <= t_max):
            raise SystemExit(f"--{nm}={tv} must lie in [1, t_max={t_max}]")

    dm, _ = build_training_datamodule(cfg)
    _split_loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
                      "test": dm.test_dataloader}
    vl = _split_loaders[args.split]() or dm.train_dataloader()
    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs)
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]
    print(f"[nll] t_nll={args.t_nll} t_var={args.t_var if use_var else '—'} "
          f"t_max={t_max} fit={fit_tok.shape[0]} eval={pos_tok.shape[0]}")

    # ---- variance baseline: clean Phi at t_var (deterministic mean features) ----
    Lc = mu = sigma = d = None
    if use_var:
        @torch.no_grad()
        def _feats(tok, t):
            outs = []
            for i in range(0, tok.shape[0], args.chunk):
                tb = tok[i : i + args.chunk]
                x_t = _det_mean_xt(tb, t, K, device)
                tt = torch.full((tb.shape[0],), float(t), device=device)
                outs.append(model.get_hidden_states(x_t, tt).cpu())
            return torch.cat(outs)

        fc = _feats(fit_tok, args.t_var)
        d = fc.shape[-1]
        mu = fc.reshape(-1, d).mean(0)
        sigma = fc.reshape(-1, d).std(0).clamp_min(1e-6)
        zc = ((fc.reshape(-1, d) - mu) / sigma).double()
        Phi = zc.T @ zc / zc.shape[0]
        jit = args.ridge * Phi.trace() / d
        Lc = torch.linalg.cholesky(Phi + jit * torch.eye(d, dtype=Phi.dtype))
        print(f"[nll] variance baseline: d={d} t_var={args.t_var}")

    @torch.no_grad()
    def score(tok):
        """token_ids (B,L) -> (NLL_t (B,L), Var_t (B,L) or None)."""
        Ns, Vs = [], []
        for i in range(0, tok.shape[0], args.chunk):
            tb = tok[i : i + args.chunk]
            b, Lq = tb.shape
            # NLL at t_nll
            x_n = _det_mean_xt(tb, args.t_nll, K, device)
            tt = torch.full((b,), float(args.t_nll), device=device)
            logits = model.forward(x_n, tt)
            nll = -F.log_softmax(logits, -1).gather(
                -1, tb.to(device).long().unsqueeze(-1)).squeeze(-1)
            Ns.append(nll.cpu())
            if use_var:
                x_v = _det_mean_xt(tb, args.t_var, K, device)
                tv = torch.full((b,), float(args.t_var), device=device)
                h = model.get_hidden_states(x_v, tv).reshape(b * Lq, d)
                z = ((h.cpu() - mu) / sigma).double()
                w = torch.linalg.solve_triangular(Lc, z.T, upper=False)
                Vs.append((w * w).sum(0).reshape(b, Lq).float())
        return torch.cat(Ns), (torch.cat(Vs) if use_var else None)

    def _corrupt(tok, scheme, rate, seed):
        if scheme == "replace":
            return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        if scheme == "shuffle":
            return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
        out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)

    # ---- corruption ladder: sequence + per-token AUROC for NLL and Var ----------
    Np, Vp = score(pos_tok)
    Np_seq = Np.mean(1).numpy()
    Vp_seq = Vp.mean(1).numpy() if use_var else None
    rows = [{"scheme": None, "rate": 0.0, "n": int(pos_tok.shape[0]),
             "nll_seq_mean": float(Np.mean()),
             "var_seq_mean": (float(Vp.mean()) if use_var else None)}]
    print(f"\n{'scheme':>9} {'rate':>5} {'NLL_AUseq':>10} {'NLL_AUtok':>10} "
          f"{'Var_AUseq':>10} {'Var_AUtok':>10}")
    for scheme in [s for s in args.schemes.split(",") if s.strip()]:
        for r in [float(x) for x in args.rates.split(",") if x.strip()]:
            ot = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
            No, Vo = score(ot)
            lab = np.r_[np.zeros(len(Np_seq)), np.ones(No.shape[0])]
            au_n_seq = _auroc(np.r_[Np_seq, No.mean(1).numpy()], lab)
            changed = (ot != pos_tok)
            cm = changed.numpy().reshape(-1).astype(int)
            has_tok = changed.any() and (~changed).any()
            au_n_tok = _auroc(No.numpy().reshape(-1), cm) if has_tok else float("nan")
            au_v_seq = au_v_tok = float("nan")
            if use_var:
                au_v_seq = _auroc(np.r_[Vp_seq, Vo.mean(1).numpy()], lab)
                au_v_tok = _auroc(Vo.numpy().reshape(-1), cm) if has_tok else float("nan")
            print(f"{scheme:>9} {r:>5.2f} {au_n_seq:>10.4f} {au_n_tok:>10.4f} "
                  f"{au_v_seq:>10.4f} {au_v_tok:>10.4f}")
            rows.append({"scheme": scheme, "rate": r, "n": int(ot.shape[0]),
                         "auroc_seq_nll": au_n_seq, "auroc_token_nll": au_n_tok,
                         "auroc_seq_var": au_v_seq, "auroc_token_var": au_v_tok,
                         "nll_seq_mean": float(No.mean()),
                         "var_seq_mean": (float(Vo.mean()) if use_var else None)})

    # ---- per-token examples (clean + corrupt30) for heatmaps --------------------
    examples = []
    ex_tok = pos_tok[:2]
    ex_corr = corrupt_token_ids(ex_tok.clone(), vocab_size=K, corrupt_rate=0.3, seed=7)
    ex_changed = ex_corr != ex_tok
    for tag, tk in [("clean", ex_tok), ("corrupt30", ex_corr)]:
        Ne, Ve = score(tk)
        for b in range(tk.shape[0]):
            ids = tk[b].tolist()
            ch = (ex_changed[b] if tag == "corrupt30"
                  else torch.zeros_like(ex_tok[b], dtype=torch.bool))
            examples.append({
                "which": tag, "idx": b,
                "text": "".join(_ALPH[i] if i < len(_ALPH) else "?" for i in ids[:120]),
                "NLL_t": [round(float(x), 4) for x in Ne[b][:120].tolist()],
                "Var_t": ([round(float(x), 4) for x in Ve[b][:120].tolist()]
                          if use_var else None),
                "corrupted": [bool(x) for x in ch[:120].tolist()],
            })

    out_path = Path(args.out or (Path(args.ckpt).parent / "denoiser_nll_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "t_nll": args.t_nll,
        "t_var": (args.t_var if use_var else None), "t_max": t_max,
        "d_model": d, "K": K, "ridge": args.ridge,
        "detector": "DenoiserNLL",
        "detector_long": "denoiser per-token NLL (training-free); +Bayesian-linear "
                         "variance baseline at t_var; per-token + sequence",
        "rows": rows, "examples": examples,
    }, indent=2))
    print(f"\nWrote {out_path}")

    if args.plot:
        try:
            from scripts.plot_ood_nll import plot_from_json
            for p in plot_from_json(out_path, max_pos=args.max_pos):
                print(f"Wrote {p}")
        except ImportError as e:
            print(f"[plot] skipped ({e}); re-plot with scripts/plot_ood_nll.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
