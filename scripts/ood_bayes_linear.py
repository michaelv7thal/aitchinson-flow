"""Bayesian linear-head OOD detector ("BayesLinHead") on DirichletFM.

A linear energy head  E(zs) = w·zs + b  on the (standardised, DETERMINISTIC)
backbone features, trained with a per-TOKEN hinge (valid char -> E~0, corrupted
char -> E>=margin). A Laplace / Bayesian-linear posterior on the weights gives a
predictive VARIANCE = the uncertainty:

    Sigma_w = (Phi + lambda*I)^-1 ,  Phi = sum over in-dist per-token zs zs^T
    Var[E(zs)] = zs^T Sigma_w zs            (= a Mahalanobis-style quadratic form)

Both granularities come from the SAME w, b, Sigma_w:
    per-token :  E_t   = w·zs_t + b ,  Var_t   = zs_t^T   Sigma_w zs_t     (B, L)
    sequence  :  E_seq = mean_t E_t  ,  Var_seq = zs_seq^T Sigma_w zs_seq  (B,)
(E_seq = mean_t E_t exactly, since the head is linear and pooling is a mean.)

E = "is it invalid" (discriminative); Var = "how uncertain / how far from the
in-distribution manifold". DirichletFM only (DirichletFlowMatching); guards
against the Discrete-FM 'DFM' arm. Distinct from HingeSVGP / PerPosVarGP /
PerPosMaha.

Usage:
    python scripts/ood_bayes_linear.py \
        --ckpt runs/.../DirichletFM/epoch_final.pt \
        --out  ood_out/bayeslin/bayes_linear_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

from scripts.ood_variance_perpos import (  # noqa: E402
    _load_dirichletfm,
    _perpos_feats,
    _auroc,
)
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 (best-effort for example dump)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-eval", type=float, default=None,
                    help="path-time for the ENERGY readout (discriminative, ~4.5)")
    ap.add_argument("--var-t-eval", type=float, default=7.5,
                    help="path-time for the VARIANCE readout. The Mahalanobis "
                         "variance is INVERTED at t_eval~4.5 (corrupt -> LOWER var) "
                         "and only non-inverted in the peaky regime (~7.5).")
    ap.add_argument(
        "--fit-seqs",
        type=int,
        default=512,
        help="# in-distribution sequences for head training + Laplace",
    )
    ap.add_argument("--n", type=int, default=256, help="# eval sequences per split")
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="0 = raw d_model features (default); >0 = top-k PCs",
    )
    ap.add_argument(
        "--train-rate",
        type=float,
        default=0.5,
        help="replace-corruption rate for the per-token training negatives",
    )
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument(
        "--ridge",
        type=float,
        default=0.1,
        help="Laplace prior precision: Sigma_w=(Phi + ridge*tr(Phi)/d*I)^-1",
    )
    ap.add_argument("--max-fit-pos", type=int, default=120000)
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="dataset split for fit/eval sequences "
                         "(test = held-out last-5M text8 split)")
    ap.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write AUROC-sweep + per-token heatmap PNGs next to --out",
    )
    ap.add_argument(
        "--max-pos",
        type=int,
        default=120,
        help="positions shown in the per-token heatmaps",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K
    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    var_t = float(args.var_t_eval)
    t_max = float(cfg.dirichlet_fm.t_max)
    if not (1.0 <= var_t <= t_max):
        raise SystemExit(f"--var-t-eval={var_t} must lie in [1, t_max={t_max}]")
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

    # ---- features: clean fit set + corrupted (for per-token training labels) ----
    fc = _perpos_feats(model, fit_tok, t_eval, device)  # (Nf,L,d)
    d = fc.shape[-1]
    corr_tok = corrupt_token_ids(
        fit_tok.clone(), vocab_size=K, corrupt_rate=args.train_rate, seed=args.seed
    )
    fk = _perpos_feats(model, corr_tok, t_eval, device)  # (Nf,L,d)
    label_corr = corr_tok != fit_tok  # (Nf,L) bool

    def _make_proj(fc_flat):
        """Standardiser (+optional PCA) fitted on a CLEAN feature cloud. Returns
        (proj_fn, feat_dim). Built twice: once for the energy features (@t_eval)
        and once, independently, for the variance features (@var_t)."""
        d_ = fc_flat.shape[-1]
        mu = fc_flat.mean(0)
        sigma = fc_flat.std(0).clamp_min(1e-6)
        pca_V = None
        if args.pca_dim and args.pca_dim < d_:
            Xs0 = (fc_flat - mu) / sigma
            if Xs0.shape[0] > args.max_fit_pos:
                Xs0 = Xs0[torch.randperm(Xs0.shape[0])[: args.max_fit_pos]]
            _, _, V = torch.pca_lowrank(Xs0, q=min(args.pca_dim, d_))
            pca_V = V[:, : args.pca_dim]
        fdim = args.pca_dim if pca_V is not None else d_

        def proj(h_flat):  # raw (N,d) -> standardised (+pca); device-safe
            m = mu.to(h_flat.device)
            s = sigma.to(h_flat.device)
            z = (h_flat - m) / s
            return z @ pca_V.to(h_flat.device) if pca_V is not None else z

        return proj, fdim

    _proj, feat_dim = _make_proj(fc.reshape(-1, d))  # ENERGY standardiser @ t_eval

    zc = _proj(fc.reshape(-1, d)).to(device)  # clean per-token
    zk = _proj(fk.reshape(-1, d)).to(device)  # corrupted per-token
    yk = label_corr.reshape(-1).to(device).float()  # 1 = corrupted char
    print(
        f"[bayeslin] d_model={d} feat_dim={feat_dim} t_eval={t_eval} "
        f"clean_tok={zc.shape[0]} corr_tok(label1)={int(yk.sum())}"
    )

    # ---- train the linear head with a per-TOKEN hinge ----
    head = nn.Linear(feat_dim, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    margin = float(args.margin)
    # positives (valid chars): all clean tokens + the UNCHANGED tokens of corrupt seqs
    zpos = torch.cat([zc, zk[yk == 0]], 0)
    zneg = zk[yk == 1]
    for step in range(args.steps):
        opt.zero_grad()
        ep = head(zpos).squeeze(-1)
        en = head(zneg).squeeze(-1)
        loss = ep.pow(2).mean() + torch.relu(margin - en).mean()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(
                f"  step {step:4d}  E_valid={ep.mean():+.3f}  E_corrupt={en.mean():+.3f}"
                f"  gap={(en.mean() - ep.mean()):+.3f}"
            )

    # ---- Laplace posterior covariance, read in the PEAKY regime (var_t) -------
    # The variance is read at var_t (>> t_eval). At t_eval the Mahalanobis
    # variance is INVERTED (corrupt seq -> LOWER var, the pooled feature collapses
    # toward the ID core); the near-one-hot var_t input instead pushes a corrupted
    # token genuinely off-manifold, so Var is non-inverted (corrupt -> HIGHER var).
    # Sigma_w = (Phi + ridge*tr(Phi)/m * I)^-1 over CLEAN per-token feats @ var_t.
    fc_v = _perpos_feats(model, fit_tok, var_t, device)  # (Nf,L,d) clean @ var_t
    dv = fc_v.shape[-1]
    _proj_v, feat_dim_v = _make_proj(fc_v.reshape(-1, dv))  # VARIANCE standardiser
    zc_v = _proj_v(fc_v.reshape(-1, dv)).to(device)
    Phi = zc_v.double().T @ zc_v.double() / zc_v.shape[0]
    jit = args.ridge * Phi.trace() / feat_dim_v
    L = torch.linalg.cholesky(
        Phi + jit * torch.eye(feat_dim_v, device=device, dtype=Phi.dtype)
    )  # Var = ||L^-1 zs||^2
    print(f"[bayeslin] energy @ t_eval={t_eval}  variance @ var_t={var_t} "
          f"(feat_dim_var={feat_dim_v})")

    head.eval()

    @torch.no_grad()
    def score(tok, chunk=16):
        """token_ids (B,L) -> per-token energy E_t (B,L) and variance Var_t (B,L)."""
        Es, Vs = [], []
        for i in range(0, tok.shape[0], chunk):
            tb = tok[i : i + chunk]
            h = _perpos_feats(model, tb, t_eval, device)  # energy feats @ t_eval
            b, Lq, _ = h.shape
            z = _proj(h.reshape(-1, d)).to(device)  # (b*L,feat_dim)
            E = head(z).squeeze(-1).reshape(b, Lq).cpu()
            hv = _perpos_feats(model, tb, var_t, device)  # variance feats @ var_t
            zv = _proj_v(hv.reshape(-1, dv)).to(device)
            w = torch.linalg.solve_triangular(L, zv.double().T, upper=False)
            V = (w * w).sum(0).reshape(b, Lq).float().cpu()
            Es.append(E)
            Vs.append(V)
        return torch.cat(Es), torch.cat(Vs)

    def _corrupt(tok, scheme, rate, seed):
        if scheme == "replace":
            return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        if scheme == "shuffle":
            return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
        out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)

    # ---- eval: sequence-level AND per-token AUROC over the corruption ladder ----
    Ep, Vp = score(pos_tok)
    Ep_seq, Vp_seq = Ep.mean(1).numpy(), Vp.mean(1).numpy()
    rows = [
        {
            "scheme": None,
            "rate": 0.0,
            "n": int(pos_tok.shape[0]),
            "E_seq_mean": float(Ep.mean()),
            "Var_seq_mean": float(Vp.mean()),
        }
    ]
    print(
        f"\n{'scheme':>9} {'rate':>5} {'AUROC_seq_E':>12} {'AUROC_seq_Var':>14} "
        f"{'tokAUROC_E':>11} {'tokAUROC_Var':>13}"
    )
    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    for scheme in schemes:
        for r in rates:
            ot = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
            Eo, Vo = score(ot)
            lab = np.r_[np.zeros(len(Ep_seq)), np.ones(Eo.shape[0])]
            au_E = _auroc(np.r_[Ep_seq, Eo.mean(1).numpy()], lab)  # seq energy
            au_V = _auroc(np.r_[Vp_seq, Vo.mean(1).numpy()], lab)  # seq uncertainty
            changed = ot != pos_tok
            tl_E = tl_V = float("nan")
            if changed.any() and (~changed).any():
                cm = changed.numpy().reshape(-1).astype(int)
                tl_E = _auroc(Eo.numpy().reshape(-1), cm)  # per-token energy
                tl_V = _auroc(Vo.numpy().reshape(-1), cm)  # per-token uncertainty
            print(
                f"{scheme:>9} {r:>5.2f} {au_E:>12.4f} {au_V:>14.4f} {tl_E:>11.4f} {tl_V:>13.4f}"
            )
            rows.append(
                {
                    "scheme": scheme,
                    "rate": r,
                    "n": int(ot.shape[0]),
                    "auroc_seq_energy": au_E,
                    "auroc_seq_uncertainty": au_V,
                    "auroc_token_energy": tl_E,
                    "auroc_token_uncertainty": tl_V,
                    "E_seq_mean": float(Eo.mean()),
                    "Var_seq_mean": float(Vo.mean()),
                }
            )

    # ---- a couple of per-token examples (char, E_t, Var_t, corrupted) for heatmaps ----
    examples = []
    ex_tok = pos_tok[:2]
    ex_corr = corrupt_token_ids(ex_tok.clone(), vocab_size=K, corrupt_rate=0.3, seed=7)
    ex_changed = ex_corr != ex_tok  # (2, L) bool
    for tag, tk in [("clean", ex_tok), ("corrupt30", ex_corr)]:
        E, V = score(tk)
        for b in range(tk.shape[0]):
            ids = tk[b].tolist()
            # which positions are corrupted vs the clean source (clean -> none)
            ch = (
                ex_changed[b]
                if tag == "corrupt30"
                else torch.zeros_like(ex_tok[b], dtype=torch.bool)
            )
            examples.append(
                {
                    "which": tag,
                    "idx": b,
                    "text": "".join(
                        _ALPH[i] if i < len(_ALPH) else "?" for i in ids[:120]
                    ),
                    "E_t": [round(float(x), 4) for x in E[b][:120].tolist()],
                    "Var_t": [round(float(x), 4) for x in V[b][:120].tolist()],
                    "corrupted": [bool(x) for x in ch[:120].tolist()],
                }
            )

    out_path = Path(args.out or (Path(args.ckpt).parent / "bayes_linear_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ckpt": args.ckpt,
                "t_eval": t_eval,
                "var_t_eval": var_t,
                "d_model": d,
                "feat_dim": feat_dim,
                "pca_dim": args.pca_dim,
                "margin": margin,
                "ridge": args.ridge,
                "detector": "BayesLinHead",
                "detector_long": "Bayesian linear energy head; E=discriminative, Var=uncertainty; per-token + sequence",
                "rows": rows,
                "examples": examples,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    if args.plot:
        try:
            from scripts.plot_ood_bayes_linear import plot_from_json

            for p in plot_from_json(out_path, max_pos=args.max_pos):
                print(f"Wrote {p}")
        except ImportError as e:
            print(
                f"[plot] skipped ({e}). Install matplotlib (`uv sync` or run via "
                f"`uv run`) or pass --no-plot. Re-plot any time with:\n"
                f"  python scripts/plot_ood_bayes_linear.py --json {out_path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
