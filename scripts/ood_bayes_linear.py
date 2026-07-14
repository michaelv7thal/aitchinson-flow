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
from scripts._bench_common import (  # noqa: E402
    heal_style_examples, make_adversarial_negatives, det_metrics, word_metrics,
)
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
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
    ap.add_argument("--head", choices=["hinge", "logistic"], default="hinge",
                    help="energy head: 'hinge' (margin, GPU minibatch) or 'logistic' "
                         "(sklearn LogisticRegression full-dim — the latent-split "
                         "estimator, deployed HELD-OUT; model-agnostic).")
    ap.add_argument("--logistic-c", type=float, default=1.0,
                    help="[logistic] inverse L2 regularisation strength")
    ap.add_argument("--logistic-max-iter", type=int, default=2000)
    ap.add_argument("--logistic-max-fit", type=int, default=60000,
                    help="[logistic] cap on per-class training tokens (subsampled)")
    ap.add_argument("--train-schemes", type=str, default="replace",
                    help="comma list of corruption schemes for the SYNTHETIC "
                         "ADVERSARIAL training negatives (replace/shuffle/falseinfo/"
                         "both/plausible). Default 'replace' (back-compat). "
                         "'replace,shuffle,falseinfo,both' = the general head. Adding "
                         "**plausible** (model-guided min-NLL swap) tests whether ONE "
                         "head can also cover the corruption that anti-transfers: the "
                         "specialists' mean shifts oppose (cos=-0.49) but their LEARNED "
                         "directions align (cos=+0.40), so a unifying w should exist.")
    ap.add_argument("--train-plausible-seqs", type=int, default=128,
                    help="# fit windows to build 'plausible' negatives on (it costs "
                         "n_cands forward passes per swapped word, so it runs on a "
                         "subset; other schemes use all fit windows)")
    ap.add_argument("--train-n-cands", type=int, default=48,
                    help="[plausible] candidate words scored per swap slot")
    ap.add_argument("--train-t-nll", type=float, default=3.0,
                    help="[plausible] path-time for the min-NLL swap search")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument(
        "--ridge",
        type=float,
        default=0.1,
        help="Laplace prior precision: Sigma_w=(Phi + ridge*tr(Phi)/d*I)^-1",
    )
    ap.add_argument("--max-fit-pos", type=int, default=120000)
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both,falseinfo")
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

    # ---- features: clean fit set ----
    fc = _perpos_feats(model, fit_tok, t_eval, device)  # (Nf,L,d)
    d = fc.shape[-1]

    # ---- synthetic ADVERSARIAL negatives: one corrupted copy per train scheme ----
    train_schemes = [s for s in args.train_schemes.split(",") if s.strip()]
    by_len_tr = None
    if any(s in ("falseinfo", "wordswap") for s in train_schemes):
        _txt = " ".join("".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?"
                                for i in row) for row in fit_tok)
        by_len_tr = build_vocab_by_len(_txt)
    # `plausible` in the mix is the model-guided min-NLL swap: expensive (n_cands
    # forward passes per swapped word), so it is generated on a subset of the fit
    # windows. It contributes fewer tokens to the pool, which is fine — each scheme's
    # features and mask are flattened independently below.
    adv = make_adversarial_negatives(fit_tok, K=K, by_len=by_len_tr,
                                     schemes=train_schemes, rate=args.train_rate,
                                     seed=args.seed,
                                     model=model, t_nll=args.train_t_nll,
                                     device=device, n_cands=args.train_n_cands,
                                     plausible_seqs=args.train_plausible_seqs)
    print(f"[bayeslin] adversarial train mix: "
          f"{[(s, int(m.sum())) for s, _, m in adv]}  (scheme, #corrupt tokens)")

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

    # Keep all per-token features on CPU (fit_seqs*L*d_model is multi-GB; moving the
    # whole pool to an 8 GB GPU OOMs — esp. the adversarial multi-scheme mix). The
    # head trains on minibatches shuttled to the GPU per step; only the tiny head
    # lives on the device.
    zc = _proj(fc.reshape(-1, d))             # clean per-token (CPU)
    zk_list, yk_list = [], []                 # corrupted per-token, pooled over schemes
    for _scheme, ct, mask in adv:
        fk_s = _perpos_feats(model, ct, t_eval, device)
        zk_list.append(_proj(fk_s.reshape(-1, d)))          # CPU
        yk_list.append(mask.reshape(-1).float())            # CPU
    zk = torch.cat(zk_list, 0)
    yk = torch.cat(yk_list, 0)
    print(
        f"[bayeslin] d_model={d} feat_dim={feat_dim} t_eval={t_eval} "
        f"clean_tok={zc.shape[0]} corr_tok(label1)={int(yk.sum())} "
        f"mix={'+'.join(train_schemes)}"
    )

    # ---- train the per-token energy head: hinge (default) OR logistic ----
    # positives (valid chars): all clean tokens + the UNCHANGED tokens of corrupt seqs
    zpos = torch.cat([zc, zk[yk == 0]], 0)    # CPU
    zneg = zk[yk == 1]                        # CPU
    if args.head == "logistic":
        # The latent-split ceiling was a full-dim LOGISTIC probe; deploy exactly that
        # as a HELD-OUT detector. energy = signed distance to the LR boundary.
        from sklearn.linear_model import LogisticRegression

        def _samp(t, cap):
            if t.shape[0] > cap:
                return t[torch.randperm(t.shape[0])[:cap]]
            return t
        Xp = _samp(zpos, args.logistic_max_fit).numpy()
        Xn = _samp(zneg, args.logistic_max_fit).numpy()
        Xtr = np.concatenate([Xp, Xn], 0)
        ytr = np.r_[np.zeros(Xp.shape[0]), np.ones(Xn.shape[0])]
        clf = LogisticRegression(C=args.logistic_c, max_iter=args.logistic_max_iter)
        clf.fit(Xtr, ytr)
        print(f"[bayeslin] LOGISTIC head fit on {Xtr.shape[0]} tok "
              f"(C={args.logistic_c}); energy = LR decision function")

        def energy_of(z_cpu):  # (N,feat) CPU -> (N,) CPU energy
            return torch.from_numpy(clf.decision_function(z_cpu.numpy())).float()
    else:
        head = nn.Linear(feat_dim, 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        margin = float(args.margin)
        bs = int(min(16384, zpos.shape[0], zneg.shape[0]))
        gen = torch.Generator().manual_seed(args.seed)
        for step in range(args.steps):
            pi = torch.randint(zpos.shape[0], (bs,), generator=gen)
            ni = torch.randint(zneg.shape[0], (bs,), generator=gen)
            opt.zero_grad()
            ep = head(zpos[pi].to(device)).squeeze(-1)
            en = head(zneg[ni].to(device)).squeeze(-1)
            loss = ep.pow(2).mean() + torch.relu(margin - en).mean()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.steps - 1:
                print(f"  step {step:4d}  E_valid={ep.mean():+.3f}  "
                      f"E_corrupt={en.mean():+.3f}  gap={(en.mean() - ep.mean()):+.3f}")
        head.eval()

        def energy_of(z_cpu):
            return head(z_cpu.to(device)).squeeze(-1).cpu()

    # ---- Laplace posterior covariance, read in the PEAKY regime (var_t) -------
    # The variance is read at var_t (>> t_eval). At t_eval the Mahalanobis
    # variance is INVERTED (corrupt seq -> LOWER var, the pooled feature collapses
    # toward the ID core); the near-one-hot var_t input instead pushes a corrupted
    # token genuinely off-manifold, so Var is non-inverted (corrupt -> HIGHER var).
    # Sigma_w = (Phi + ridge*tr(Phi)/m * I)^-1 over CLEAN per-token feats @ var_t.
    fc_v = _perpos_feats(model, fit_tok, var_t, device)  # (Nf,L,d) clean @ var_t
    dv = fc_v.shape[-1]
    _proj_v, feat_dim_v = _make_proj(fc_v.reshape(-1, dv))  # VARIANCE standardiser
    # Accumulate Phi = E[zs zs^T] in CPU-double chunks (never materialise the whole
    # standardised cloud on the GPU) and factor on CPU (GPU cusolver OOMs/errors
    # once the head features already fill the 8 GB card).
    Zv = _proj_v(fc_v.reshape(-1, dv))  # CPU float
    Phi = torch.zeros(feat_dim_v, feat_dim_v, dtype=torch.float64)
    nrow = 0
    for i in range(0, Zv.shape[0], 32768):
        zb = Zv[i:i + 32768].double()
        Phi += zb.T @ zb
        nrow += zb.shape[0]
    Phi /= nrow
    jit = args.ridge * Phi.trace() / feat_dim_v
    L = torch.linalg.cholesky(
        Phi + jit * torch.eye(feat_dim_v, dtype=Phi.dtype)
    )  # CPU double; Var = ||L^-1 zs||^2
    print(f"[bayeslin] energy @ t_eval={t_eval}  variance @ var_t={var_t} "
          f"(feat_dim_var={feat_dim_v})")

    @torch.no_grad()
    def score(tok, chunk=16):
        """token_ids (B,L) -> per-token energy E_t (B,L) and variance Var_t (B,L)."""
        Es, Vs = [], []
        for i in range(0, tok.shape[0], chunk):
            tb = tok[i : i + chunk]
            h = _perpos_feats(model, tb, t_eval, device)  # energy feats @ t_eval
            b, Lq, _ = h.shape
            z = _proj(h.reshape(-1, d))  # (b*L,feat_dim) CPU
            E = energy_of(z).reshape(b, Lq)
            hv = _perpos_feats(model, tb, var_t, device)  # variance feats @ var_t
            zv = _proj_v(hv.reshape(-1, dv))              # CPU
            w = torch.linalg.solve_triangular(L, zv.double().T, upper=False)  # CPU
            V = (w * w).sum(0).reshape(b, Lq).float()
            Es.append(E)
            Vs.append(V)
        return torch.cat(Es), torch.cat(Vs)

    # false-info replacement vocab (built once from the fit windows), if requested
    _req_schemes = [s for s in args.schemes.split(",") if s.strip()]
    by_len = None
    if any(s in ("falseinfo", "wordswap") for s in _req_schemes):
        fit_txt = " ".join(
            "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in row)
            for row in fit_tok
        )
        by_len = build_vocab_by_len(fit_txt)
        print(f"[blr] false-info vocab: lengths {min(by_len)}-{max(by_len)}")

    def _corrupt(tok, scheme, rate, seed):
        if scheme == "replace":
            return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        if scheme == "shuffle":
            return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
        if scheme in ("falseinfo", "wordswap"):
            return corrupt_false_info(tok, rate, by_len, seed=seed)
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
        f"{'tokAUROC_E':>11} {'tokAUROC_Var':>13} {'tokP@5':>7} {'tokR@5':>7} {'tokF1@5':>8}"
    )
    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    Ep_tok = Ep.numpy().reshape(-1)  # clean per-token energy: threshold calibration
    for scheme in schemes:
        for r in rates:
            ot = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
            Eo, Vo = score(ot)
            Eo_seq = Eo.mean(1).numpy()
            lab = np.r_[np.zeros(len(Ep_seq)), np.ones(Eo.shape[0])]
            au_E = _auroc(np.r_[Ep_seq, Eo_seq], lab)  # seq energy
            au_V = _auroc(np.r_[Vp_seq, Vo.mean(1).numpy()], lab)  # seq uncertainty
            changed = ot != pos_tok
            tl_E = tl_V = float("nan")
            cm = changed.numpy().reshape(-1).astype(int)
            if changed.any() and (~changed).any():
                tl_E = _auroc(Eo.numpy().reshape(-1), cm)  # per-token energy
                tl_V = _auroc(Vo.numpy().reshape(-1), cm)  # per-token uncertainty
            # --- operating-point metrics on the ENERGY head (the deployed score) ---
            m_seq = det_metrics(Ep_seq, Ep_seq, Eo_seq)
            eo_tok = Eo.numpy().reshape(-1)
            m_tok = det_metrics(Ep_tok, eo_tok[cm == 0], eo_tok[cm == 1])
            # WORD level — the common unit vs the BPE-tokenised LM baselines.
            w_max = word_metrics(Ep, pos_tok, Eo, ot, changed, op="max")
            w_mean = word_metrics(Ep, pos_tok, Eo, ot, changed, op="mean")
            p5 = m_tok["prf"].get("0.05", {})
            print(
                f"{scheme:>9} {r:>5.2f} {au_E:>12.4f} {au_V:>14.4f} {tl_E:>11.4f} "
                f"{tl_V:>13.4f} {p5.get('precision', float('nan')):>7.3f} "
                f"{p5.get('recall', float('nan')):>7.3f} "
                f"{p5.get('f1', float('nan')):>8.3f}"
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
                    "prf_seq": m_seq,
                    "prf_token": m_tok,
                    "word_max": w_max,
                    "word_mean": w_mean,
                    "auroc_word_max": w_max["word"].get("auroc"),
                    "auroc_word_mean": w_mean["word"].get("auroc"),
                    "E_seq_mean": float(Eo.mean()),
                    "Var_seq_mean": float(Vo.mean()),
                }
            )

    # ---- healing-style per-token examples (clean/replace30/falseinfo30 + flagged) --
    print("\n=== healing-style examples ===")
    examples, ex_flag_thr = heal_style_examples(
        lambda t: score(t)[0], pos_tok, pos_tok[:2], K=K, by_len=by_len,
        score_key="E_t", example_fpr=0.05, max_pos=args.max_pos)

    out_path = Path(args.out or (Path(args.ckpt).parent / "bayes_linear_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ckpt": args.ckpt,
                "train_schemes": train_schemes,
                "head": args.head,
                "t_eval": t_eval,
                "var_t_eval": var_t,
                "d_model": d,
                "feat_dim": feat_dim,
                "pca_dim": args.pca_dim,
                "margin": (args.margin if args.head == "hinge" else None),
                "ridge": args.ridge,
                "detector": "BayesLinHead",
                "detector_long": "Bayesian linear energy head; E=discriminative, Var=uncertainty; per-token + sequence",
                "flag_thr": ex_flag_thr,
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
