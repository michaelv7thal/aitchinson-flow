"""Per-position BAYESIAN (Dirichlet-process) infinite-mixture OOD detector
("PerPosBGMM") on DirichletFM.

This is the ``scripts/ood_gmm_perpos.py`` detector with the fixed-K
``GaussianMixture`` swapped for a ``BayesianGaussianMixture`` with a
**Dirichlet-process (stick-breaking) prior** on the mixture weights. Instead of
committing to ``--n-components 8`` a priori, we set a generous UPPER BOUND
(``--max-components``) and let the variational posterior PRUNE unused clusters:
the effective number of active components is inferred from the data (reported as
``n_effective`` = # weights above ``--active-thresh``). This is the "correct
number of clusters" the fixed-K GMM has to guess — the concentration prior
(``--weight-conc-prior`` = the DP ``alpha``) is the only knob, and smaller alpha
=> fewer clusters.

Score is unchanged: per-token negative log-likelihood under the (Bayesian)
mixture density

    score(z) = -log p(z),   p(z) = sum_k pi_k N(z; mu_k, Sigma_k),

read on the SAME deterministic Dirichlet-mean backbone features every other
per-position detector uses (``ood_variance_perpos._perpos_feats`` at ``t_eval``).
The determinism is load-bearing (a stochastic Dirichlet sample collapses
clean-vs-corrupt features).

MOTIVATION mirrors the GMM detector: a mixture density over the DirichletFM
contextual backbone features may flag "false information" (a valid same-length
word swapped for a different valid word) that the local char-NLL localizer
misses — an off-manifold wrong-word-in-context point scores as low density. The
Bayesian variant removes the ``n_components`` free parameter, so the density is
not under/over-fit by a hand-picked K.

OUTPUT. Like ``ood_gmm_perpos.py`` we sweep ``--t-evals`` and the corruption
ladder and record SEQUENCE (max- and mean-pooled) + PER-TOKEN AUROC per cell.
Additionally — matching ``scripts/ood_denoiser_nll.py`` /
``ood_out_best/nll/denoiser_nll_sweep.json`` — we emit the "normal OOD numbers":
``auroc_seq_gmm`` / ``auroc_token_gmm`` fields and a per-token ``examples`` block
(clean + corrupt30 GMM-NLL arrays with ``corrupted`` flags) for heatmaps.

DISTINCT detector from PerPosGMM (fixed-K), PerPosMaha (single Gaussian),
HingeSVGP and PerPosVarGP — different fit (variational DP mixture), own output
(bgmm_perpos_sweep.json). DirichletFM == DirichletFlowMatching (Stark et al.
2024). NOT the Discrete-FM 'DFM' arm; the loader guards against it.

Usage:
    python scripts/ood_bgmm_perpos.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt \
        --split test --max-components 20 --covariance-type full --pca-dim 64 \
        --schemes replace,shuffle,both,falseinfo --t-evals 1.5,3.0,4.5,6.0 \
        --out ood_out/bgmm/bgmm_perpos_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


_bootstrap()

# Reuse the variance detector's load+guard and feature extraction so all
# per-position detectors are guaranteed identical on those steps.
from scripts.ood_variance_perpos import (  # noqa: E402
    _load_dirichletfm,
    _perpos_feats,
    _auroc,
)
from scripts._bench_common import det_metrics, word_metrics  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
)
from sklearn.mixture import BayesianGaussianMixture  # noqa: E402

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 decode (a-z + space)


def _decode_tokens(tok: torch.Tensor) -> str:
    """Decode (N, L) token windows to one concatenated text8 string (space-joined)."""
    return " ".join(
        "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in row)
        for row in tok
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-eval", type=float, default=None,
                    help="path-time for the deterministic-mean features (default "
                         "cfg.dfm_svgp.t_eval, ~4.5). Ignored if --t-evals is set.")
    ap.add_argument("--t-evals", type=str, default="",
                    help="comma list of t_eval values to sweep (e.g. 1.5,3.0,4.5,6.0). "
                         "Refits PCA+BGMM and reruns the ladder per t; records t_eval "
                         "per row. Lower t = more context-dependent features.")
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="# in-distribution sequences to fit the mixture")
    ap.add_argument("--n", type=int, default=256, help="# eval sequences per split")
    ap.add_argument("--max-fit-pos", type=int, default=80000,
                    help="cap on # per-position vectors used to fit the BGMM")
    # --- Bayesian (Dirichlet-process) mixture knobs ---
    ap.add_argument("--max-components", type=int, default=20,
                    help="UPPER BOUND on clusters; the DP posterior prunes unused "
                         "ones. Effective K is inferred (reported as n_effective).")
    ap.add_argument("--weight-conc-prior", type=float, default=None,
                    help="Dirichlet-process concentration (alpha). None => sklearn "
                         "default 1/max_components. Smaller => fewer active clusters.")
    ap.add_argument("--active-thresh", type=float, default=None,
                    help="a component counts as 'active' if its weight exceeds this "
                         "(default 1/(2*max_components)).")
    ap.add_argument("--covariance-type", choices=["diag", "full", "tied", "spherical"],
                    default="full",
                    help="mixture covariance. 'full' (after PCA to a small --pca-dim) "
                         "is the natural choice for a DP mixture; use 'diag' for high-d.")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="0 = full-d_model mixture. >0 truncates to top-k PCs of the "
                         "standardised ID cloud before fitting (recommended for 'full').")
    ap.add_argument("--reg-covar", type=float, default=1e-4,
                    help="added to the covariance diagonal (singular-cov guard).")
    ap.add_argument("--n-init", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=500,
                    help="variational EM iterations. A full-cov DP mixture on the "
                         "PCA-64 cloud does NOT converge in the plain-GMM default of "
                         "~200 (sklearn ConvergenceWarning); 500 clears it.")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="variational lower-bound convergence tolerance.")
    ap.add_argument("--init-params",
                    choices=["kmeans", "k-means++", "random", "random_from_data"],
                    default="k-means++",
                    help="responsibility init. 'k-means++' starts EM far closer to a "
                         "good mode than the plain-'kmeans' default, so it converges "
                         "in far fewer iterations.")
    # --- corruption ladder ---
    ap.add_argument("--schemes", type=str, default="replace,shuffle,both,falseinfo")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="dataset split for fit/eval sequences "
                         "(test = held-out last-5M text8 split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos", type=int, default=120,
                    help="positions kept in the per-token examples block")
    ap.add_argument("--example-t", type=float, default=None,
                    help="t_eval used to SCORE the healing-style examples (default: "
                         "the largest swept t, the best falseinfo performer)")
    ap.add_argument("--example-fpr", type=float, default=0.05,
                    help="clean-token FPR used to pick the example flag threshold "
                         "(like the heal calibration): flagged = score > q(1-fpr))")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device0 = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device0)
    device = next(model.parameters()).device
    K = cfg.text8_dataset.K
    default_t = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    t_evals = ([float(x) for x in args.t_evals.split(",") if x.strip()]
               if args.t_evals.strip() else [default_t])
    active_thresh = (args.active_thresh if args.active_thresh is not None
                     else 1.0 / (2 * args.max_components))
    dm, _ = build_training_datamodule(cfg)
    _split_loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
                      "test": dm.test_dataloader}
    loader = _split_loaders[args.split]() or dm.train_dataloader()

    # ---- in-distribution sequences (fit + eval-positive) ------------------
    seqs = []
    for batch in loader:
        seqs.append(batch["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs, dim=0)
    fit_tok = seqs[: args.fit_seqs]
    pos_tok = seqs[args.fit_seqs: args.fit_seqs + args.n]

    schemes = [s for s in args.schemes.split(",") if s.strip()]
    rates = [float(r) for r in args.rates.split(",") if r.strip()]

    # ---- false-info replacement vocab (built once from the fit windows) ----
    by_len = None
    if any(s in ("falseinfo", "wordswap") for s in schemes):
        by_len = build_vocab_by_len(_decode_tokens(fit_tok))
        print(f"[bgmm] false-info vocab: lengths {min(by_len)}-{max(by_len)} "
              f"(e.g. len-6: {by_len.get(6, [])[:5]})")

    def _corrupt(tok, scheme, rate, seed):
        if scheme == "replace":
            return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        if scheme == "shuffle":
            return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
        if scheme in ("falseinfo", "wordswap"):
            return corrupt_false_info(tok, rate, by_len, seed=seed)
        out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
        return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)

    def run_one_teval(t_eval: float):
        """Fit PCA+BGMM at t_eval and score the full corruption ladder.

        Returns (rows, meta, scorer) so the caller can reuse the fitted scorer
        for the per-token examples block.
        """
        fit_feats = _perpos_feats(model, fit_tok, t_eval, device)        # (Nf,L,d)
        d_model = fit_feats.shape[-1]
        X = fit_feats.reshape(-1, d_model)
        if X.shape[0] > args.max_fit_pos:
            X = X[torch.randperm(X.shape[0])[: args.max_fit_pos]]
        # standardise on the ID cloud (raw activation scales vary by orders of
        # magnitude — a mixture likelihood on unscaled features is numerically poor).
        mu = X.mean(0)
        sigma = X.std(0).clamp_min(1e-6)
        Xz = (X - mu) / sigma
        pca_V = None
        if args.pca_dim and args.pca_dim < d_model:
            _, _, V = torch.pca_lowrank(Xz, q=min(args.pca_dim, Xz.shape[1]))
            pca_V = V[:, : args.pca_dim]
            Xz = Xz @ pca_V
        feat_dim = Xz.shape[1]
        bgmm = BayesianGaussianMixture(
            n_components=args.max_components,
            covariance_type=args.covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=args.weight_conc_prior,
            reg_covar=args.reg_covar, n_init=args.n_init, max_iter=args.max_iter,
            tol=args.tol, init_params=args.init_params,
            random_state=args.seed,
        ).fit(Xz.cpu().numpy())
        weights = np.asarray(bgmm.weights_)
        n_eff = int((weights > active_thresh).sum())
        # store ALL weights (sorted desc; max_components is small) so the cluster
        # plot's bars match n_effective, and keep the top-10 for the log line.
        sorted_w = np.sort(weights)[::-1]
        top_w = sorted_w[:min(10, len(sorted_w))]
        if not bgmm.converged_:
            print(f"[bgmm] WARNING t_eval={t_eval} did NOT converge in "
                  f"{args.max_iter} iters (n_iter={bgmm.n_iter_}); density/"
                  f"n_effective are provisional — raise --max-iter.")
        print(f"[bgmm] t_eval={t_eval} d_model={d_model} feat_dim={feat_dim} "
              f"max_K={args.max_components} n_effective={n_eff} "
              f"cov={args.covariance_type} fit_pos={X.shape[0]} "
              f"converged={bgmm.converged_} n_iter={bgmm.n_iter_} "
              f"lb={bgmm.lower_bound_:.3f} "
              f"alpha={bgmm.weight_concentration_prior_:.4g}")
        print(f"[bgmm]   top weights: "
              f"{', '.join(f'{w:.3f}' for w in top_w)}")

        @torch.no_grad()
        def perpos_bgmm(tok, chunk=16):
            """token_ids (B,L) -> per-position BGMM NLL (B,L), higher = more OOD."""
            outs = []
            for i in range(0, tok.shape[0], chunk):
                tb = tok[i:i + chunk].to(device).long()
                B, Lq = tb.shape
                t = torch.full((B,), t_eval, device=device)
                beta = torch.ones(B, Lq, model.K, device=device)
                beta.scatter_(-1, tb.unsqueeze(-1), float(t_eval))
                x_t = beta / beta.sum(-1, keepdim=True)   # deterministic (match _perpos_feats)
                h = model.get_hidden_states(x_t, t).reshape(B * Lq, d_model)
                z = (h.cpu() - mu) / sigma
                if pca_V is not None:
                    z = z @ pca_V
                nll = -bgmm.score_samples(z.numpy())      # (B*Lq,)
                outs.append(torch.from_numpy(nll).float().reshape(B, Lq))
            return torch.cat(outs, dim=0)

        pos = perpos_bgmm(pos_tok)
        pos_max = pos.max(dim=1).values.numpy()
        pos_mean = pos.mean(dim=1).numpy()
        print(f"[bgmm] t_eval={t_eval} positive NLL: per-pos mean={pos.mean():.3f} "
              f"seq-max mean={pos_max.mean():.3f}")

        rows = [{"scheme": None, "rate": 0.0, "t_eval": t_eval,
                 "n": int(pos_tok.shape[0]), "n_effective": n_eff,
                 "mean_nll": float(pos.mean()), "seq_max_mean": float(pos_max.mean())}]
        pos_tokf = pos.numpy().reshape(-1)  # clean per-token NLL: threshold calibration
        print(f"\n{'t_eval':>6} {'scheme':>10} {'rate':>5} {'AUseq_max':>10} "
              f"{'AUseq_mean':>11} {'AUtok':>8} {'tokP@5':>7} {'tokR@5':>7} {'tokF1@5':>8}")
        for scheme in schemes:
            for r in rates:
                ood_tok = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
                ood = perpos_bgmm(ood_tok)
                ood_max = ood.max(dim=1).values.numpy()
                ood_mean = ood.mean(dim=1).numpy()
                lab = np.concatenate([np.zeros(len(pos_max)), np.ones(len(ood_max))])
                au_max = _auroc(np.concatenate([pos_max, ood_max]), lab)
                au_mean = _auroc(np.concatenate([pos_mean, ood_mean]), lab)
                changed = (ood_tok != pos_tok)
                cm = changed.numpy().reshape(-1).astype(int)
                tokloc = float("nan")
                if changed.any() and (~changed).any():
                    tokloc = _auroc(ood.numpy().reshape(-1), cm)
                # --- operating-point metrics (P/R/F1 at clean-calibrated thresholds) ---
                # sequence score = MEAN-pooled per-token NLL (the primary readout).
                m_seq = det_metrics(pos_mean, pos_mean, ood_mean)
                ood_tokf = ood.numpy().reshape(-1)
                m_tok = det_metrics(pos_tokf, ood_tokf[cm == 0], ood_tokf[cm == 1])
                # WORD level — the common unit vs the BPE-tokenised LM baselines.
                w_max = word_metrics(pos, pos_tok, ood, ood_tok, changed, op="max")
                w_mean = word_metrics(pos, pos_tok, ood, ood_tok, changed, op="mean")
                p5 = m_tok["prf"].get("0.05", {})
                print(f"{t_eval:>6.2f} {scheme:>10} {r:>5.2f} {au_max:>10.4f} "
                      f"{au_mean:>11.4f} {tokloc:>8.4f} "
                      f"{p5.get('precision', float('nan')):>7.3f} "
                      f"{p5.get('recall', float('nan')):>7.3f} "
                      f"{p5.get('f1', float('nan')):>8.3f}")
                rows.append({"scheme": scheme, "rate": r, "t_eval": t_eval,
                             "n": int(ood_tok.shape[0]), "n_effective": n_eff,
                             "mean_nll": float(ood.mean()),
                             "seq_max_mean": float(ood_max.mean()),
                             # NLL-sweep-style names ("normal OOD numbers"):
                             "auroc_seq_gmm": au_mean, "auroc_seq_gmm_max": au_max,
                             "auroc_token_gmm": tokloc,
                             "prf_seq": m_seq, "prf_token": m_tok,
                             "word_max": w_max, "word_mean": w_mean,
                             "auroc_word_max": w_max["word"].get("auroc"),
                             "auroc_word_mean": w_mean["word"].get("auroc")})
        return rows, {"n_effective": n_eff, "converged": bool(bgmm.converged_),
                      "n_iter": int(bgmm.n_iter_),
                      "weights": [float(w) for w in sorted_w]}, perpos_bgmm

    all_rows: list[dict] = []
    per_t_meta: dict[str, dict] = {}
    scorer_by_t = {}
    for t_eval in t_evals:
        rows, meta, scorer = run_one_teval(t_eval)
        all_rows.extend(rows)
        per_t_meta[f"{t_eval}"] = meta
        scorer_by_t[t_eval] = scorer

    # ---- healing-style per-token examples, scored at the BEST-performing t -------
    # Show clean vs replace vs FALSEINFO on the same passages; annotate each token
    # with the detector score, the true-change mask (^) and a calibrated flag mask
    # (*), mirroring scripts/heal_dirichlet.py's clean/corr/flag example dump.
    example_t = args.example_t if args.example_t is not None else max(t_evals)
    ex_scorer = scorer_by_t[example_t]
    ex_by_len = by_len or build_vocab_by_len(_decode_tokens(fit_tok))
    # flag threshold from CLEAN scores at the target FPR (heal-style calibration)
    clean_all = ex_scorer(pos_tok).numpy().reshape(-1)
    flag_thr = float(np.quantile(clean_all, 1.0 - args.example_fpr))
    print(f"\n=== healing-style examples @ t_eval={example_t} "
          f"(flag thr={flag_thr:.3f} @ fpr={args.example_fpr}) ===")

    mp = args.max_pos
    ex_tok = pos_tok[:2]
    variants = [
        ("clean", ex_tok.clone()),
        ("replace30", corrupt_token_ids(ex_tok.clone(), vocab_size=K,
                                        corrupt_rate=0.3, seed=7)),
        ("falseinfo30", corrupt_false_info(ex_tok.clone(), 0.3, ex_by_len, seed=7)),
    ]
    examples = []
    for tag, tk in variants:
        sc = ex_scorer(tk)
        changed = (tk != ex_tok)
        flagged = sc > flag_thr
        for b in range(tk.shape[0]):
            clean_txt = "".join(_ALPH[int(i)] for i in ex_tok[b][:mp].tolist())
            corr_txt = "".join(_ALPH[int(i)] for i in tk[b][:mp].tolist())
            ch = changed[b][:mp]
            fl = flagged[b][:mp]
            examples.append({
                "which": tag, "idx": b, "t_eval": example_t, "flag_thr": flag_thr,
                "clean_text": clean_txt, "text": corr_txt,
                "GMMNLL_t": [round(float(x), 4) for x in sc[b][:mp].tolist()],
                "corrupted": [bool(x) for x in ch.tolist()],
                "flagged": [bool(x) for x in fl.tolist()],
            })
            if b == 0:  # heal-style ASCII dump for the first passage of each variant
                print(f"\n--- {tag} #{b} ---")
                print(f"  clean : {clean_txt!r}")
                if tag != "clean":
                    print(f"  corr  : {corr_txt!r}")
                    print(f"  truec : {''.join('^' if x else ' ' for x in ch.tolist())}")
                print(f"  flag  : {''.join('*' if x else ' ' for x in fl.tolist())}")
                tp = int((fl & ch).sum())
                fp = int((fl & ~ch).sum())
                fn = int((~fl & ch).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                print(f"  flagged P={prec:.2f} R={rec:.2f} (tp={tp} fp={fp} fn={fn})")

    out_path = Path(args.out or (Path(args.ckpt).parent / "bgmm_perpos_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "split": args.split, "t_evals": t_evals,
        "max_components": args.max_components,
        "weight_concentration_prior": args.weight_conc_prior,
        "active_thresh": active_thresh,
        "covariance_type": args.covariance_type,
        "pca_dim": args.pca_dim, "reg_covar": args.reg_covar,
        "n_effective_per_t": per_t_meta,
        "detector": "PerPosBGMM",
        "detector_long": "per-position Bayesian (Dirichlet-process) infinite-mixture "
                         "density; score = negative log-likelihood; effective cluster "
                         "count inferred by the variational posterior",
        "rows": all_rows, "examples": examples,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())