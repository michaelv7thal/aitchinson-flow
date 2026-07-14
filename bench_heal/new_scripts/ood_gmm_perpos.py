"""Per-position GAUSSIAN-MIXTURE OOD detector ("PerPosGMM") on DirichletFM.

A density generalisation of the single-Gaussian Mahalanobis detector
(``scripts/ood_mahalanobis_perpos.py``). Fit a ``GaussianMixture`` to IN-DISTRIBUTION
per-position DirichletFM features and score each token by its negative
log-likelihood under the mixture

    score(z) = -log p_GMM(z),   p_GMM(z) = sum_k pi_k N(z; mu_k, Sigma_k).

MOTIVATION. The current best per-token localizer is the training-free denoiser NLL
(``scripts/ood_denoiser_nll.py``). It reads *local* character surprise, so it detects
char noise well but MISSES "false information" — a word replaced by a *different real,
same-length* word (lexically valid, semantically wrong): every character still spells
valid English, so the char-NLL barely moves (see ``RESULTS_HEAL_POC_INSULIN.md``:
Track C recall 0.20, net/corrupt -0.142). A GMM density over the DirichletFM
*contextual backbone features* may flag such swaps if they land in a low-density
region of the ID feature manifold. A mixture (vs one Gaussian) captures the
multi-modal structure of valid English contextual features, so an off-manifold
"wrong-word-in-context" point scores as low density more sharply.

The features are the SAME deterministic Dirichlet-mean backbone features every other
per-position detector reads (``ood_variance_perpos._perpos_feats`` at ``t_eval``);
the determinism is load-bearing (a stochastic Dirichlet sample collapses
clean-vs-corrupt features). Because at high ``t_eval`` the mean input is strongly
token-dominated (a valid same-length swap may land ON the ID token manifold), the
detector supports a ``--t-evals`` sweep: lower t forces ``get_hidden_states`` to lean
on context, which is where an off-manifold swap should show up. If the GMM misses
false-info across all t, that is itself a clean result ("feature-density does not
fact-check either").

DISTINCT detector from PerPosMaha (single Gaussian), HingeSVGP (trained mean/prob)
and PerPosVarGP (GP variance) — different score, different name, different output
(gmm_perpos_sweep.json). DirichletFM == DirichletFlowMatching (Stark et al. 2024).
NOT the Discrete-FM 'DFM' arm; the loader guards against it.

Usage:
    python scripts/ood_gmm_perpos.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt \
        --split test --n-components 8 --covariance-type diag --pca-dim 64 \
        --schemes replace,shuffle,falseinfo --t-evals 1.5,3.0,4.5,6.0 \
        --out ood_out/gmm/gmm_perpos_sweep.json
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
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
)
from sklearn.mixture import GaussianMixture  # noqa: E402

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
                         "Refits PCA+GMM and reruns the ladder per t; records t_eval "
                         "per row. Lower t = more context-dependent features.")
    ap.add_argument("--fit-seqs", type=int, default=512,
                    help="# in-distribution sequences to fit the mixture")
    ap.add_argument("--n", type=int, default=256, help="# eval sequences per split")
    ap.add_argument("--max-fit-pos", type=int, default=80000,
                    help="cap on # per-position vectors used to fit the GMM")
    # --- GMM / feature knobs ---
    ap.add_argument("--n-components", type=int, default=8)
    ap.add_argument("--covariance-type", choices=["diag", "full", "tied", "spherical"],
                    default="diag",
                    help="GMM covariance. 'diag' (default) is the cheap, stable choice "
                         "for high-d; use 'full' only after PCA to a small --pca-dim.")
    ap.add_argument("--pca-dim", type=int, default=64,
                    help="0 = full-d_model GMM. >0 truncates to top-k PCs of the "
                         "standardised ID cloud before fitting (recommended for 'full').")
    ap.add_argument("--reg-covar", type=float, default=1e-4,
                    help="added to the covariance diagonal (GMM singular-cov guard).")
    ap.add_argument("--n-init", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=100)
    # --- corruption ladder ---
    ap.add_argument("--schemes", type=str, default="replace,shuffle,falseinfo")
    ap.add_argument("--rates", type=str, default="0.1,0.3,0.5,0.7,1.0")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="dataset split for fit/eval sequences "
                         "(test = held-out last-5M text8 split)")
    ap.add_argument("--seed", type=int, default=42)
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
        print(f"[gmm] false-info vocab: lengths {min(by_len)}-{max(by_len)} "
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

    def run_one_teval(t_eval: float) -> list[dict]:
        """Fit PCA+GMM at t_eval and score the full corruption ladder."""
        fit_feats = _perpos_feats(model, fit_tok, t_eval, device)        # (Nf,L,d)
        d_model = fit_feats.shape[-1]
        X = fit_feats.reshape(-1, d_model)
        if X.shape[0] > args.max_fit_pos:
            X = X[torch.randperm(X.shape[0])[: args.max_fit_pos]]
        # standardise on the ID cloud (raw activation scales vary by orders of
        # magnitude — a GMM likelihood on unscaled features is numerically poor).
        mu = X.mean(0)
        sigma = X.std(0).clamp_min(1e-6)
        Xz = (X - mu) / sigma
        pca_V = None
        if args.pca_dim and args.pca_dim < d_model:
            _, _, V = torch.pca_lowrank(Xz, q=min(args.pca_dim, Xz.shape[1]))
            pca_V = V[:, : args.pca_dim]
            Xz = Xz @ pca_V
        feat_dim = Xz.shape[1]
        gmm = GaussianMixture(
            n_components=args.n_components, covariance_type=args.covariance_type,
            reg_covar=args.reg_covar, n_init=args.n_init, max_iter=args.max_iter,
            random_state=args.seed,
        ).fit(Xz.cpu().numpy())
        print(f"[gmm] t_eval={t_eval} d_model={d_model} feat_dim={feat_dim} "
              f"K_comp={args.n_components} cov={args.covariance_type} "
              f"fit_pos={X.shape[0]} converged={gmm.converged_} "
              f"lb={gmm.lower_bound_:.3f}")

        @torch.no_grad()
        def perpos_gmm(tok, chunk=16):
            """token_ids (B,L) -> per-position GMM NLL (B,L), higher = more OOD."""
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
                nll = -gmm.score_samples(z.numpy())       # (B*Lq,)
                outs.append(torch.from_numpy(nll).float().reshape(B, Lq))
            return torch.cat(outs, dim=0)

        pos = perpos_gmm(pos_tok)
        pos_max = pos.max(dim=1).values.numpy()
        pos_mean = pos.mean(dim=1).numpy()
        print(f"[gmm] t_eval={t_eval} positive NLL: per-pos mean={pos.mean():.3f} "
              f"seq-max mean={pos_max.mean():.3f}")

        rows = [{"scheme": None, "rate": 0.0, "t_eval": t_eval,
                 "n": int(pos_tok.shape[0]), "mean_nll": float(pos.mean()),
                 "seq_max_mean": float(pos_max.mean())}]
        print(f"\n{'t_eval':>6} {'scheme':>10} {'rate':>5} {'AUROC_max':>10} "
              f"{'AUROC_mean':>11} {'tokloc_AUROC':>13}")
        for scheme in schemes:
            for r in rates:
                ood_tok = _corrupt(pos_tok.clone(), scheme, r, args.seed + int(1000 * r))
                ood = perpos_gmm(ood_tok)
                ood_max = ood.max(dim=1).values.numpy()
                ood_mean = ood.mean(dim=1).numpy()
                lab = np.concatenate([np.zeros(len(pos_max)), np.ones(len(ood_max))])
                au_max = _auroc(np.concatenate([pos_max, ood_max]), lab)
                au_mean = _auroc(np.concatenate([pos_mean, ood_mean]), lab)
                changed = (ood_tok != pos_tok)
                tokloc = float("nan")
                if changed.any() and (~changed).any():
                    tokloc = _auroc(ood.numpy().reshape(-1),
                                    changed.numpy().reshape(-1).astype(int))
                print(f"{t_eval:>6.2f} {scheme:>10} {r:>5.2f} {au_max:>10.4f} "
                      f"{au_mean:>11.4f} {tokloc:>13.4f}")
                rows.append({"scheme": scheme, "rate": r, "t_eval": t_eval,
                             "n": int(ood_tok.shape[0]), "mean_nll": float(ood.mean()),
                             "seq_max_mean": float(ood_max.mean()),
                             "auroc_gmm_max": au_max, "auroc_gmm_mean": au_mean,
                             "auroc_token_localization": tokloc})
        return rows

    all_rows: list[dict] = []
    for t_eval in t_evals:
        all_rows.extend(run_one_teval(t_eval))

    out_path = Path(args.out or (Path(args.ckpt).parent / "gmm_perpos_sweep.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "split": args.split, "t_evals": t_evals,
        "n_components": args.n_components, "covariance_type": args.covariance_type,
        "pca_dim": args.pca_dim, "reg_covar": args.reg_covar,
        "detector": "PerPosGMM",
        "detector_long": "per-position Gaussian-mixture density; score = negative log-likelihood",
        "rows": all_rows,
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
