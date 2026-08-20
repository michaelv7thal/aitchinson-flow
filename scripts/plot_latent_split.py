"""Visualize the valid-vs-invalid split in DirichletFM's latent space.

Scatter plots of the backbone's DETERMINISTIC Dirichlet-mean features (the exact
representation the OOD heads read, ``_perpos_feats`` at ``t_eval``) for CLEAN vs
CORRUPTED text, color-coded, so you can *see* how well/poorly the two classes
separate — and crucially WHERE the separation lives:

  * PCA (unsupervised)  — top-2 principal axes of the clean cloud. Honest
    geometry: if the classes overlap here, the split is NOT in the dominant
    variance directions.
  * LDA / energy axis (supervised LINEAR) — the best single linear separating
    direction (the thing a linear BLR/hinge head exploits) vs the leading
    orthogonal PC. Separation here ⇒ the split is linearly accessible.
  * t-SNE (nonlinear, exploratory) — can manufacture clusters; shown only to
    check whether any *extra* nonlinear structure exists beyond the linear axis.

Two granularities:
  * per-sequence — one point per sequence (mean-pooled standardized feature);
    swept over replace AND shuffle corruption.
  * per-token   — clean tokens vs replaced tokens (the localization view);
    PCA top-PCs typically encode char identity, so validity sits off-axis and
    the trained energy axis is what splits it.

Each panel is annotated with a probe AUROC (how separable the shown view is) and
a full-dim linear ceiling, so "how well" is quantified, not just eyeballed.
Disjoint fit/eval splits: directions are fit on a fit set, scatter shows held-out
eval points (no memorization).

Usage:
    uv run python scripts/plot_latent_split.py \
        --ckpt runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_final.pt \
        --out-dir ood_out/latent_split
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

from scripts.ood_variance_perpos import _load_dirichletfm, _perpos_feats  # noqa: E402
from aitchinson_flow.training import build_training_datamodule  # noqa: E402
from aitchinson_flow.data.corruption import (  # noqa: E402
    corrupt_token_ids,
    partially_shuffle_token_ids,
    corrupt_false_info,
    build_vocab_by_len,
)

_ALPH = "abcdefghijklmnopqrstuvwxyz "  # text8 K=27 decode (for the falseinfo vocab)

VALID_C, INVALID_C = "#1f77b4", "#d62728"  # blue = valid, red = invalid


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if (labels == 1).sum() < 2 or (labels == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _probe_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """AUROC of a logistic-regression probe (5-fold-ish: fit on all, score all —
    a separability *ceiling* for the given representation)."""
    from sklearn.linear_model import LogisticRegression

    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return _auroc(clf.decision_function(X), y)


def _probe_dir(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit weight vector of the SAME probe _probe_auroc scores.

    Refits with identical data and hyper-parameters (lbfgs is deterministic), so
    the returned direction is the one whose AUROC is reported. Used only to give
    the full-dimension probe a 2-D panel: its decision axis against the leading
    orthogonal PC, the same display the LDA / energy axes get.
    """
    from sklearn.linear_model import LogisticRegression

    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return np.zeros(X.shape[1])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    w = clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-12)


_WORDSWAP = ("falseinfo", "wordswap", "plausible")


def _probe_null(X, y, n_perm, seed):
    """Mean in-sample probe AUROC under LABEL PERMUTATION.

    The probes below are fit on the points they score, so their value is biased
    upward by however much an unconstrained linear fit can memorise at this (n,
    d) and feature covariance. Refitting on shuffled labels measures exactly that
    bias: it is what the estimator returns when there is nothing to find. Per
    sequence n=512 against d=1280, where any labelling is linearly separable, so
    this is the number the reported AUROC has to be read against.
    """
    if n_perm <= 0:
        return None
    rng = np.random.default_rng(seed)
    return float(np.mean([_probe_auroc(X, rng.permutation(y))
                          for _ in range(n_perm)]))


def _corrupt(tok, scheme, rate, K, seed, by_len=None, *, model=None, device=None,
             t_nll=3.0, n_cands=48):
    if scheme == "replace":
        return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
    if scheme in ("falseinfo", "wordswap"):
        if by_len is None:
            raise ValueError("falseinfo corruption needs a by_len vocab")
        return corrupt_false_info(tok, rate, by_len, seed=seed)
    if scheme == "plausible":
        # Same slots as falseinfo (shared RNG order), but each is filled with the
        # same-length real word the MODEL itself scores as most fluent — the
        # adversarial case of the corruption ladder. Reuses the swap of the
        # plausible-vs-random experiment verbatim so the two agree by construction.
        if by_len is None or model is None:
            raise ValueError("plausible corruption needs a by_len vocab and the model")
        import time

        from scripts.ood_plausible_swap import plausible_swap

        t0 = time.time()
        out, n_sw = plausible_swap(model, tok, rate, by_len, t_nll=t_nll, K=K,
                                   device=device, n_cands=n_cands, seed=seed)
        # multi-hour stage: log the cost so a ladder over rates can be budgeted
        print(f"[plausible] {tok.shape[0]} seqs @ rate {rate}: {n_sw} slots, "
              f"{time.time() - t0:.0f}s", flush=True)
        return out.cpu()
    out = corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    return partially_shuffle_token_ids(out, shuffle_rate=rate, seed=seed + 1)


def _lda_dir(Zc: np.ndarray, Zk: np.ndarray) -> np.ndarray:
    """Unit Fisher-LDA direction separating clean (Zc) from corrupt (Zk)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X = np.concatenate([Zc, Zk], 0)
    y = np.r_[np.zeros(len(Zc)), np.ones(len(Zk))]
    lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage="auto")
    lda.fit(X, y)
    w = lda.scalings_[:, 0].astype(np.float64)
    w = w / (np.linalg.norm(w) + 1e-12)
    # orient so INVALID (corrupt) projects higher than VALID -> AUROC >= 0.5
    if (Zk @ w).mean() < (Zc @ w).mean():
        w = -w
    return w


def _orth_pc(Z_fit: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Leading PC of Z_fit AFTER removing the component along ``axis`` (unit)."""
    R = Z_fit - np.outer(Z_fit @ axis, axis)
    R = R - R.mean(0)
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    pc = Vt[0]
    return pc / (np.linalg.norm(pc) + 1e-12)


def _scatter(ax, xy, lab, title, xlabel, ylabel):
    m = lab == 0
    ax.scatter(xy[m, 0], xy[m, 1], s=7, c=VALID_C, alpha=0.45, lw=0, label="valid")
    ax.scatter(xy[~m, 0], xy[~m, 1], s=7, c=INVALID_C, alpha=0.45, lw=0, label="invalid")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="bench_ood_final/latent_split")
    ap.add_argument("--t-eval", type=float, default=None)
    ap.add_argument("--fit-seqs", type=int, default=192)
    ap.add_argument("--n", type=int, default=256, help="# eval seqs per split")
    ap.add_argument("--rate", type=float, default=0.3, help="corruption rate shown")
    ap.add_argument("--schemes", type=str, default="replace,shuffle,falseinfo",
                    help="comma list of per-sequence corruption schemes to show as "
                         "rows (e.g. replace,shuffle,falseinfo). 'falseinfo' = "
                         "lexically-valid same-length word swap (the hard semantic axis).")
    ap.add_argument("--token-scheme", type=str, default="falseinfo",  # fig:latent-token reads the falseinfo panel
                    help="corruption scheme(s) for the per-token figure, comma list "
                         "(replace, shuffle, falseinfo, plausible). Each becomes one "
                         "row with its own energy-hinge head, so the per-token view "
                         "covers the same ladder as the per-sequence one.")
    ap.add_argument("--t-nll", type=float, default=3.0,
                    help="path time at which the plausible swap scores its candidates "
                         "(the denoiser-NLL detector's own t)")
    ap.add_argument("--n-cands", type=int, default=48,
                    help="candidate words per slot for the plausible swap")
    ap.add_argument("--perm-null", type=int, default=0, metavar="N",
                    help="also report each in-sample probe's label-permutation "
                         "null, averaged over N shuffles (0 = off). This is the "
                         "value the probe returns with no signal present, which "
                         "the reported AUROC must be read against")
    ap.add_argument("--no-tsne", action="store_true",
                    help="skip the exploratory t-SNE column (it is not read by any "
                         "paper figure and dominates the wall time on 4 schemes)")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--head-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="dataset split for fit/eval sequences (test = held-out "
                         "last-5M text8; still IN-DISTRIBUTION — clean text8 is the "
                         "valid negative, only corruption is OOD)")
    ap.add_argument("--tsne-n", type=int, default=1200, help="points/class for t-SNE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-coords", action="store_true",
                    help="also save the plotted 2-D coordinates + labels to "
                         "latent_split_coords.npz (for external re-plotting)")
    args = ap.parse_args()
    coords: dict[str, np.ndarray] = {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_dirichletfm(args.ckpt, device)
    K = cfg.text8_dataset.K
    t_eval = float(args.t_eval if args.t_eval is not None else cfg.dfm_svgp.t_eval)
    dm, _ = build_training_datamodule(cfg)
    _loaders = {"train": dm.train_dataloader, "val": dm.val_dataloader,
                "test": dm.test_dataloader}
    vl = _loaders[args.split]() or dm.train_dataloader()

    seqs = []
    for b in vl:
        seqs.append(b["token_ids"].long())
        if sum(s.shape[0] for s in seqs) >= args.fit_seqs + args.n:
            break
    seqs = torch.cat(seqs)
    fit_tok = seqs[: args.fit_seqs]
    eval_tok = seqs[args.fit_seqs : args.fit_seqs + args.n]
    print(f"[latent] ckpt={args.ckpt}\n[latent] t_eval={t_eval} K={K} "
          f"fit={fit_tok.shape[0]} eval={eval_tok.shape[0]} rate={args.rate}")

    schemes = [s for s in args.schemes.split(",") if s.strip()]
    tschemes = [s for s in args.token_scheme.split(",") if s.strip()]
    # false-info replacement vocab (built once from the fit windows) if needed
    by_len = None
    if any(s in _WORDSWAP for s in schemes + tschemes):
        fit_txt = " ".join(
            "".join(_ALPH[int(i)] if int(i) < len(_ALPH) else "?" for i in row)
            for row in fit_tok
        )
        by_len = build_vocab_by_len(fit_txt)
        print(f"[latent] false-info vocab: lengths {min(by_len)}-{max(by_len)} "
              f"(e.g. len-6: {by_len.get(6, [])[:5]})")

    # ---- standardiser from clean per-token feats (matches the detectors) ----
    fc = _perpos_feats(model, fit_tok, t_eval, device)  # (Nf,L,d)
    d = fc.shape[-1]
    mu = fc.reshape(-1, d).mean(0)
    sigma = fc.reshape(-1, d).std(0).clamp_min(1e-6)

    def ztok(tok):  # token_ids -> standardized per-token feats z (B,L,d) numpy
        h = _perpos_feats(model, tok, t_eval, device)
        z = ((h - mu) / sigma)
        return z.numpy()

    def zseq(tok):  # -> mean-pooled standardized seq feats (B,d) numpy
        return ztok(tok).mean(1)

    def corrupt(tok, scheme, rate, seed):  # one call site for every scheme
        return _corrupt(tok, scheme, rate, K, seed, by_len=by_len, model=model,
                        device=device, t_nll=args.t_nll, n_cands=args.n_cands)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    metrics = {"ckpt": args.ckpt, "split": args.split, "t_eval": t_eval,
               "d_model": d, "rate": args.rate, "fit_seqs": args.fit_seqs,
               "n": args.n, "schemes": schemes, "token_scheme": args.token_scheme,
               "token_schemes": tschemes, "per_sequence": {}, "per_token": {}}

    # ===================== FIGURE 1 — per-SEQUENCE ==========================
    Zc_fit = zseq(fit_tok)                                    # clean fit seqs
    Zc_ev = zseq(eval_tok)                                    # clean eval seqs
    nrows = len(schemes)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.5, 4.3 * nrows), squeeze=False)
    for row, scheme in enumerate(schemes):
        Zk_fit = zseq(corrupt(fit_tok.clone(), scheme, args.rate, args.seed + 1))
        Zk_ev = zseq(corrupt(eval_tok.clone(), scheme, args.rate, args.seed + 2))
        Xev = np.concatenate([Zc_ev, Zk_ev], 0)
        yev = np.r_[np.zeros(len(Zc_ev)), np.ones(len(Zk_ev))]

        # -- PCA (unsupervised), fit on clean fit cloud --
        cmu = Zc_fit.mean(0)
        _, _, Vt = np.linalg.svd(Zc_fit - cmu, full_matrices=False)
        pcs = Vt[:2]
        xy = (Xev - cmu) @ pcs.T
        au_pca = _probe_auroc(xy, yev)
        _scatter(axes[row, 0], xy, yev,
                 f"{scheme}@{args.rate}  PCA (unsupervised)\nPC1–2 probe AUROC={au_pca:.3f}",
                 "PC1", "PC2")

        # -- LDA axis (supervised linear) vs orthogonal PC --
        wl = _lda_dir(Zc_fit, Zk_fit)
        ortho = _orth_pc(np.concatenate([Zc_fit, Zk_fit], 0), wl)
        xax = Xev @ wl
        yax = Xev @ ortho
        au_lda = _auroc(xax, yev)
        au_full = _probe_auroc(Xev, yev)
        _scatter(axes[row, 1], np.c_[xax, yax], yev,
                 f"{scheme}@{args.rate}  LDA axis (supervised linear)\n"
                 f"LDA-axis AUROC={au_lda:.3f}  (full-dim ceiling={au_full:.3f})",
                 "LDA separating axis", "leading orthogonal PC")

        # -- t-SNE (nonlinear) --
        if args.no_tsne:
            axes[row, 2].axis("off")
        else:
            nse = min(args.tsne_n, len(Zc_ev))
            Xt = np.concatenate([Zc_ev[:nse], Zk_ev[:nse]], 0)
            yt = np.r_[np.zeros(nse), np.ones(nse)]
            emb = TSNE(n_components=2, perplexity=30, init="pca",
                       random_state=args.seed).fit_transform(Xt)
            _scatter(axes[row, 2], emb, yt,
                     f"{scheme}@{args.rate}  t-SNE (nonlinear, exploratory)",
                     "t-SNE 1", "t-SNE 2")
        if args.dump_coords:
            # Full-dimension probe: its own decision axis vs the leading
            # orthogonal PC, so the third readout can be shown and not only
            # tabulated. In sample by construction, like the AUROC above it.
            wf = _probe_dir(Xev, yev)
            coords[f"seq_{scheme}_pca_xy"] = xy
            coords[f"seq_{scheme}_lda_xy"] = np.c_[xax, yax]
            coords[f"seq_{scheme}_full_xy"] = np.c_[Xev @ wf,
                                                    Xev @ _orth_pc(Xev, wf)]
            coords[f"seq_{scheme}_labels"] = yev
        axes[row, 0].legend(fontsize=7, markerscale=1.6, loc="best")
        metrics["per_sequence"][scheme] = {
            "pca_pc12_probe_auroc": au_pca, "lda_axis_auroc": au_lda,
            "full_dim_linear_auroc": au_full,
            "pca_pc12_probe_null": _probe_null(xy, yev, args.perm_null, args.seed),
            "full_dim_linear_null": _probe_null(Xev, yev, args.perm_null, args.seed),
            "n_points": int(len(yev))}
        print(f"[seq:{scheme}] PCA(PC1-2)={au_pca:.3f}  LDA-axis={au_lda:.3f}  "
              f"full-linear={au_full:.3f}  "
              f"nulls={metrics['per_sequence'][scheme]['pca_pc12_probe_null']} / "
              f"{metrics['per_sequence'][scheme]['full_dim_linear_null']}")
    fig.suptitle(
        f"Latent split — PER-SEQUENCE (mean-pooled DirichletFM features @ t={t_eval}); "
        f"blue=valid, red=invalid", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = out / "latent_split_sequence.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    print(f"Wrote {p1}")

    # ===================== FIGURE 2 — per-TOKEN, one row per scheme =========
    # Each scheme gets its OWN energy-hinge head (BayesLinHead recipe), trained on
    # the fit set against that scheme's negatives, so the axis shown is the one a
    # detector for that corruption would actually use. The clean features are
    # scheme-independent and computed once.
    zc_fit = ztok(fit_tok).reshape(-1, d)
    zc_ev = ztok(eval_tok).reshape(-1, d)
    fig2, ax2 = plt.subplots(len(tschemes), 3,
                             figsize=(13.5, 4.5 * len(tschemes)), squeeze=False)
    for row, tscheme in enumerate(tschemes):
        corr_fit = corrupt(fit_tok.clone(), tscheme, 0.5, args.seed)
        zk_fit_all = ztok(corr_fit).reshape(-1, d)
        yk_fit = (corr_fit != fit_tok).reshape(-1).numpy()
        head = nn.Linear(d, 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        zpos = torch.tensor(np.concatenate([zc_fit, zk_fit_all[yk_fit == 0]], 0),
                            device=device)
        zneg = torch.tensor(zk_fit_all[yk_fit == 1], device=device)
        for step in range(args.head_steps):
            opt.zero_grad()
            loss = head(zpos).squeeze(-1).pow(2).mean() + \
                torch.relu(args.margin - head(zneg).squeeze(-1)).mean()
            loss.backward()
            opt.step()
        w_energy = head.weight.detach().cpu().numpy()[0]
        w_energy = w_energy / (np.linalg.norm(w_energy) + 1e-12)

        # Eval per-token: clean tokens vs CORRUPTED tokens (balanced sample).
        corr_ev = corrupt(eval_tok.clone(), tscheme, args.rate, args.seed + 5)
        zk_ev = ztok(corr_ev).reshape(-1, d)
        chg = (corr_ev != eval_tok).reshape(-1).numpy()
        rng = np.random.default_rng(args.seed)   # per row: order-independent draws
        n_inv = int(chg.sum())
        n_show = min(n_inv, 4000)
        inv_idx = rng.choice(np.where(chg)[0], n_show, replace=False)
        val_idx = rng.choice(zc_ev.shape[0], n_show, replace=False)
        Zt = np.concatenate([zc_ev[val_idx], zk_ev[inv_idx]], 0)
        yt = np.r_[np.zeros(n_show), np.ones(n_show)]

        # PCA per-token (fit on clean tokens)
        cmu = zc_ev.mean(0)
        _, _, Vt = np.linalg.svd(zc_ev[val_idx] - cmu, full_matrices=False)
        xy = (Zt - cmu) @ Vt[:2].T
        au_pca = _probe_auroc(xy, yt)
        _scatter(ax2[row, 0], xy, yt,
                 f"{tscheme}@{args.rate}  per-token PCA (unsupervised)\n"
                 f"PC1–2 probe AUROC={au_pca:.3f}", "PC1", "PC2")
        ax2[row, 0].legend(fontsize=7, markerscale=1.6)
        # Energy-hinge axis (the detector's actual axis) vs orthogonal PC
        ortho = _orth_pc(Zt, w_energy)
        xax, yax = Zt @ w_energy, Zt @ ortho
        au_e = _auroc(xax, yt)
        au_full = _probe_auroc(Zt, yt)
        _scatter(ax2[row, 1], np.c_[xax, yax], yt,
                 f"{tscheme}@{args.rate}  per-token energy-hinge axis (supervised)\n"
                 f"energy AUROC={au_e:.3f}  (full-dim ceiling={au_full:.3f})",
                 "trained energy axis  E=w·z", "leading orthogonal PC")
        if args.dump_coords:
            wf = _probe_dir(Zt, yt)
            coords[f"tok_{tscheme}_pca_xy"] = xy
            coords[f"tok_{tscheme}_energy_xy"] = np.c_[xax, yax]
            coords[f"tok_{tscheme}_full_xy"] = np.c_[Zt @ wf,
                                                     Zt @ _orth_pc(Zt, wf)]
            coords[f"tok_{tscheme}_labels"] = yt
        # t-SNE per-token
        if args.no_tsne:
            ax2[row, 2].axis("off")
        else:
            nse = min(args.tsne_n, n_show)
            emb = TSNE(n_components=2, perplexity=30, init="pca",
                       random_state=args.seed).fit_transform(
                np.concatenate([zc_ev[val_idx][:nse], zk_ev[inv_idx][:nse]], 0))
            _scatter(ax2[row, 2], emb, np.r_[np.zeros(nse), np.ones(nse)],
                     f"{tscheme}@{args.rate}  per-token t-SNE (exploratory)",
                     "t-SNE 1", "t-SNE 2")
        metrics["per_token"][tscheme] = {
            "token_scheme": tscheme, "pca_pc12_probe_auroc": au_pca,
            "energy_axis_auroc": au_e, "full_dim_linear_auroc": au_full,
            "pca_pc12_probe_null": _probe_null(xy, yt, args.perm_null, args.seed),
            "full_dim_linear_null": _probe_null(Zt, yt, args.perm_null, args.seed),
            "n_per_class": n_show}
        print(f"[token:{tscheme}] PCA(PC1-2)={au_pca:.3f}  energy-axis={au_e:.3f}  "
              f"full-linear={au_full:.3f}  "
              f"nulls={metrics['per_token'][tscheme]['pca_pc12_probe_null']} / "
              f"{metrics['per_token'][tscheme]['full_dim_linear_null']}")
    fig2.suptitle(
        f"Latent split — PER-TOKEN @{args.rate} (clean tokens vs corrupted "
        f"tokens); blue=valid, red=invalid", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = out / f"latent_split_token_{'-'.join(tschemes)}.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    print(f"Wrote {p2}")

    (out / "latent_split.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out / 'latent_split.json'}")
    if args.dump_coords:
        np.savez_compressed(out / "latent_split_coords.npz", **coords)
        print(f"Wrote {out / 'latent_split_coords.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
