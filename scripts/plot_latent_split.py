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


def _corrupt(tok, scheme, rate, K, seed, by_len=None):
    if scheme == "replace":
        return corrupt_token_ids(tok, vocab_size=K, corrupt_rate=rate, seed=seed)
    if scheme == "shuffle":
        return partially_shuffle_token_ids(tok, shuffle_rate=rate, seed=seed)
    if scheme in ("falseinfo", "wordswap"):
        if by_len is None:
            raise ValueError("falseinfo corruption needs a by_len vocab")
        return corrupt_false_info(tok, rate, by_len, seed=seed)
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
    ap.add_argument("--out-dir", default="ood_out/latent_split")
    ap.add_argument("--t-eval", type=float, default=None)
    ap.add_argument("--fit-seqs", type=int, default=192)
    ap.add_argument("--n", type=int, default=256, help="# eval seqs per split")
    ap.add_argument("--rate", type=float, default=0.3, help="corruption rate shown")
    ap.add_argument("--schemes", type=str, default="replace,shuffle",
                    help="comma list of per-sequence corruption schemes to show as "
                         "rows (e.g. replace,shuffle,falseinfo). 'falseinfo' = "
                         "lexically-valid same-length word swap (the hard semantic axis).")
    ap.add_argument("--token-scheme", type=str, default="replace",
                    help="corruption scheme for the per-token figure (replace or "
                         "falseinfo). falseinfo shows whether the swapped-word tokens "
                         "separate from clean tokens in feature space.")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--head-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test",
                    help="dataset split for fit/eval sequences (test = held-out "
                         "last-5M text8; still IN-DISTRIBUTION — clean text8 is the "
                         "valid negative, only corruption is OOD)")
    ap.add_argument("--tsne-n", type=int, default=1200, help="points/class for t-SNE")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
    # false-info replacement vocab (built once from the fit windows) if needed
    by_len = None
    if any(s in ("falseinfo", "wordswap")
           for s in schemes + [args.token_scheme]):
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    metrics = {"ckpt": args.ckpt, "split": args.split, "t_eval": t_eval,
               "d_model": d, "rate": args.rate, "fit_seqs": args.fit_seqs,
               "n": args.n, "schemes": schemes, "token_scheme": args.token_scheme,
               "per_sequence": {}, "per_token": {}}

    # ===================== FIGURE 1 — per-SEQUENCE ==========================
    Zc_fit = zseq(fit_tok)                                    # clean fit seqs
    Zc_ev = zseq(eval_tok)                                    # clean eval seqs
    nrows = len(schemes)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.5, 4.3 * nrows), squeeze=False)
    for row, scheme in enumerate(schemes):
        Zk_fit = zseq(_corrupt(fit_tok.clone(), scheme, args.rate, K, args.seed + 1,
                               by_len=by_len))
        Zk_ev = zseq(_corrupt(eval_tok.clone(), scheme, args.rate, K, args.seed + 2,
                              by_len=by_len))
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
        nse = min(args.tsne_n, len(Zc_ev))
        Xt = np.concatenate([Zc_ev[:nse], Zk_ev[:nse]], 0)
        yt = np.r_[np.zeros(nse), np.ones(nse)]
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=args.seed).fit_transform(Xt)
        _scatter(axes[row, 2], emb, yt,
                 f"{scheme}@{args.rate}  t-SNE (nonlinear, exploratory)",
                 "t-SNE 1", "t-SNE 2")
        axes[row, 0].legend(fontsize=7, markerscale=1.6, loc="best")
        metrics["per_sequence"][scheme] = {
            "pca_pc12_probe_auroc": au_pca, "lda_axis_auroc": au_lda,
            "full_dim_linear_auroc": au_full}
        print(f"[seq:{scheme}] PCA(PC1-2)={au_pca:.3f}  LDA-axis={au_lda:.3f}  "
              f"full-linear={au_full:.3f}")
    fig.suptitle(
        f"Latent split — PER-SEQUENCE (mean-pooled DirichletFM features @ t={t_eval}); "
        f"blue=valid, red=invalid", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = out / "latent_split_sequence.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    print(f"Wrote {p1}")

    # ===================== FIGURE 2 — per-TOKEN (replace) ===================
    # Train the real energy-hinge head (BayesLinHead recipe) on the fit set.
    tscheme = args.token_scheme
    zc_fit = ztok(fit_tok).reshape(-1, d)
    corr_fit = _corrupt(fit_tok.clone(), tscheme, 0.5, K, args.seed, by_len=by_len)
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

    # Eval per-token: clean tokens vs REPLACED tokens (balanced sample).
    corr_ev = _corrupt(eval_tok.clone(), tscheme, args.rate, K, args.seed + 5,
                       by_len=by_len)
    zk_ev = ztok(corr_ev).reshape(-1, d)
    chg = (corr_ev != eval_tok).reshape(-1).numpy()
    zc_ev = ztok(eval_tok).reshape(-1, d)
    rng = np.random.default_rng(args.seed)
    n_inv = int(chg.sum())
    n_show = min(n_inv, 4000)
    inv_idx = rng.choice(np.where(chg)[0], n_show, replace=False)
    val_idx = rng.choice(zc_ev.shape[0], n_show, replace=False)
    Zt = np.concatenate([zc_ev[val_idx], zk_ev[inv_idx]], 0)
    yt = np.r_[np.zeros(n_show), np.ones(n_show)]

    fig2, ax2 = plt.subplots(1, 3, figsize=(13.5, 4.5))
    # PCA per-token (fit on clean tokens)
    cmu = zc_ev.mean(0)
    _, _, Vt = np.linalg.svd(zc_ev[val_idx] - cmu, full_matrices=False)
    xy = (Zt - cmu) @ Vt[:2].T
    au_pca = _probe_auroc(xy, yt)
    _scatter(ax2[0], xy, yt,
             f"per-token PCA (unsupervised)\nPC1–2 probe AUROC={au_pca:.3f}",
             "PC1", "PC2")
    ax2[0].legend(fontsize=7, markerscale=1.6)
    # Energy-hinge axis (the detector's actual axis) vs orthogonal PC
    ortho = _orth_pc(Zt, w_energy)
    xax, yax = Zt @ w_energy, Zt @ ortho
    au_e = _auroc(xax, yt)
    au_full = _probe_auroc(Zt, yt)
    _scatter(ax2[1], np.c_[xax, yax], yt,
             f"per-token energy-hinge axis (supervised)\n"
             f"energy AUROC={au_e:.3f}  (full-dim ceiling={au_full:.3f})",
             "trained energy axis  E=w·z", "leading orthogonal PC")
    # t-SNE per-token
    nse = min(args.tsne_n, n_show)
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=args.seed).fit_transform(
        np.concatenate([zc_ev[val_idx][:nse], zk_ev[inv_idx][:nse]], 0))
    _scatter(ax2[2], emb, np.r_[np.zeros(nse), np.ones(nse)],
             "per-token t-SNE (nonlinear, exploratory)", "t-SNE 1", "t-SNE 2")
    fig2.suptitle(
        f"Latent split — PER-TOKEN, {tscheme}@{args.rate} (clean tokens vs "
        f"corrupted tokens); blue=valid, red=invalid", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = out / f"latent_split_token_{tscheme}.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)
    metrics["per_token"] = {"token_scheme": tscheme,
                            "pca_pc12_probe_auroc": au_pca, "energy_axis_auroc": au_e,
                            "full_dim_linear_auroc": au_full, "n_per_class": n_show}
    print(f"[token] PCA(PC1-2)={au_pca:.3f}  energy-axis={au_e:.3f}  "
          f"full-linear={au_full:.3f}")
    print(f"Wrote {p2}")

    (out / "latent_split.json").write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out / 'latent_split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
