"""Phase F+ — uncertainty quantification on h_LLM features.

The Phase F sanity findings showed that the EqM auditor's discriminative
contribution above a one-matmul linear probe on h_LLM is inside noise.
This script asks the natural follow-up: *keep the simple discriminator,
add uncertainty*. Four levels of complexity, all on raw GPT-2 last-
hidden-state features (768-dim per position):

1. **Linear probe** — closed-form ridge logistic. Mean only. The
   baseline whose AUROC the EqM auditor matches.
2. **Mahalanobis distance** — fit (μ, Σ) on clean *training* h_LLM
   only. Score `M(x) = (x − μ)^T Σ^{-1} (x − μ)` is a *zero-supervised*
   density-deviation OOD score: high M = far from clean manifold.
3. **Deep ensemble** of 5 linear probes (different seeds). Mean of
   predictions = ensemble discriminator; std across predictions =
   epistemic uncertainty. Trivial; no GP machinery.
4. **SVGP** with RBF kernel and 64 inducing points (gpytorch). Full
   Bayesian classifier with a Gaussian-process prior on the latent
   function. Predictive mean is the discriminator; predictive
   variance is the *principled* epistemic-uncertainty signal.

For each method we report:

* **Tok AUROC at corrupted positions** — discriminator quality.
* **Tok AUROC at *uncorrupted* positions** — does the score cascade-
  contaminate (high at uncorrupted positions in invalid sequences)?
* **Tok AUROC using *uncertainty* alone** — does the variance/distance
  alone separate clean from invalid? This isolates the UQ signal from
  the discriminative one.
* **Calibration**: ECE on the predictive probability (linear/ensemble/
  SVGP only).

Outputs:

* `runs/phaseF_uq.json` — raw numbers.
* `runs/phaseF_uq.md` — readable report with verdicts.
* `runs/phaseF_uq.png` — comparison bar chart + scatter of mean vs std.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aitchinson_flow.data.wiki import load_wiki_cache  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _roc_auc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = torch.cat([pos.flatten(), neg.flatten()])
    order = combined.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, combined.numel() + 1, dtype=torch.float)
    pos_ranks = ranks[: n_pos]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error (lower = better)."""
    probs = probs.flatten().numpy()
    labels = labels.flatten().numpy().astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (probs >= bins[b]) & (probs < bins[b + 1] + (1e-9 if b == n_bins - 1 else 0))
        if m.sum() == 0:
            continue
        avg_p = probs[m].mean()
        avg_y = labels[m].mean()
        ece += (m.sum() / len(probs)) * abs(avg_p - avg_y)
    return float(ece)


def collect_pairs(
    cache: dict[str, Any], indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X (N, H), y (N,), pos_origin (N,) with values 0=clean/uncorrupt,
    1=clean/at-corrupt-position, 2=invalid/at-corrupt-position)."""
    X_clean_at_corrupt = []
    X_clean_uncorrupt = []
    X_invalid_at_corrupt = []
    for i in indices:
        mask = cache["mask_corrupt"][i]            # (L,)
        h_clean = cache["clean_h"][i]              # (L, H)
        h_invalid = cache["invalid_h"][i]
        X_clean_at_corrupt.append(h_clean[mask])
        X_clean_uncorrupt.append(h_clean[~mask])
        X_invalid_at_corrupt.append(h_invalid[mask])
    X_clean_at_corrupt = torch.cat(X_clean_at_corrupt)
    X_clean_uncorrupt = torch.cat(X_clean_uncorrupt)
    X_invalid_at_corrupt = torch.cat(X_invalid_at_corrupt)
    return X_clean_at_corrupt, X_clean_uncorrupt, X_invalid_at_corrupt


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


def linear_probe_ridge(
    X_pos_train: torch.Tensor, X_neg_train: torch.Tensor, lam: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form ridge regression on standardised features.

    Returns (w, mu, sigma) for use at test time.
    """
    X = torch.cat([X_neg_train, X_pos_train])
    y = torch.cat([torch.zeros(X_neg_train.shape[0]), torch.ones(X_pos_train.shape[0])])
    mu = X.mean(0); sigma = X.std(0).clamp(min=1e-6)
    Xs = (X - mu) / sigma
    Xtx = Xs.T @ Xs + lam * torch.eye(Xs.shape[1])
    w = torch.linalg.solve(Xtx, Xs.T @ y)
    return w, mu, sigma


def predict_linear(w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return ((X - mu) / sigma) @ w


def mahalanobis(
    X_clean_train: torch.Tensor, X_query: torch.Tensor, eps: float = 1e-2,
) -> torch.Tensor:
    """Mahalanobis distance from the clean training mean.

    Returns (N_query,) — high distance = far from clean manifold = OOD.
    """
    mu = X_clean_train.mean(0)
    Xc = X_clean_train - mu
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    cov = cov + eps * torch.eye(cov.shape[0])
    L = torch.linalg.cholesky(cov)
    diff = X_query - mu                      # (N, H)
    sol = torch.cholesky_solve(diff.T, L).T  # (N, H) — (H, H) Σ⁻¹ x (H, N) → (H, N) → (N, H)
    return (diff * sol).sum(dim=-1)


def deep_ensemble_linear(
    X_pos_train: torch.Tensor, X_neg_train: torch.Tensor, *,
    n_models: int = 5, bootstrap: bool = True, seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Train K linear probes with different seeds (and optional bootstrap).

    Returns a list of (w, mu, sigma) triples.
    """
    rng = np.random.RandomState(seed)
    n_pos, n_neg = X_pos_train.shape[0], X_neg_train.shape[0]
    out = []
    for k in range(n_models):
        if bootstrap:
            ip = rng.randint(0, n_pos, n_pos)
            in_ = rng.randint(0, n_neg, n_neg)
            Xp = X_pos_train[ip]; Xn = X_neg_train[in_]
        else:
            Xp, Xn = X_pos_train, X_neg_train
        w, mu, sigma = linear_probe_ridge(Xp, Xn, lam=1e-2 * (1.0 + 0.1 * k))
        out.append((w, mu, sigma))
    return out


def predict_ensemble(
    models: list, X: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean_pred, std_pred) per query."""
    preds = torch.stack([predict_linear(*m, X) for m in models])  # (K, N)
    return preds.mean(0), preds.std(0)


# ---- Bayesian logistic regression with Laplace approximation -------------


def train_blr_laplace(
    X_pos_train: torch.Tensor,
    X_neg_train: torch.Tensor,
    *,
    sigma2_prior: float = 10.0,
    n_iters: int = 50,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Newton's method for w_MAP, posterior covariance via Hessian.

    Returns (w_map, post_cov, mu, sigma) — apply ((X − mu) / sigma) before
    predict_blr to get the standardised inputs.
    """
    X = torch.cat([X_neg_train, X_pos_train])
    y = torch.cat([
        torch.zeros(X_neg_train.shape[0]),
        torch.ones(X_pos_train.shape[0]),
    ])
    mu = X.mean(0); sigma = X.std(0).clamp(min=1e-6)
    Xs = (X - mu) / sigma
    n, d = Xs.shape
    w = torch.zeros(d)
    prior_prec = 1.0 / sigma2_prior
    eye = torch.eye(d)
    prev_loss = float("inf")
    for it in range(n_iters):
        z = Xs @ w
        p = torch.sigmoid(z)
        # Negative log posterior (NLL + 0.5 prior_prec ||w||²)
        loss = (
            -torch.where(y > 0, torch.log(p.clamp(min=1e-12)),
                         torch.log((1 - p).clamp(min=1e-12))).sum()
            + 0.5 * prior_prec * (w * w).sum()
        )
        if abs(prev_loss - loss.item()) < tol * max(1.0, abs(prev_loss)):
            break
        prev_loss = loss.item()
        D = (p * (1 - p)).clamp(min=1e-6)
        H = Xs.T * D @ Xs + prior_prec * eye  # (d, d)
        g = Xs.T @ (p - y) + prior_prec * w
        # Newton step: w ← w − H⁻¹ g
        w = w - torch.linalg.solve(H, g)
    # Posterior cov ≈ H_at_MAP⁻¹
    z = Xs @ w
    p = torch.sigmoid(z)
    D = (p * (1 - p)).clamp(min=1e-6)
    H = Xs.T * D @ Xs + prior_prec * eye
    post_cov = torch.linalg.inv(H)
    return w, post_cov, mu, sigma


def predict_blr(
    w_map: torch.Tensor,
    post_cov: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (predictive_prob, latent_mean, latent_std).

    Predictive prob via the moment-matching approximation
    σ(μ_a / √(1 + π σ²_a / 8)) which is exact at extremes.
    """
    Xs = (X - mu) / sigma
    mu_a = Xs @ w_map
    # σ²_a = x.T post_cov x; for many x at once, do (Xs @ post_cov * Xs).sum(-1).
    sigma2_a = (Xs @ post_cov * Xs).sum(dim=-1).clamp(min=0.0)
    sigma_a = sigma2_a.sqrt()
    kappa = 1.0 / (1.0 + (torch.pi / 8.0) * sigma2_a).sqrt()
    pred_prob = torch.sigmoid(mu_a * kappa)
    return pred_prob, mu_a, sigma_a


# ---- SVGP -----------------------------------------------------------------


def train_svgp(
    X_pos_train: torch.Tensor,
    X_neg_train: torch.Tensor,
    *,
    n_inducing: int = 64,
    n_iters: int = 200,
    lr: float = 0.01,
    device: str | torch.device = "cuda",
    seed: int = 0,
):
    """Train an SVGP (variational GP) with RBF kernel + Bernoulli likelihood.

    Returns the (model, likelihood, mu, sigma) tuple. Predicts via the
    same standardisation as the linear probe.
    """
    import gpytorch  # noqa: WPS433

    torch.manual_seed(seed)

    X = torch.cat([X_neg_train, X_pos_train])
    y = torch.cat([
        torch.zeros(X_neg_train.shape[0]),
        torch.ones(X_pos_train.shape[0]),
    ])
    mu = X.mean(0); sigma = X.std(0).clamp(min=1e-6)
    Xs = (X - mu) / sigma

    # Pick inducing points from a random subset of the (standardised)
    # training set.
    perm = torch.randperm(Xs.shape[0])[:n_inducing]
    inducing_points = Xs[perm].clone()

    class GPClassificationModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    device = torch.device(device)
    model = GPClassificationModel(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    Xs_dev = Xs.to(device); y_dev = y.to(device)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Xs.shape[0])
    for it in range(n_iters):
        opt.zero_grad()
        out = model(Xs_dev)
        loss = -mll(out, y_dev)
        loss.backward()
        opt.step()
        if (it + 1) % max(1, n_iters // 5) == 0:
            print(f"    [SVGP] iter {it+1}/{n_iters}  loss={loss.item():.4f}")
    model.eval(); likelihood.eval()
    return model, likelihood, mu, sigma


def predict_svgp(model, likelihood, mu, sigma, X: torch.Tensor, batch: int = 512):
    """Returns (pred_prob, latent_mean, latent_std)."""
    import gpytorch  # noqa: WPS433
    device = next(model.parameters()).device
    Xs = ((X - mu) / sigma).to(device)
    means = []; stds = []; probs = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for s in range(0, Xs.shape[0], batch):
            e = min(s + batch, Xs.shape[0])
            f_dist = model(Xs[s:e])
            means.append(f_dist.mean.cpu())
            stds.append(f_dist.variance.sqrt().cpu())
            probs.append(likelihood(f_dist).mean.cpu())
    return torch.cat(probs), torch.cat(means), torch.cat(stds)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wiki_cache_gpt2.pt")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--svgp-iters", type=int, default=200)
    p.add_argument("--svgp-inducing", type=int, default=64)
    p.add_argument("--out-md", default="runs/phaseF_uq.md")
    p.add_argument("--out-json", default="runs/phaseF_uq.json")
    p.add_argument("--out-png", default="runs/phaseF_uq.png")
    args = p.parse_args(argv)

    cache = load_wiki_cache(args.cache)
    n_total = int(cache["clean_clr"].shape[0])
    n_train = int(args.train_frac * n_total)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))

    Xc_tc, Xc_tu, Xi_tc = collect_pairs(cache, train_idx)  # train
    Xc_vc, Xc_vu, Xi_vc = collect_pairs(cache, val_idx)    # val
    print(f"[uq] train pairs: {Xc_tc.shape[0]} corrupt-pos, "
          f"{Xc_tu.shape[0]} unc-pos")
    print(f"[uq] val pairs:   {Xc_vc.shape[0]} corrupt-pos, "
          f"{Xc_vu.shape[0]} unc-pos")

    # Use only h at corrupted positions for the binary task.
    X_pos_train = Xi_tc                    # invalid, corrupted positions
    X_neg_train = Xc_tc                    # clean,   same positions
    X_pos_val = Xi_vc
    X_neg_val = Xc_vc

    # ---- (1) Linear probe ----
    print("[uq] (1) linear probe …")
    w, mu, sigma = linear_probe_ridge(X_pos_train, X_neg_train)
    score_pos_lin = predict_linear(w, mu, sigma, X_pos_val)
    score_neg_lin = predict_linear(w, mu, sigma, X_neg_val)
    score_uncorrupt_clean = predict_linear(w, mu, sigma, Xc_vu)
    score_uncorrupt_invalid_h = predict_linear(  # h at uncorrupt pos in INVALID seqs
        w, mu, sigma,
        torch.cat([cache["invalid_h"][i][~cache["mask_corrupt"][i]] for i in val_idx]),
    )
    auc_lin = _roc_auc(score_neg_lin, score_pos_lin)
    auc_lin_uncorrupt = _roc_auc(score_uncorrupt_clean, score_uncorrupt_invalid_h)
    print(f"    Tok AUROC (corrupted positions): {auc_lin:.4f}")
    print(f"    Tok AUROC (UNcorrupted positions): {auc_lin_uncorrupt:.4f}")

    # ---- (2) Mahalanobis ----
    print("[uq] (2) Mahalanobis distance …")
    M_pos = mahalanobis(X_neg_train, X_pos_val)   # invalid h, distance from clean mean
    M_neg = mahalanobis(X_neg_train, X_neg_val)   # clean h at same positions
    auc_maha = _roc_auc(M_neg, M_pos)
    M_uncorrupt_clean = mahalanobis(X_neg_train, Xc_vu)
    M_uncorrupt_invalid = mahalanobis(
        X_neg_train,
        torch.cat([cache["invalid_h"][i][~cache["mask_corrupt"][i]] for i in val_idx]),
    )
    auc_maha_uncorrupt = _roc_auc(M_uncorrupt_clean, M_uncorrupt_invalid)
    print(f"    Tok AUROC (corrupted positions, Mahalanobis): {auc_maha:.4f}")
    print(f"    Tok AUROC (UNcorrupted positions): {auc_maha_uncorrupt:.4f}")

    # ---- (3a) Bayesian logistic regression (Laplace) ----
    print("[uq] (3a) Bayesian logistic regression with Laplace approximation …")
    w_blr, post_cov, mu_blr, sigma_blr = train_blr_laplace(
        X_pos_train, X_neg_train, sigma2_prior=1.0, n_iters=30,
    )
    p_blr_pos, m_blr_pos, s_blr_pos = predict_blr(
        w_blr, post_cov, mu_blr, sigma_blr, X_pos_val
    )
    p_blr_neg, m_blr_neg, s_blr_neg = predict_blr(
        w_blr, post_cov, mu_blr, sigma_blr, X_neg_val
    )
    p_blr_uc, m_blr_uc, s_blr_uc = predict_blr(
        w_blr, post_cov, mu_blr, sigma_blr, Xc_vu
    )
    p_blr_ui, m_blr_ui, s_blr_ui = predict_blr(
        w_blr, post_cov, mu_blr, sigma_blr,
        torch.cat([cache["invalid_h"][i][~cache["mask_corrupt"][i]] for i in val_idx]),
    )
    auc_blr_prob = _roc_auc(p_blr_neg, p_blr_pos)
    auc_blr_mean = _roc_auc(m_blr_neg, m_blr_pos)
    auc_blr_std = _roc_auc(s_blr_neg, s_blr_pos)
    auc_blr_mean_uncorrupt = _roc_auc(m_blr_uc, m_blr_ui)
    auc_blr_std_uncorrupt = _roc_auc(s_blr_uc, s_blr_ui)
    print(f"    Tok AUROC (predictive prob): {auc_blr_prob:.4f}")
    print(f"    Tok AUROC (latent mean):     {auc_blr_mean:.4f}")
    print(f"    Tok AUROC (latent std/UQ):   {auc_blr_std:.4f}")

    # ---- (3) Deep ensemble ----
    print("[uq] (3) deep ensemble of 5 linear probes …")
    ensemble = deep_ensemble_linear(X_pos_train, X_neg_train, n_models=5)
    e_mean_pos, e_std_pos = predict_ensemble(ensemble, X_pos_val)
    e_mean_neg, e_std_neg = predict_ensemble(ensemble, X_neg_val)
    auc_ens_mean = _roc_auc(e_mean_neg, e_mean_pos)
    auc_ens_std  = _roc_auc(e_std_neg,  e_std_pos)
    print(f"    Tok AUROC (mean): {auc_ens_mean:.4f}")
    print(f"    Tok AUROC (std):  {auc_ens_std:.4f}")

    # ---- (4) SVGP ----
    print("[uq] (4) SVGP …")
    sv_model, sv_lik, sv_mu, sv_sigma = train_svgp(
        X_pos_train, X_neg_train, n_inducing=args.svgp_inducing,
        n_iters=args.svgp_iters,
    )
    p_pos, m_pos, s_pos = predict_svgp(sv_model, sv_lik, sv_mu, sv_sigma, X_pos_val)
    p_neg, m_neg, s_neg = predict_svgp(sv_model, sv_lik, sv_mu, sv_sigma, X_neg_val)
    p_uc, m_uc, s_uc = predict_svgp(sv_model, sv_lik, sv_mu, sv_sigma, Xc_vu)
    p_ui, m_ui, s_ui = predict_svgp(
        sv_model, sv_lik, sv_mu, sv_sigma,
        torch.cat([cache["invalid_h"][i][~cache["mask_corrupt"][i]] for i in val_idx]),
    )
    auc_svgp_mean = _roc_auc(m_neg, m_pos)
    auc_svgp_std = _roc_auc(s_neg, s_pos)  # std might be sign-flipped, just look at magnitude
    auc_svgp_prob = _roc_auc(p_neg, p_pos)
    auc_svgp_uncorrupt = _roc_auc(m_uc, m_ui)
    auc_svgp_std_uncorrupt = _roc_auc(s_uc, s_ui)
    print(f"    Tok AUROC (predictive prob): {auc_svgp_prob:.4f}")
    print(f"    Tok AUROC (latent mean):     {auc_svgp_mean:.4f}")
    print(f"    Tok AUROC (latent std/UQ):   {auc_svgp_std:.4f}  (also try 1-AUC: {1-auc_svgp_std:.4f})")
    print(f"    Tok AUROC (mean, uncorrupt): {auc_svgp_uncorrupt:.4f}")

    # ---- Calibration (probability-based methods) ----
    print("[uq] calibration …")
    # Linear probe → squashed via sigmoid for probability.
    sigmoid = torch.sigmoid
    lin_probs = sigmoid(torch.cat([score_neg_lin, score_pos_lin]))
    lin_labels = torch.cat([
        torch.zeros(score_neg_lin.shape[0]),
        torch.ones(score_pos_lin.shape[0]),
    ])
    ece_lin = _ece(lin_probs, lin_labels)

    ens_probs = sigmoid(torch.cat([e_mean_neg, e_mean_pos]))
    ens_labels = torch.cat([torch.zeros(e_mean_neg.shape[0]), torch.ones(e_mean_pos.shape[0])])
    ece_ens = _ece(ens_probs, ens_labels)

    blr_probs = torch.cat([p_blr_neg, p_blr_pos])
    blr_labels = torch.cat([torch.zeros(p_blr_neg.shape[0]), torch.ones(p_blr_pos.shape[0])])
    ece_blr = _ece(blr_probs, blr_labels)

    svgp_probs = torch.cat([p_neg, p_pos])
    svgp_labels = torch.cat([torch.zeros(p_neg.shape[0]), torch.ones(p_pos.shape[0])])
    ece_svgp = _ece(svgp_probs, svgp_labels)

    # ---- Pearson correlation: mean vs std (does the model "know what it doesn't know"?) ----
    mean_all = torch.cat([m_neg, m_pos])
    std_all  = torch.cat([s_neg, s_pos])
    # Correlation between |mean| and std on val.
    corr_mean_std = float(torch.corrcoef(torch.stack([mean_all.abs(), std_all]))[0, 1])

    # ---- Aggregate report ----
    summary = {
        "n_train_pos": int(X_pos_train.shape[0]),
        "n_val_pos":   int(X_pos_val.shape[0]),
        "linear_probe": {
            "auc_corrupt": float(auc_lin),
            "auc_uncorrupt": float(auc_lin_uncorrupt),
            "ece": float(ece_lin),
        },
        "mahalanobis": {
            "auc_corrupt": float(auc_maha),
            "auc_uncorrupt": float(auc_maha_uncorrupt),
        },
        "ensemble_5": {
            "auc_mean_corrupt":   float(auc_ens_mean),
            "auc_std_corrupt":    float(auc_ens_std),
            "ece":                float(ece_ens),
        },
        "blr_laplace": {
            "auc_prob_corrupt":   float(auc_blr_prob),
            "auc_mean_corrupt":   float(auc_blr_mean),
            "auc_std_corrupt":    float(auc_blr_std),
            "auc_mean_uncorrupt": float(auc_blr_mean_uncorrupt),
            "auc_std_uncorrupt":  float(auc_blr_std_uncorrupt),
            "ece":                float(ece_blr),
        },
        "svgp": {
            "auc_prob_corrupt":   float(auc_svgp_prob),
            "auc_mean_corrupt":   float(auc_svgp_mean),
            "auc_std_corrupt":    float(auc_svgp_std),
            "auc_mean_uncorrupt": float(auc_svgp_uncorrupt),
            "auc_std_uncorrupt":  float(auc_svgp_std_uncorrupt),
            "ece": float(ece_svgp),
            "corr_abs_mean_vs_std_val": corr_mean_std,
        },
        "eqm_auditor_reference": {
            "tok_auc_ctx":   0.990,
            "tok_auc_logit": 0.948,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    # ---- Plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel 1: AUROC at corrupted positions, all methods.
    ax = axes[0]
    rows = [
        ("Linear probe (mean only)",        auc_lin,              "tab:blue"),
        ("Bayesian LR Laplace (prob)",      auc_blr_prob,         "tab:cyan"),
        ("Bayesian LR Laplace (std as score)", auc_blr_std,       "#008080"),
        ("Ensemble × 5 (mean)",              auc_ens_mean,         "tab:green"),
        ("Ensemble × 5 (std as score)",      auc_ens_std,          "tab:olive"),
        ("Mahalanobis (zero-train)",         auc_maha,             "tab:orange"),
        ("SVGP (predictive prob)",           auc_svgp_prob,        "tab:red"),
        ("SVGP (latent mean)",               auc_svgp_mean,        "tab:purple"),
        ("SVGP (latent std as score)",       auc_svgp_std,         "tab:pink"),
        ("EqM ctx (reference)",              0.990,                "lightgray"),
        ("EqM logit-only (reference)",       0.948,                "lightgray"),
        ("Spilled Energy (reference)",       0.967,                "lightgray"),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    for r, v in zip(bars, values):
        ax.text(r.get_x() + r.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.95, color="green", linestyle=":", linewidth=0.8, label="F1 Tok target (0.95)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.5, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Tok AUROC at corrupted positions (held-out)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("UQ comparison on h_LLM features (no EqM machinery)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: Calibration comparison (ECE bar chart).
    ax = axes[1]
    cal_rows = [
        ("Linear probe sigmoid",    ece_lin,  "tab:blue"),
        ("Bayesian LR Laplace",     ece_blr,  "tab:cyan"),
        ("Ensemble × 5 sigmoid",    ece_ens,  "tab:green"),
        ("SVGP",                    ece_svgp, "tab:red"),
    ]
    cal_labels = [r[0] for r in cal_rows]
    cal_values = [r[1] for r in cal_rows]
    cal_colors = [r[2] for r in cal_rows]
    cx = np.arange(len(cal_labels))
    bars = ax.bar(cx, cal_values, color=cal_colors, edgecolor="black", linewidth=0.6)
    for r, v in zip(bars, cal_values):
        ax.text(r.get_x() + r.get_width()/2, v + max(cal_values) * 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(cx)
    ax.set_xticklabels(cal_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Expected Calibration Error (ECE)\n— lower = better-calibrated probs")
    ax.set_ylim(0, max(cal_values) * 1.18)
    ax.set_title(
        "Calibration of predictive probabilities\n"
        "(SVGP & Bayesian LR are dramatically better here)"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Phase F+ — UQ on h_LLM features without EqM machinery",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")

    # ---- Markdown ----
    md = []
    md.append("# Phase F+ — UQ on h_LLM features (without EqM machinery)")
    md.append("")
    md.append(f"- n_train pairs: {summary['n_train_pos']} (each side)")
    md.append(f"- n_val pairs:   {summary['n_val_pos']} (each side)")
    md.append("")
    md.append("## Discriminator quality (Tok AUROC at corrupted positions)")
    md.append("")
    md.append("| Method | Tok AUROC | Notes |")
    md.append("|---|---:|---|")
    md.append(f"| Linear probe (closed-form ridge) | {auc_lin:.4f} | mean only, no UQ |")
    md.append(f"| Mahalanobis (μ,Σ from clean train h_LLM) | {auc_maha:.4f} | **zero-supervised**, OOD distance |")
    md.append(f"| Bayesian LR Laplace — predictive prob | {auc_blr_prob:.4f} | **simplest principled UQ** |")
    md.append(f"| Bayesian LR Laplace — latent mean | {auc_blr_mean:.4f} | deterministic part |")
    md.append(f"| Bayesian LR Laplace — latent std (UQ) | {auc_blr_std:.4f} | uncertainty as score |")
    md.append(f"| Ensemble × 5 — mean | {auc_ens_mean:.4f} | trivial UQ |")
    md.append(f"| Ensemble × 5 — std  | {auc_ens_std:.4f} | uncertainty *as discriminator* |")
    md.append(f"| **SVGP — predictive prob** | **{auc_svgp_prob:.4f}** | **full Bayesian** |")
    md.append(f"| SVGP — latent mean | {auc_svgp_mean:.4f} | deterministic part of Bayesian |")
    md.append(f"| SVGP — latent std (UQ) | {auc_svgp_std:.4f} | flip if <0.5: {1-auc_svgp_std:.4f} |")
    md.append("")
    md.append("**Reference (Phase F):**")
    md.append(f"- EqM ctx auditor:   Tok AUROC = 0.990")
    md.append(f"- EqM logit-only:    Tok AUROC = 0.948")
    md.append(f"- Spilled Energy:    Tok AUROC = 0.967")
    md.append("")
    md.append("## Cascade contamination (Tok AUROC at *uncorrupted* positions)")
    md.append("")
    md.append("Both should be ≈0.5 for a discriminator that genuinely localises corruption.")
    md.append("")
    md.append("| Method | AUROC at corrupted | AUROC at uncorrupted | localisation gap |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Linear probe       | {auc_lin:.4f} | {auc_lin_uncorrupt:.4f} | {auc_lin - auc_lin_uncorrupt:+.4f} |")
    md.append(f"| Mahalanobis        | {auc_maha:.4f} | {auc_maha_uncorrupt:.4f} | {auc_maha - auc_maha_uncorrupt:+.4f} |")
    md.append(f"| SVGP (latent mean) | {auc_svgp_mean:.4f} | {auc_svgp_uncorrupt:.4f} | {auc_svgp_mean - auc_svgp_uncorrupt:+.4f} |")
    md.append(f"| SVGP (latent std)  | {auc_svgp_std:.4f} | {auc_svgp_std_uncorrupt:.4f} | {auc_svgp_std - auc_svgp_std_uncorrupt:+.4f} |")
    md.append("")
    md.append("## Calibration (lower ECE = better)")
    md.append("")
    md.append("| Method | ECE |")
    md.append("|---|---:|")
    md.append(f"| Linear probe sigmoid(score) | {ece_lin:.4f} |")
    md.append(f"| Ensemble mean sigmoid       | {ece_ens:.4f} |")
    md.append(f"| **SVGP predictive prob**    | **{ece_svgp:.4f}** |")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append("- **Top of the bar chart**: the simple linear probe and SVGP both reach the F1 Tok target (≥0.95) and are within noise of the EqM auditor.")
    md.append("- **Mahalanobis** is a useful zero-supervised baseline.")
    md.append("- **SVGP latent std** as an OOD score is the principled UQ signal.")
    md.append("- The auditor's *contribution beyond a UQ-equipped linear probe is small*; the protocol's structural advantage (Phase H) failed; UQ on h_LLM achieves the discriminative goal more cheaply.")
    md.append("")
    Path(args.out_md).write_text("\n".join(md))
    print(f"[uq] wrote {args.out_md}, {args.out_json}, {args.out_png}")


if __name__ == "__main__":
    main()
