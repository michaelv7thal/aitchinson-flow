#!/usr/bin/env python
"""Derive every band constant from (K, epsilon, sigma) — no GPU, no hard-coding.

This is the calculation behind docs/band_geometry_theory.md. It replaces three
sets of hand-tabulated constants (gamma_star, gamma_lo, g_for_alpha) with one
set of principles, so a vocabulary or smoothing change stays correct by
construction.

    uv run python scripts/band_geometry.py                 # text8 defaults
    uv run python scripts/band_geometry.py --K 256         # a BPE-ish vocab
    uv run python scripts/band_geometry.py --yaml          # emit sweep overrides

Everything marked EXACT is closed form or quadrature to machine precision.
rho() is Monte-Carlo (no separable 1-D form exists — see its docstring); its
standard error is printed so you can see the noise floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats


# ── 1. Scalar geometry ──────────────────────────────────────────────────────
def geometry(K: int, eps: float, sigma: float) -> dict:
    """Δ, and the three norms. All EXACT.

    x1 is the CLR of a label-smoothed one-hot: p_hi = 1-eps+eps/K, p_lo = eps/K.
    Δ = log(p_hi/p_lo) is the correct-vs-competitor log-ratio gap. The CLR
    coordinates are (K-1)Δ/K on the true token and -Δ/K on the other K-1, which
    is zero-sum by construction, so the gap between them is exactly Δ.
    """
    Delta = np.log((1 - eps + eps / K) * K / eps)
    return {
        "Delta": Delta,
        "n1": Delta * np.sqrt((K - 1) / K),  # ||x1||
        "n0": sigma * np.sqrt(K - 1),  # ||x0||, x0 = sigma*randn on V_d
    }


def t_of(gamma, Delta, sigma):
    """The standardised margin. EXACT.

    Everything below is a function of (K, t) alone: gamma, sigma and eps reach
    the statistics only through this one dimensionless number.
    """
    return gamma * Delta / ((1 - gamma) * sigma)


def gamma_of(t, Delta, sigma):
    """Inverse of t_of. EXACT."""
    return sigma * t / (Delta + sigma * t)


def norm_at(gamma, n0, n1):
    """||x_gamma||, using x0 ⟂ x1 in expectation. EXACT."""
    return np.hypot((1 - gamma) * n0, gamma * n1)


# ── 2. Decodability → gamma_star ────────────────────────────────────────────
def A_K(t: float, K: int) -> float:
    """P(argmax x_gamma == true token). EXACT (1-D quadrature).

    Condition on the true coordinate's noise g_c; the other K-1 competitors are
    then independent, so P = ∫ φ(x)·Φ(x+t)^(K-1) dx. Evaluate Φ^(K-1) as
    exp((K-1)·logΦ) or it underflows for large K.
    """

    def integrand(x):
        return stats.norm.pdf(x) * np.exp((K - 1) * stats.norm.logcdf(x + t))

    return integrate.quad(integrand, -12.0, 12.0 + abs(t), limit=200)[0]


def gamma_star(acc: float, K: int, Delta: float, sigma: float) -> tuple[float, float]:
    """gamma* = the gamma at which `acc` of tokens decode correctly. EXACT.

    gamma* IS the equilibrium, so `acc` is a hard ceiling on the fixed point's
    token accuracy — not a tuning knob.
    """
    t = optimize.brentq(lambda t: A_K(t, K) - acc, 1e-6, 40.0, xtol=1e-12)
    return t, gamma_of(t, Delta, sigma)


# ── 3. Learnable fraction → gamma_lo, requirement N ─────────────────────────
def rho(t: float, K: int, n: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Bayes R² of predicting x1 from x_gamma. MONTE CARLO.

    Given x_gamma, the source is determined (x0 = (x_gamma - gamma*x1)/(1-gamma)),
    so the flow target c(gamma)(x0-x1) = [c/(1-gamma)]*(x_gamma - x1) and the
    regression is exactly "predict x1 from x_gamma" — the scalar cancels, so rho
    is a property of the geometry, not of c.

    The posterior over the token is a softmax whose argument reduces to
        p_k ∝ exp(t*g_k + t²*δ_kc)
    (the zero-mean shift is constant in k and cancels). With a uniform prior,
        rho = (E[Σ_k p_k²] - 1/K) / (1 - 1/K)
    Σ_k p_k² is a ratio of sums over k, so it does not separate into a 1-D
    integral the way A_K does — hence Monte Carlo. Returns (rho, stderr).
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, K))
    g -= g.mean(1, keepdims=True)  # project to V_d (cancels anyway)
    c = rng.integers(0, K, n)
    lg = t * g
    lg[np.arange(n), c] += t * t
    lg -= lg.max(1, keepdims=True)  # softmax stability
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    coll = np.square(p).sum(1)  # Σ_k p_k², per sample
    r = (coll.mean() - 1 / K) / (1 - 1 / K)
    se = coll.std(ddof=1) / np.sqrt(n) / (1 - 1 / K)
    return r, se


def rho_grid(K, Delta, sigma, gmax, m=120, n=60_000, seed=1):
    """rho on a gamma grid, for integrating W. Interpolate rather than re-running
    Monte Carlo inside a quadrature loop (that is ~100x slower for no accuracy)."""
    G = np.linspace(1e-5, gmax, m)
    R = np.array([rho(t_of(g, Delta, sigma), K, n=n, seed=seed)[0] for g in G])
    return G, R


def W(glo: float, gstar: float, G, R) -> float:
    """Irreducible share of the c²-weighted training signal on [glo, gstar].

    W = ∫ c²(1-rho) dgamma / ∫ c² dgamma.  This is the noise floor of the
    regression: the fraction of every gradient that is posterior spread rather
    than signal. NOT a bias — the Bayes predictor is still what gets learned;
    W inflates gradient variance.
    """
    m = (G >= glo) & (G <= gstar)
    g, r = G[m], R[m]
    c2 = np.clip(1 - g / gstar, 0, None) ** 2
    return float(np.trapezoid(c2 * (1 - r), g) / np.trapezoid(c2, g))


# ── 4. Requirement S: the initialisation must lie inside the band ───────────
def t_support(kappa: float, K: int) -> float:
    """Requirement S. EXACT (given kappa).

    The sampler starts at x0 ~ N(0, sigma²P). The band at gamma_lo is a mixture
    of K Gaussians centred at gamma_lo*x1^(k) with the same covariance. The
    displacement between the init's centre and a component centre, measured in
    per-coordinate noise std, is gamma*||x1||/((1-gamma)*sigma) = t*sqrt((K-1)/K)
    ≈ t. The Gaussian's own radius in K-1 dimensions is sqrt(K-1). Requiring the
    init to sit within a fraction kappa of that radius gives t <= kappa*sqrt(K-1).

    kappa is a choice; the FORM is what is analytic, and it is what makes S
    directly comparable to the SNR requirement N (both live in t).
    """
    return kappa * np.sqrt(K - 1)


# ── 5. Renormalisation constants ────────────────────────────────────────────
# The factors the L=256 recovery ladder was actually run with (drive_band_L256.sh
# g_for_alpha): radius-matched to a fixed |x_gamma| = 0.539, tabulated by hand
# before gamma_match() existed. Kept here only for the comparison in
# docs/band_geometry_theory.md 5.3; new runs should use gamma_match().
DRIVER_G = {0.1: 0.0390, 0.3: 0.0237, 0.5: 0.0160, 0.6: 0.01342, 0.7: 0.01165, 0.8: 0.01027, 1.0: 0.0083}

def gamma_match(alpha: float, Delta: float, sigma: float) -> float:
    """Where a rescaled alpha-perturbed input lands. EXACT.

    A scalar rescale cannot rotate, so the angle to x1 is fixed by alpha alone.
    Equating cos(g*z_a, x1) = cos(x_gamma, x1) and solving, the sqrt(K-1)
    cancels exactly:  gamma_match = sigma / (sigma + alpha*Delta).
    Independent of gamma* — the landing point does not move when gamma* does.
    """
    return sigma / (sigma + alpha * Delta)


def g_for_alpha(alpha, Delta, sigma, n0, n1):
    """The rescale factor: put the perturbed input on the band manifold. EXACT.

    IDENTITY: g(alpha) == gamma_match(alpha) == sigma/(sigma + alpha*Delta).

    Proof: angle-matching (docs section 5.2) gives (1-gm)*||x0|| = gm*||x1||*sqrt(K)*alpha,
    so  ||x_gm||^2 = (gm*||x1||*sqrt(K)*alpha)^2 + (gm*||x1||)^2 = gm^2*||x1||^2*(1+K*alpha^2),
    i.e. ||x_gm|| = gm*||z_a||, hence g = ||x_gm||/||z_a|| = gm.

    So the rescale factor IS the landing gamma, and the whole hand-tabulated
    g_for_alpha table in the drivers collapses to one closed form. The tabulated
    constants are 1.6-9.6% off (worst at alpha=0.3, the only losing rung).
    """
    gm = gamma_match(alpha, Delta, sigma)
    return norm_at(gm, n0, n1), gm


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--K", type=int, default=27)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument(
        "--acc",
        type=float,
        default=0.99,
        help="decode-accuracy target for gamma* (= the fixed point's ceiling)",
    )
    ap.add_argument(
        "--w",
        type=float,
        default=0.30,
        help="requirement N: max irreducible share of the training signal",
    )
    ap.add_argument(
        "--kappa",
        type=float,
        default=0.123,
        help="requirement S: init within this fraction of the noise radius",
    )
    ap.add_argument("--alphas", default="0.3,0.5,0.6,0.7,0.8,1.0")
    ap.add_argument("--mc", type=int, default=60_000, help="MC samples per grid point")
    ap.add_argument("--yaml", action="store_true", help="emit sweep overrides and exit")
    ap.add_argument(
        "--json",
        default=None,
        help="also write the EXACT constants to this path (the claims.tsv artifact)",
    )
    a = ap.parse_args()

    K, eps, sigma = a.K, a.eps, a.sigma
    G_ = geometry(K, eps, sigma)
    Delta, n0, n1 = G_["Delta"], G_["n0"], G_["n1"]

    # --- gamma* -------------------------------------------------------------
    _t_gs, gs = gamma_star(a.acc, K, Delta, sigma)
    clip = 1.0 * gs / 0.03  # eqm.py: sample_grad_clip scales with gamma*

    # --- rho grid, W, requirement N ----------------------------------------
    G, R = rho_grid(K, Delta, sigma, gmax=gs * 1.3, n=a.mc)
    grid = np.linspace(1e-5, gs * 0.75, 240)
    Ws = np.array([W(x, gs, G, R) for x in grid])
    iN = int(np.argmin(np.abs(Ws - a.w)))
    glo_N, tN = grid[iN], t_of(grid[iN], Delta, sigma)

    # --- requirement S ------------------------------------------------------
    tS = t_support(a.kappa, K)
    glo_S = gamma_of(tS, Delta, sigma)

    if a.yaml:
        print(
            f"    # derived by scripts/band_geometry.py --K {K} --eps {eps} "
            f"--sigma {sigma} --acc {a.acc}"
        )
        print(
            f"    eqm.gamma_lo: {glo_S:.5f}          # requirement S, kappa={a.kappa}"
        )
        print(f"    eqm.gamma_hi: {gs:.5f}")
        print(f"    eqm.gamma_star: {gs:.5f}        # {a.acc:.1%} decode ceiling")
        print(f"    eqm.sample_grad_clip: {clip:.3f}   # 1.0 * gamma*/0.03")
        print(f"    eqm.source_sigma: {sigma}")
        return

    print("=" * 74)
    print(f"BAND GEOMETRY   K={K}  eps={eps:g}  sigma={sigma}")
    print("=" * 74)
    print("\n1. SCALARS [EXACT]")
    print(f"   Delta = log((1-eps+eps/K)*K/eps)      = {Delta:.4f}")
    print(f"   ||x1|| = Delta*sqrt((K-1)/K)          = {n1:.4f}")
    print(f"   ||x0|| = sigma*sqrt(K-1)              = {n0:.4f}")
    print(f"   data/noise scale gap                  = {n1 / n0:.1f}x")

    print("\n2. gamma* FROM DECODABILITY [EXACT]  (A_K inversion)")
    print(f"   {'target':>8} {'t*':>8} {'gamma*':>9}")
    gs_table = {}
    for acc in (0.50, 0.95, 0.99, 0.999):
        t_, g_ = gamma_star(acc, K, Delta, sigma)
        gs_table[acc] = (float(t_), float(g_))
        mark = "  <- chosen" if abs(acc - a.acc) < 1e-9 else ""
        print(f"   {acc:>8.1%} {t_:>8.4f} {g_:>9.5f}{mark}")
    print(
        f"   ||x_gamma*|| = {norm_at(gs, n0, n1):.4f}   sample_grad_clip -> {clip:.3f}"
    )
    print(
        f"   alpha_cross = sigma(1-g*)/(g*·Delta)  = {sigma * (1 - gs) / (gs * Delta):.4f}"
    )

    print("\n3. gamma_lo — TWO REQUIREMENTS, both in t")
    _r0, se0 = rho(t_of(0.005, Delta, sigma), K, n=200_000)
    print(f"   [N] SNR:     W <= {a.w:.0%}  ->  t = {tN:6.3f}  gamma_lo = {glo_N:.5f}")
    print(
        f"   [S] support: init within kappa={a.kappa} of the noise radius sqrt(K-1)={np.sqrt(K - 1):.3f}"
    )
    print(f"                             ->  t = {tS:6.3f}  gamma_lo = {glo_S:.5f}")
    verdict = "COMPATIBLE (S >= N)" if tS >= tN else "INCOMPATIBLE (S < N)"
    print(f"   ratio t_S/t_N = {tS / tN:.2f}   ->  {verdict}")
    print(f"\n   {'gamma':>8} {'decode':>8} {'rho':>8}   (rho stderr ~{se0:.4f})")
    for g_ in (0.005, 0.010, 0.0155, gs):
        rr, _ = rho(t_of(g_, Delta, sigma), K, n=120_000)
        print(f"   {g_:>8.4f} {A_K(t_of(g_, Delta, sigma), K):>8.3f} {rr:>8.3f}")

    print("\n4. RESCALE CONSTANTS [EXACT] — independent of gamma*")
    print(
        f"   {'alpha':>6} {'gamma_match':>12} {'|x_gm|':>8} {'|z_a|':>9} {'g':>9}"
        f" {'t':>8} {'1/alpha':>8} {'A_K(t_z)':>9}"
    )
    # t(gamma_match(alpha)) == 1/alpha EXACTLY (docs/band_geometry_theory.md 5.2.1):
    # sigma, Delta and K all cancel. Asserted, not just printed -- this repo has no
    # test suite, and the identity is load-bearing for reading the recovery ladder
    # as a sweep over t. A_K(t_z) uses z_alpha's OWN margin, which is larger by
    # sqrt(K/(K-1)) because the perturbation is not projected onto V_d; that column
    # is what token_acc_perturbed_band should match.
    rescale_rows = []
    for al in [float(x) for x in a.alphas.split(",")]:
        nx, gm = g_for_alpha(al, Delta, sigma, n0, n1)
        za = n1 * np.sqrt(1 + K * al * al)
        t_land = t_of(gm, Delta, sigma)
        assert abs(t_land - 1.0 / al) < 1e-9, (
            f"t(gamma_match({al})) = {t_land} != 1/alpha = {1 / al}"
        )
        t_z = np.sqrt(K / (K - 1)) / al
        print(
            f"   {al:>6.2f} {gm:>12.5f} {nx:>8.4f} {za:>9.3f} {nx / za:>9.5f}"
            f" {t_land:>8.4f} {1 / al:>8.4f} {A_K(t_z, K):>9.4f}"
        )
        drv = DRIVER_G.get(al)
        rescale_rows.append(
            {
                "alpha": al,
                "gamma_match": float(gm),
                "x_gamma_norm": float(nx),
                "z_alpha_norm": float(za),
                "t": float(t_land),
                "decode_A_K": float(A_K(t_z, K)),
                "driver_g": drv,
                "driver_rel_err": None if drv is None else float((drv - gm) / gm),
            }
        )
    print("\n" + "=" * 74)

    if a.json:
        out = {
            "_note": (
                "EXACT constants of docs/band_geometry_theory.md, written by "
                "scripts/band_geometry.py --json. Everything here is closed form or "
                "1-D quadrature (scipy.integrate.quad) + Brent inversion "
                "(scipy.optimize.brentq); no Monte Carlo, no GPU."
            ),
            "inputs": {"K": K, "eps": eps, "sigma": sigma, "acc": a.acc},
            "scalars": {"Delta": float(Delta), "x1_norm": float(n1), "x0_norm": float(n0)},
            "gamma_star": {
                f"{acc:g}": {"t_star": t_, "gamma_star": g_} for acc, (t_, g_) in gs_table.items()
            },
            "decode_at_gamma": {
                f"{g_:g}": float(A_K(t_of(g_, Delta, sigma), K)) for g_ in (0.005, 0.01, 0.03, float(gs))
            },
            "rescale": rescale_rows,
        }
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
