# Non-Autoregressive GP Auditor: Findings

*Date: 2026-06-23. Frozen model: `runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM_continue/epoch_final.pt`
(DirichletFM, K=27, L=256, d_model=1280, t_eval=4.5). Code: `scripts/heal_dirichlet.py`
(`--localizer {linear,gp}`, `--gp-mode {oneclass,contrastive}`, `--uq gp_variance`);
sweeps in `scripts/scratch_*.py`.*

## Goal / framing

We want OOD / hallucination detection **and** healing in a **non-autoregressive (non-AR)**
setting — using a flow-matching model on the simplex (DirichletFM) and its own
features/energy/uncertainty, **without** an autoregressive LLM.

The reference (supervisor's *Discrete Hilbert Flow Matching*) gets token-level AUROC
**0.96–0.996** and sequence-level **0.998** on WikiText-2 — but those use **GPT-2 / Qwen2.5
logit distributions** and **Spilled Energy**, which are **autoregressive by construction**
(Spilled Energy is a chain-rule-consistency property of an AR factorization; top-K logits are
the AR next-token distribution). Getting top numbers there is trivial *because* a large
pretrained AR model is imported. **That crutch is off the table by design.** Our lower numbers
are the honest cost of the non-AR constraint, which is the contribution.

## Headline: per-token detection AUROC (non-AR, frozen DirichletFM features)

Full-scale (fit=256, held=64 seqs), each discriminative head trained with **matched negatives**
(same corruption it is tested on); one-class trained on **valid data only**.

| axis | linear (discrim.) | gp_energy (Matern) | oneclass_var (valid-only) |
|---|---|---|---|
| substitution | **0.915** | 0.911 | 0.697 |
| shuffle (order) | **0.902** | 0.897 | 0.710 |

Reference (AR, **not** comparable; LM logits, WikiText-2): token 0.96–0.996, seq 0.998.

### Three load-bearing conclusions

1. **The GP energy never beats the linear head** — on either axis, at any inducing count.
   Both substitution and shuffle are **near-linearly-separable** in DirichletFM feature space,
   so the Matern nonlinearity adds nothing for *detection*. Do not pitch the GP as a better
   detector.
2. **One-class variance is NOT blind to order corruption** (0.710 on shuffle, *higher* than
   substitution). Expected near-chance (shuffled tokens are valid / on-manifold), but the
   backbone is **bidirectional/contextual**, so a valid token in the *wrong context* yields an
   *off-manifold contextualized feature*. ⇒ the one-class variance is a **single,
   corruption-agnostic, label-free detector** that catches both substitution and order
   corruption (~0.70) **without matched negatives** — the discriminative heads need a separate
   negative set per corruption type. This is the cleanest edge for the label-free non-AR story.
3. **One-class variance caps at ~0.70.** Substitution/shuffle are subtle in a small char-level
   denoiser's features; a label-free distance-to-manifold signal is intrinsically blunter than a
   discriminative head that learns the corruption direction.

## Why the one-class variance caps at ~0.70 (every lever swept)

All sweeps on raw standardized 1280-d features, one-class SVGP (regression-to-0 on clean,
inducing seeded on the clean manifold, small fixed lengthscale). Metric: held-out per-token
variance AUROC (substitution).

- **Lengthscale** (`scripts/scratch_gp_oneclass_sweep.py`): `ls_scale` 0.05–0.10 → ~0.70;
  0.15 → 0.58–0.63. **Small lengthscale helps** (sharp variance contrast off-manifold); large
  lengthscale flattens variance to a constant.
- **t_eval** (`scratch_gp_teval_sweep.py`): 1.0 → 0.49 (chance; near-uniform input, no token
  signal), 2.0 → 0.67, 3.0 → 0.68, **4.5 → 0.71**, 6.0 → 0.713, 7.8 → 0.71. Plateau past ~4.5.
- **Inducing points M** (`scratch_gp_M_sweep.py`): 256 → 0.699, 512 → 0.704, **1024 → 0.707**,
  2048 → 0.706, 4096 → 0.691. Flat, peaks at 1024, then declines. **More inducing points do not
  help** — consistent with the reference getting 0.96 at M=100. The cap is a *signal* ceiling,
  not an *approximation* ceiling.
- **Downsampling / projection** (`scratch_gp_proj_sweep.py`): raw 1280-d → **0.704**;
  learned DKL projection (32–256) → 0.693 (identical across dims = **feature collapse**);
  random projection (32–256) → 0.59–0.62. **Downsampling hurts.** The OOD signal lives in
  specific high-dim directions; PCA/random/learned projection discards more signal than the
  kernel-concentration fix recovers. Matches the repo's own `BayesLinHead` choice of
  `pca_dim=0` (full dim).

## GP-vs-BLR variance (why the *contrastive* GP variance was dead, and BLR works)

The SVGP predictive variance depends only on distance-to-inducing-points and the variational
covariance — **never on labels**. Training it discriminatively (corrupt negatives, free inducing
points, learned projection, ELBO/KL pressure) makes the inducing set cover *both* manifolds and
the variance collapses to a constant (observed: `V_valid == V_corrupt`). BLR's variance
`zᵀ(Φ+λI)⁻¹z` is an **exact, closed-form, one-class Mahalanobis** on the full clean covariance —
large in the low-variance directions where OOD lives — and is never optimized, so it cannot
collapse. The fix that revives the GP variance is exactly the one-class recipe above (fit on
clean only, inducing on the clean manifold, small fixed lengthscale) — which makes the SVGP a
Matern-kernel analog of BLR.

## Hybrid deliverable: discriminative localization + one-class variance UQ

Healing run: `--localizer linear --uq gp_variance`, corrupt_rate 0.15, n_demo 16, n_seeds 5,
nfe 100. Linear is the localizer (best detector); the valid-only GP variance is a parallel
calibrated-uncertainty channel.

**Healing (linear localizer, selected fpr=0.02):**
locP 0.624, locR 0.737, fix_rate 0.479, damage_rate 0.056, **net/corrupt +0.165 ± 0.036**.

**UQ channel (valid-only one-class GP variance):** the variance *mean* is near-constant
(~0.083 everywhere) but its *ordering* carries the signal.

| UQ metric | all corrupt | flagged-only (confound-controlled) |
|---|---|---|
| detection (pre-heal corrupt vs clean) | 0.659 | — |
| pre-heal variance predicts UNFIXED | 0.969 | **0.830** |
| residual (post-heal) variance flags UNFIXED | 0.781 | **0.707** |

The naive 0.969 was inflated by a confound (un-flagged corruptions are never healed → trivially
"unfixed"). Restricted to **flagged** corrupt tokens (heal actually ran), the pre-heal variance
still predicts heal-failure at **0.83**, and residual variance flags still-broken positions at
**0.71**. So the UQ channel earns its place: detection is weak (handled by the linear head), but
**heal-confidence** — *which corruptions the inpainter can't fix* — is genuine and is something a
discriminative head structurally cannot provide. Mechanism: the most off-manifold corruptions
are the hardest to heal, and the variance flags them ex-ante.

## Recommendations

- **Detection (with labels):** linear discriminative head on frozen DirichletFM features
  (~0.90 on both axes). The GP is not a better detector; don't use it for detection.
- **Label-free detection:** one-class GP variance (~0.70, corruption-agnostic across both axes).
- **UQ / heal-confidence:** one-class GP variance (0.83 flagged-only) — the GP's real niche.
- **GP settings:** raw standardized features (`--gp-d-latent 0`), small **fixed** lengthscale
  (`--gp-lengthscale-scale 0.1`), M≈512–1024 (more is wasted). No downsampling.
- **The contribution is the non-AR setting + transport-based healer + label-free UQ**, not a
  detection-AUROC win (only AR LM-logit methods get that, trivially).

## Open / next

- **Healer on the shuffle axis:** does the transport-based masked-Euler inpaint repair *order*
  corruption (not just substitution)? A non-AR bidirectional field could repair by global
  re-arrangement rather than left-to-right — a genuinely non-AR healing test.
- **Non-AR screening signal:** compare `DenoiserNLL` (`scripts/ood_denoiser_nll.py`) vs the
  GP/linear head as the non-AR analog of "Spilled Energy (seq) vs GP (token)" complementarity.
