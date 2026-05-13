# Positioning — conservative energy-based diffusion language models

This document scopes the capstone contribution against the prior work it
needs to cite, and pins down the three claims (A, B, C) the experimental
program is designed to substantiate. Companion artefacts:
[`TRAINING_PLAN.md`](TRAINING_PLAN.md) (execution), `sweeps/dsm_vs_eqm.yaml`
(sweep spec), [`NOTE_WHY_EBM_INIT_STUCK.md`](NOTE_WHY_EBM_INIT_STUCK.md)
(failure-mode theory the project rests on).

## Headline

We study **conservative-energy parameterisations** of noise-conditional
score models on a continuous lift of discrete text. The repo's
Equilibrium Flow Matching (EqM) is one such model: it parameterises a
scalar energy $E_\theta(x) = \langle x, f_\theta(x)\rangle$ and trains
its gradient by flow matching. EqM fails on the larger d=1024 recipe
in a way `NOTE_WHY_EBM_INIT_STUCK.md` predicts mathematically: the FM
regression's Bayes-optimum on Euclidean lifts of discrete data is the
*single unigram attractor*, regardless of metric or chart.

The contribution: **swap the training signal from flow matching to
denoising score matching while keeping the conservative-energy
parameterisation**, run a controlled three-cell comparison on text8,
and report what is gained / lost.

## Three claims (A, B, C)

**(A) Energy-gradient vs. direct-score parameterisation on text.**
Standard diffusion LMs parameterise the score directly as a network
output, with no guarantee that the learned field is the gradient of a
scalar — see (Salimans & Ho, 2021) for the image-domain analysis. We
instantiate both parameterisations in the same backbone, on the same
data, with the same DSM training signal, and report perplexity vs.
recovery / basin-structure trade-off. No prior text-domain head-to-head
exists.

**(B) The §3 attractor diagnosis as a measurable phenotype.**
`NOTE_WHY_EBM_INIT_STUCK.md` argues that FM regression has a single-basin
Bayes-optimum on Euclidean lifts of discrete data while DSM does not.
We turn this into an experimental claim: run EqM-FM, ScoreDSM, and
EqMDSM on identical hardware and report predicted phenotypes —
flat-field unigram bowl for FM (matching the v3 phenotype we already
diagnosed in `runs/ae_d1024_l8_z128_v3/`) versus carved multi-modal
basins for both DSM variants. The diffusion-LM literature does not
measure basin structure; the EqM literature did not have a controlled
comparison.

**(C) Recovery / healing as a diffusion-LM benchmark.**
Existing diffusion-LM papers report perplexity / generation BLEU. We
add a recovery benchmark: perturb held-out text by α·embed_norm and
measure token-level recovery accuracy as the chain anneals back. This
is the natural EBM test for "do basins exist around real data" and is
under-reported. We use the existing `scripts/recovery_check.py` test
harness from this repo, extended for σ-conditional models.

## Where this sits in the literature

### Diffusion language models on continuous lifts

- **Diffusion-LM** (Li et al., 2022, NeurIPS) — Gaussian diffusion on
  learned token embeddings with rounding to discrete tokens. Direct-score
  parameterisation. The canonical reference for "diffusion on a
  continuous lift of text." We do not aim to beat its perplexity; we
  ask a different question about its energy structure.
- **SSD-LM** (Han et al., 2023) — semi-autoregressive Gaussian diffusion
  on continuous embeddings. Direct-score head.
- **CDCD** (Dieleman et al., 2022) — Continuous Diffusion for Categorical
  Data. Cross-entropy training signal, direct-score head.
- **Plaid 1B** (Gulrajani & Hashimoto, 2024) — likelihood-bounded
  continuous-embedding diffusion LM. Direct-score.

These set the "DSM on continuous lift of text" baseline. The training
signal is the relevant axis for us; the discrete-side machinery
(rounding, anchored losses, classifier-free guidance) is largely
orthogonal and we hold it constant via the frozen-AE setup.

### Discrete-state-space diffusion / score models

- **SEDD** (Lou et al., 2023) — score-based discrete diffusion via
  concrete-score / ratio matching. No continuous lift; the score is a
  rate function on $\{1,\ldots,K\}^L$. Orthogonal to our parameterisation
  axis (their score is also not a gradient of a scalar; we share the
  "conservativity?" question).
- **MD4 / MDLM** (Shi et al., 2024) — masked discrete diffusion;
  perplexity-matched to autoregressive LMs. Different design space, but
  recovery semantics are well-defined on their lattice.

We do not contest these. They occupy the discrete-state cell of the
design matrix; we sit in the continuous-lift cell and study the
parameterisation knob within it.

### EBM training-signal axis

- **Du & Mordatch** (2019) — image EBMs via contrastive divergence with
  Langevin negatives. The canonical "MLE-trained EBM with conservative
  energy". Not applied to text successfully at scale.
- **Salimans & Ho** (2021) — "Should EBMs model the energy or the score?"
  for images. Found: energy parameterisation is harder to train, but
  delivers exact log-likelihood and recoverable $Z$. **This is the
  paper we extend to text.** Their conclusion was image-specific; the
  text answer is not in the literature.
- **Bao et al.** (2023) — equivalence between consistency models and
  DSM under specific weightings; useful background for the σ-schedule
  derivations.
- **Bakhtin et al.** (2021) — Residual EBM for text generation. Tried
  contrastive divergence on text; reported the difficulty (negative
  sampling for text is the bottleneck). We sidestep that bottleneck by
  using DSM, where "negatives" are noise-perturbed positives.

### Equilibrium Flow Matching, on which the EqM baseline rests

The EqM construction in this repo trains the conservative gradient of
$E_\theta(x) = \langle x, f_\theta(x)\rangle$ by FM regression. The
recipe is documented in `comp_runs_explainer.md` and the failure mode
in `NOTE_WHY_EBM_INIT_STUCK.md`. Related public methods:

- **Equilibrium Matching for generation** (Geffner et al., variants in
  the FM-for-images line) — the conservative-gradient FM idea
  predates this repo; we use it as the *parameterisation choice* and
  swap the training signal.
- **Riemannian Flow Matching** (Chen & Lipman, 2023) — generalises FM
  to manifolds. The simplex variant is conceptually adjacent; we
  remain in Euclidean charts (CLR / AE-latent) because the EBM
  diagnostics (curvature, recovery) are easier there.

## What we are *not* claiming

- We are not introducing a new diffusion LM method.
- We are not improving perplexity over Diffusion-LM / Plaid.
- We are not solving discrete EBMs (SEDD's territory).
- We are not making claims about generation quality on benchmarks
  where autoregressive LMs are saturated.

## What we are claiming, precisely

1. **EqM-FM exhibits a measurable single-basin collapse on text in the
   d=1024 / variable-length / 50k-windows recipe**, with a phenotype
   (no-op recovery, unigram-flat energy) we have already documented in
   `runs/ae_d1024_l8_z128_v3/`. This collapse is *predicted* by the
   §3 / §6 argument of `NOTE_WHY_EBM_INIT_STUCK.md`.
2. **Swapping the training signal to DSM eliminates the collapse**
   on the same backbone, AE, and seed: ScoreDSM and EqMDSM both
   produce non-trivial basins around held-out text (recovery Δ > 0 at
   matched perturbation).
3. **Enforcing the score to be a conservative-energy gradient (EqMDSM
   vs ScoreDSM) is the interesting frontier**: it costs some
   perplexity / DSM-loss, gains recoverable scalar energy with
   diagnosable basin geometry. The empirical trade-off curve has not
   been reported for text.

The strongest single-headline finding the experiment can deliver is
either:

- "Energy-gradient DSM matches direct-score DSM on perplexity within
  X% while delivering recoverable energies and a curvature-based
  uncertainty surface" — supporting Salimans-Ho's image conclusion
  on text, or
- "Energy-gradient DSM lags direct-score DSM by Y% perplexity but
  exposes basin structure that direct-score DSM cannot represent;
  the right design depends on what you need from the model" — a
  honest trade-off paper.

Either is publishable. The §3 diagnosis and the recovery benchmark
travel as supporting analyses regardless of which way (3) lands.
