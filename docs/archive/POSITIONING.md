# Positioning — conservative energy-based diffusion language models


> **HISTORICAL (2026-05-13). A pre-pivot strategy document. The contribution it scopes is not the contribution the paper makes, and its Claim 2 is contradicted by the paper's central mechanism.**
>
> The project did not swap flow matching for denoising score matching. DSM survives in the paper as two cells of the 2×2 ablation (`tab:ablation2x2`), which confirm this document's sign — a noisy-view target escapes the unigram collapse — while showing it does not buy generation, because it remains a pointwise L2 regression (paper §Energy and generation).
>
> **Claim 2 is wrong where it matters.** This document attributes the recovery deficit to continuous-Langevin instability on the log-simplex, "not to model defect", and names a Gibbs sampler as the appropriate intervention. The paper's finding is the opposite: the failure is training-time and no sampler repairs it. Injected Langevin noise buys 9–19% on the bigram divergence, leaves recovery a no-op, and does not approach the transport arms (paper app. §Sampling-time failure). `CAPSTONE_SUMMARY.md` carried an action item to rescope this claim; it was never executed.
>
> Also superseded: the closing "pass-grade story" scoping. The paper reports a working non-autoregressive generator, a detector suite, and a repair loop.
>
> Still worth keeping: the literature positioning against the continuous-lift diffusion-LM line (Diffusion-LM, SSD-LM, CDCD, Plaid) and the Bakhtin residual-EBM-for-text precedent, none of which the paper cites. Corrected pointer: `comp_runs_explainer.md` is at `runs/comp_runs_explainer.md`.

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

## What we are claiming, precisely (revised after `dsm_clr_ablation` and concurrent supervisor work)

1. **EqM-FM (Wang & Du 2025, §4.2 Eq. 7 Dot Product variant) exhibits
   a measurable single-basin collapse on vertex-supported discrete data
   in the d=1024 / L=40 recipe**, with a chart-invariant no-op
   recovery phenotype documented in (a) `runs/ae_d1024_l8_z128_v3/`
   (AE-latent), (b) `runs/comp_ref_det_mse/` (CLR), and (c) concurrent
   supervisor work on DNA promoter Hilbert FM (CNN). The collapse is
   *predicted* by the §3 / §6 argument of `NOTE_WHY_EBM_INIT_STUCK.md`,
   and the prediction is **chart-, architecture-, and domain-invariant
   within the FM and flow-map signal classes** (`NOTE` §7.5).

2. **The training signal is the load-bearing axis for the
   *unconditional* component of the collapse, but not for recovery
   at this scale.** Specifically (`runs/dsm_clr_ablation/`, 2×2 over
   training signal × x₁ recipe, d=1024 / 8L / K=27 / L=40, 3 seeds for
   the FM cells):

   - **DSM with deterministic CLR `x_1`** achieves KL$_\text{uni}=0.63$ —
     comparable to the compositional FM cells (KL$_\text{uni}=0.67$) and
     **distinct from the FM-deterministic-CLR no-op signature**
     (KL$_\text{uni}=0.04$, the unigram peak). Switching the training
     signal alone, *holding the data recipe fixed*, defeats the §3
     mechanism on the unconditional axis.
   - **DSM does not transfer to recovery** at this scale
     (Δ@.50 ≈ −0.02 for both `x_1` recipes), and Dirichlet thickening
     does not help DSM the way it helps FM (Δ@.50 = +0.060 ± 0.000
     across 3 seeds for FM-Dirichlet; the only positive cell in the
     2×2).
   - Concurrent supervisor work (healer convergence study) attributes
     the recovery deficit on GP-EBM-class models to **continuous-Langevin
     instability on log-simplex**, not to model defect; discrete Gibbs
     samplers on the same trained fields recover ~1.22 nats LM log-prob.

   The scoped honest claim is therefore: *DSM eliminates the
   unconditional-KL component of the §3 collapse; the recovery
   component lives on the sampler axis and requires a separate
   intervention (concurrent supervisor work shows Gibbs[SE∪GP] is the
   appropriate intervention).*

3. **The KL$_\text{uni}$ and recovery axes are empirically
   decoupled** (`runs/comp_*` + `runs/dsm_clr_ablation/`): the cell
   with best KL$_\text{uni}$ (FM-det-CLR at 0.04) has the worst
   recovery (Δ@.50 = 0.000); the cell with worst KL$_\text{uni}$
   (FM-Dirichlet at 0.67) has the best recovery (Δ@.50 = +0.06). This
   justifies recovery, not unigram-KL, as the headline metric for
   this model class — a methodological move with no analog in the
   concurrent BPC-centric evaluations.

4. **Enforcing the score to be a conservative-energy gradient
   (EqMDSM vs ScoreDSM)** remains the interesting frontier for
   *fluent unconditional generation*, but the present POC scale does
   not produce numbers strong enough to ground a Salimans-Ho-on-text
   parity claim. Honest scope: at d=1024/L=40 on text8, neither DSM
   variant produces recovery improvement under continuous Langevin;
   the energy-vs-direct-score axis would need to be re-evaluated with
   a Gibbs-class sampler before a parametrisation parity claim can be
   made.

The strongest single-headline finding the project can deliver, given
present evidence, is:

> **"The FM-on-simplex single-basin collapse is a
> training-signal-class phenomenon (`NOTE` §7.5), confirmed across
> three independent empirical instances and partially defeated on two
> axes: data-side Dirichlet `x_1` thickening on FM (Δ@.50 = +0.06,
> 3-seed-replicated) and signal-side DSM on the unconditional KL
> (KL$_\text{uni}=0.04 \to 0.63$, det-CLR cell). Neither fix delivers
> fluent generation on its own at this scale; recovery requires a
> separate sampler-axis intervention (concurrent supervisor healer
> work). The §3 diagnosis and the unified six-axis fix taxonomy
> (`CAPSTONE_SUMMARY.md` §2.2) organise these results into a single
> framework."**

This claim is *modest, empirically defensible, and honest about
scope*. It is *not* "we built a generative LM"; it is "we diagnosed
a class of failure modes, demonstrated controlled partial fixes, and
provided the structural framework that organises the search space."
That is the pass-grade story available from the artefacts in
`runs/`.
