# Flow Matching for Text Generation & OOD — Findings Summary and Results Structure


> **MOSTLY CURRENT (2026-06-06), with one section that is now wrong. §1.1, §1.2 and §1.4 are the paper's conclusion chapter in draft and still hold. §1.3 and the OOD bullets of §2 are superseded and inverted.**
>
> What changed: the "Spilled Energy beats trained probes" claim below rested on a GPT-2 baseline that was computing a per-token NLL under the wrong name and attributing byte-pair scores down onto characters by uniform spreading (corrected 2026-07-13). At the fair word unit our training-free denoiser NLL localizes replaced words at 0.981 against 0.727 for spilled energy and 0.738 for the GPT-2 likelihood (`tab:ood-word`), and the gap survives thresholding. So §1.3's conclusion and §2's "even with home-field advantage the FM signal loses to a zero-shot LLM method" are backwards on the geometric axis. The surviving scoped version: on CONTEXTUAL corruption (false information) the pretrained model is the better sequence-level triage model.
>
> Of §4's proposed experiments, P1, P2, P4 and the t-sweep half of P5 were run and are in the paper; the epsilon sweep, P3 (probability-flow ODE likelihood baseline) and P6 (curl fraction) were not.

This document has four parts: (1) the established conceptual findings to use as the spine of the
write-up, (2) a soundness checklist for the methodology, (3) a section-by-section results structure,
and (4) a prioritized list of additional experiments.

---

## 1. Established findings (the conceptual spine)

### 1.1 The mechanism behind the collapse
- **Fixed-interpolant velocity regression collapses to the unigram, and this is a property of the optimum, not of capacity.** With $x_t=(1-t)x_0+tx_1$ and L2 target $\dot x_t$, the minimizer is the conditional mean $\mathbb{E}[\dot x_t\mid x_t]$. For categorical data embedded in $\mathbb{R}^d$ this mean is the marginal (unigram): the field points at the centroid of the simplex, not the vertices, so the ODE parks there. An infinite-capacity model has the same fixed point — so "small model" does not threaten this result.
- **Failing class:** linear/Gaussian-path Flow Matching (Lipman), Equilibrium Matching (a deterministic gradient field), and any $v_\theta=-\nabla_x E_\theta$ parameterization.
- **Escaping class:** cross-entropy / KL posterior-on-simplex denoisers — Dirichlet FM, Fisher/Spherical FM, Variational Flow Matching. They predict a *distribution* over the endpoint, which is multimodal-aware and does not average to the mean.
- **The correct cut is distributional vs point prediction, not "denoiser vs velocity."** For Gaussian/linear paths, $x_1$-prediction, velocity, noise, and score are affine reparameterizations, so an L2 *point* $x_1$-denoiser collapses exactly as hard as L2 velocity regression. What breaks the collapse is predicting a distribution with CE/KL, not predicting $x_1$ as a point with L2.

### 1.2 The central thesis: explicit energy and categorical generation are mutually exclusive
- **Explicit, readable energy for OOD** $\iff v_\theta=-\nabla_x E_\theta$ (point vector regression) $\implies$ collapse on categorical data.
- **Competitive categorical generation** $\iff$ CE/KL posterior $\implies$ a softmax over the vocabulary, which is not $\nabla_x$ of any scalar in embedding space $\implies$ no explicit energy.
- Therefore a continuous-state flow matching model cannot simultaneously (a) generate text competitively and (b) expose a conservative-field energy that hands you OOD for free.
- **Airtight caveat to state explicitly:** the *optimal marginal field of a perfect denoiser* is itself conservative for Gaussian paths (affine in the score), so the conflict is between an *explicit/readable* energy and generation — **not** between conservativity-at-the-optimum and generation. Recovering a scalar from the learned field means either integrating an imperfect, non-conservative learned field (path-dependent, ill-defined) or falling back to the probability-flow likelihood — which is the unreliable OOD signal anyway.
- **Equilibrium Matching (arXiv 2510.02300)** is the canonical named instance of the failing class. Its gradient vanishes on the data manifold and grows toward noise — exactly the wrong geometry for text, where valid sequences need sharp corrections *at* the basin. It is an image paper, untested on text; your experiments are the demonstration that this branch collapses where the CE-posterior branch escapes.

### 1.3 The OOD finding
- Density-based OOD from deep generative models is unreliable (Nalisnick et al. 2018; reaffirmed in the energy-OOD literature, Liu et al. 2020).
- The signal that works is **discriminative**, not a generative potential: a simple linear projection of the generated text carries strong OOD signal, and Spilled Energy (arXiv 2602.18671) reads energy off the autoregressive softmax (a discriminative head), is training-free, and beats trained probes.
- Conclusion: the conservative field is **neither necessary nor sufficient** for competitive text OOD.

### 1.4 Scope discipline (what to claim, what not to claim)
- **Do not claim** "continuous flow matching cannot generate text" — Dirichlet/Fisher FM do. Claim the **exclusivity** instead.
- **Do claim:** (i) fixed-interpolant L2 continuous FM collapses to the unigram (a now-understood loss/target artifact, capacity-independent); (ii) the known continuous-state fixes recover uncompetitive generation but are incompatible with a conservative field; (iii) the specific target — continuous FM as an EBM for text generation *and* OOD — is therefore structurally precluded; (iv) the useful OOD signal is discriminative anyway.
- **Out of scope but worth a forward-looking paragraph:** the discrete-state route (SEDD's concrete-score *implicit* energy via consistent ratios; EDLM's sequence-level energy on a discrete-diffusion backbone) is the structurally-consistent home for "denoising + energy," because the discrete analog of a conservative gradient field is detailed balance / reversibility, not $\nabla_x$.

---

## 2. Methodology soundness checklist

### Solid
- The pipeline is legitimate and unusually thorough for a capstone; the breadth of the FM sweep is a real strength.
- The core negative result is capacity-independent (optimum = conditional mean), so limited compute does not threaten it.
- Char-level text8 at $K=27$ rules out the large-$K$ simplex pathology as the cause of collapse, isolating the regression-target mechanism.
- The Hilbert/Aitchison input geometry is correctly motivated: the normalizer genuinely cancels because the distance is a function of the CLR coordinate $\log(p/G(p))$ and is projective (Nielsen & Sun 2023). The smoothing is required to keep one-hot points in the open simplex so the log is defined.
- The "Hilbert vs MSE → no impact" result is **confirmatory**, not a null: the geometrically-correct metric fails to rescue the collapse, which points the finger at the point-estimate target rather than the loss geometry.
- Giving the OOD detector outlier exposure (a contrastive hinge with negatives) and still losing to zero-shot Spilled Energy makes the negative conclusion stronger, not weaker.

### Tighten
- "Normalizer cancels under log" holds under the **Hilbert/Aitchison metric (= CLR)**, not under plain MSE in raw log-$\mathbb{R}^d$ (not shift-invariant). State which loss you actually optimized; if MSE-in-log, you were in Euclidean (not projective) geometry and the cancellation argument does not apply.
- Nielsen & Sun (2023) is a **graph-embedding** paper. Cite it for the input geometry and the Hilbert loss, not as endorsement of the generative pipeline.
- The OOD comparison conflates method with scale + pretraining (small from-scratch FM vs large pretrained LLM) and is asymmetric (trained-with-negatives vs zero-shot). Reframe as practical dominance — "even with outlier exposure and home-field advantage, the FM signal loses to a zero-shot LLM method" — not as a controlled method-vs-method comparison.

### Add
- A **single-variable causal experiment** isolating the prediction target (see P1).
- **Quantitative collapse metrics** instead of "poor quality" (see P2).
- A **probability-flow ODE likelihood baseline** for OOD (see P3).
- A **sampler-only ablation** to separate stochasticity from loss (see P4).
- **Hyperparameter ablations**: smoothing $\varepsilon$, and the timestep $t$ at which FM features are read for the OOD head (FM features are time-dependent) (see P5).
- An explicit **OOD benchmark specification**: what counts as OOD (shuffled character order, other language, random sequences, semantic shift), and near vs far OOD.

---

## 3. Proposed results structure

For each section: the **claim**, the **experiment**, the **metric**, and the **baseline/ablation** that makes it defensible.

**S1 — Setup and representation.**
Claim: a principled compositional embedding of text8. Experiment: one-hot → smoothed → CLR/Hilbert log representation; transformer backbone with positional encodings. Metric: n/a (describe). Baseline/ablation: smoothing $\varepsilon$ sweep; log vs CLR vs ILR (report the no-impact finding and attribute it correctly).

**S2 — Generation: the taxonomy sweep.**
Claim: only distributional (CE/KL posterior) FM variants escape the trivial solution. Experiment: Lipman FM, Equilibrium FM, Dirichlet FM, Discrete FM, Spherical/Fisher FM under a shared backbone. Metric: n-gram KL + per-position entropy + samples. Baseline/ablation: Hilbert vs MSE loss (reframed as confirmatory).

**S3 — Mechanism: why fixed-interpolant FM collapses.**
Claim: the collapse is the L2 optimum (conditional mean = unigram), capacity-independent. Experiment: the target-toggle (P1). Metric: unigram KL → 0 vs bigram/trigram KL large; show the same backbone flips. Baseline/ablation: sampler-only toggle (P4) to rule out stochasticity as the lever.

**S4 — Local structure: the corruption/healing sweep.**
Claim: usable local gradient structure exists only in a mid-corruption band; the basin is flat. Experiment: feed 0–50% corrupted sequences, measure recovery. Metric: % positions healed vs corruption level (your 6–8% peak at 20–30%, ~0 at 0–20%). Use this to bridge to per-token OOD.

**S5 — OOD detection.**
Claim: the FM-derived signal is uncompetitive; the working OOD signal is discriminative. Experiment: SVGP on FM features (specify $t$) with a contrastive energy hinge; per-token and per-sequence uncertainty. Metric: AUROC, near and far OOD. Baselines: PF-ODE likelihood (P3, expected to fail), linear probe on generated text, Spilled Energy on an LLM. State the fairness caveats.

**S6 — Synthesis: the exclusivity result.**
Claim: continuous-state FM cannot do competitive text generation and supply a conservative-field OOD signal simultaneously. Argument: §1.2, with EqM as the named failing instance and S2–S5 as evidence. Close with the discrete-state route as the structurally-consistent alternative (out of scope, future work).

---

## 4. Prioritized additional experiments

- **P1 (causal, highest value).** Fix backbone, data, and sampler; vary only the prediction target — L2 point-regression (velocity or $x_1$) vs CE/KL posterior on the simplex. Show collapse ↔ generation toggles. This converts the taxonomy from circumstantial to causal.
- **P2 (cheap, high value).** Quantify collapse: unigram KL, bigram/trigram KL, per-position entropy, for each variant. One table.
- **P3 (cheap, closes a hole).** Probability-flow ODE log-likelihood as an OOD score, to demonstrate it is unreliable (Nalisnick) and motivate the discriminative signal.
- **P4 (cheap).** Sampler-only ablation: one trained model, deterministic ODE vs stochastic SDE sampling, to separate stochasticity from the loss/target.
- **P5 (cheap).** Ablate smoothing $\varepsilon$ (sets the embedded scale/geometry) and the OOD feature-extraction timestep $t$.
- **P6 (optional, strong if it works).** Estimate the rotational/curl fraction of the learned field for collapsing vs escaping variants, to quantify the conservative-vs-not mismatch directly.

---

### Key references
- Equilibrium Matching — arXiv 2510.02300 (failing class, image-only).
- Energy Matching — arXiv 2504.10612; Variational Potential Flow Bayes — arXiv 2504.16262 (FM↔EBM unification, continuous).
- Dirichlet Flow Matching — arXiv 2402.05841; Variational Flow Matching (escaping class, CE/KL posterior).
- Energy-Based Diffusion Language Models (EDLM) — arXiv 2410.21357; SEDD / concrete score — Lou et al. (discrete-state "denoising + energy").
- Spilled Energy — arXiv 2602.18671; Energy-based OOD — Liu et al. 2020; Nalisnick et al. 2018 (discriminative OOD; density unreliable).
- Nielsen & Sun 2023, Non-linear Embeddings in Hilbert Simplex Geometry (input geometry).
