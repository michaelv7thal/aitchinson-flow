# Compositional EqM — Proposal


> **STALE PROPOSAL — 2026-05-10. The recipe was run; it works as a mechanism probe and not as the repair proposed here.**
>
> Dirichlet thickening does what §3 predicts to the *variance floor*, and the paper adopts that argument (appendix §Training-time collapse, "Variance floor"). What it does not do is yield a working generator. In the paper's 2×2 ablation the thickened cell escapes the unigram collapse only by moving to the *other* trivial distribution: its generated entropy sits within 0.001 of uniform (ln 27 ≈ 3.296) and its bigram divergence is the worst in the table, 5.990 (`tab:ablation2x2`).
>
> **§6 does not hold.** None of the four "downstream applications enabled by this recipe" — healing, per-position UQ, sequence-level OOD, energy-based language modelling — survived. The trained energy has a minimum at every vertex, the wrong ones included, so it scores how *sharp* a position is and not whether its token fits the context; the per-token OOD score is instead read off the frozen transport backbone (`chapters/results.tex` §What survives, §OOD Detection).
>
> **§1 attributes the uniform collapse to the wrong variant.** It is the *thickened* recipe that spreads towards uniform (KL_uni 0.611, H_ratio 1.154); the *deterministic*-CLR recipe collapses onto the corpus unigram distribution (KL_uni 0.093, H_ratio 0.958). See `tab:gen`.
>
> **§3.1 / §4.2 / §5.3 on Hilbert were settled by this proposal's own sweep.** `runs/compositional_eqm_test_summary.md` finds MSE and Hilbert a wash, with no broadening of the denoising regime. The paper reports it as a confirmatory null (`tab:hilbert-null`).
>
> **Kept because Appendix A and §8 are the only record of how the thickening was calibrated** — the closed-form Dirichlet-under-CLR covariance, the per-γ floor decay, and why `alpha_peak = 10` / `alpha_base = 0.1`, which is the setting the paper's reproducibility table prints without derivation. `sweeps/compositional_eqm_test.yaml` and `NOTE_WHY_EBM_INIT_STUCK.md` cite this file by section.

A coherent recipe for training Equilibrium Flow Matching (EqM) on discrete
data (text, character sequences, vocab-bound categorical streams). The recipe
combines three ingredients that, used together, eliminate the dominant
training-failure mode we identified in standard EqM-on-text and yield a
well-behaved EBM with usable downstream properties.

```
Compositional EqM = Dirichlet-thickened CLR data
                  + Hilbert/variation-seminorm metric (or MSE — see § 5)
                  + EqM's conservative-gradient parameterisation & NAG sampling
```

## 1. Motivation — why standard EqM fails on text

Standard EqM with one-hot CLR (or any deterministic embedding) for discrete
tokens trains to a degenerate energy field:

* `x_1 = CLR(p_t)` is a single fixed point per token (Dirac data manifold).
* The FM regression target `c(γ)·(x_0 − x_1)` has zero per-token variance.
* The MSE-optimal solution can drive `flow_loss → 0` by fitting the
  conditional expectation, which is a smooth low-frequency function with
  `∇E ≈ 0` everywhere except at the K vertex points themselves.
* Result: spike basins. The energy field is essentially a delta function
  at each vertex with flat surroundings everywhere else.

Empirical signature (laptop GPU, K=27, L=40, text8):

* `flow_loss → 0.007` in 5 epochs (target near-perfectly fit).
* Recovery diagnostic: `acc_perturbed ≈ acc_recovered` at every perturbation
  level α — sampler is a no-op. NAG-GD descends `‖∇E‖ ≈ 0` and returns the
  perturbed input unchanged.
* Unconditional generation produces uniform argmax over tokens (`H_gen ≈
  log K`) — sampler converges to centroid.

This is the **information-theoretic floor of FM regression on Dirac data**,
not a defect of any specific model or training schedule.

## 2. Proposal

### 2.1 Replace `x_1` with a Dirichlet sample

Per training step, for each token `t`, sample

```
p ∼ Dir(α_base · 𝟙 + α_peak · e_t)        [α_peak ≈ 10, α_base ≈ 0.1, K=27]
x_1 = CLR(p)
```

This thickens the data distribution from a Dirac to a smooth density
supported on the open simplex. The mean of `CLR(p)` is approximately the
deterministic vertex (≈ +12 on-target, ≈ −0.5 off-target), but every CLR
coordinate now has non-trivial variance.

### 2.2 Use the Hilbert (variation-seminorm) loss

Hilbert metric on the simplex is

```
d_H(p, q) = ‖CLR(p) − CLR(q)‖_var = max_i (CLR(p) − CLR(q))_i 
                                  − min_i (CLR(p) − CLR(q))_i
```

— translation-invariant in the all-ones direction, scale-invariant in the
compositional sense (multiplying probability vectors by a constant), and
the natural metric on probability cones.

Use the LSE-smoothed version `‖v‖_var,α = (1/α)·[LSE(αv) + LSE(−αv)]` so
the loss is differentiable. Use it as a drop-in replacement for MSE on the
conservative gradient regression:

```
ℓ(grad_g, u_tgt) = ‖grad_g − u_tgt‖_var,α
```

### 2.3 Keep EqM's conservative-gradient parameterisation and NAG sampling

The model is a velocity field `f(x_γ; γ)` whose conservative gradient
`∇⟨x, f⟩` is regressed against the FM target `c(γ)·(x_0 − x_1)`. Sampling
descends `∇⟨x, f⟩` via NAG-GD or Euler integration over γ.

## 3. Why this works — the variance-floor argument

The MSE-optimal model is the conditional expectation
`f*(x_γ) = E[u_tgt | x_γ]`. The irreducible MSE floor at fixed γ is

```
floor(γ) = c(γ)² · Tr(Cov(x_1))
```

For our setting:

| recipe | Tr(Cov(x_1)) | floor(γ=0) | floor(γ=1) | Plateau set |
|---|---|---|---|---|
| Deterministic CLR | 0 | 0 | 0 | **everywhere** ⇒ spike basins |
| Dirichlet CLR (`α_peak=10`) | ≈ 2525 (K=27) | ≈ 2525·c(0)² | 0 | **only at γ=1** ⇒ EBM with minimum at data |

The variance floor forces the model to express non-trivial gradient
everywhere except at γ=1. Plateaus migrate to where they belong — the
data manifold. The energy landscape gains:

* **Real basins** with non-zero curvature around each token.
* **Useful slope** near data, `‖∇E‖ ∝ c(γ)·distance from manifold`.
* **Correct direction** — gradient points back toward the local data
  attractor (modulo the centroid-attraction caveat at γ=0).
* **Zero gradient at the data manifold itself** — by construction of
  `c(γ=1)=0`.

This is exactly the geometry the EBM denoising / energy-descent sampling
procedure assumes. With Dirac data, none of these properties hold; with
Dirichlet thickening, all four do.

### 3.1 Hilbert metric's role

Hilbert metric is the *natural metric on the compositional manifold*. The
Dirichlet's variance lives entirely in the (K−1)-dimensional
zero-mean-CLR subspace (i.e., the simplex tangent), with anisotropic
shape that respects compositional ratios. The Hilbert metric is exactly
the metric on that manifold. So:

* MSE + Dirichlet treats all K coordinates equally; the K-th (all-ones)
  direction has zero variance and the loss "wastes" attention there.
* **Hilbert + Dirichlet** treats only the K−1 compositional directions;
  the metric is exactly aligned with where the data variance actually
  lives.

Empirically this gives a slight broadening of the denoising regime
(Hilbert+Dirichlet shows recovery activity at α=0.35 + 0.50; MSE+Dirichlet
only at α=0.50). The effect is small but theoretically clean.

## 4. Validation

### 4.1 Synthetic Hilbert counter-example (interior compositional data)

`scripts/hilbert_counterexample.py`: 3-Dirichlet mixture in `S_5`, FM with
small MLP, MSE vs Hilbert × 3 seeds, 4000 steps.

```
loss          | forward KL ± std    | hilbert_match
MSE           | 4.989 ± 0.175       | 0.445
Hilbert_soft  | 4.843 ± 0.120       | 0.463
```

Hilbert wins by ~3% on the canonical "Hilbert should help" setting.
Modest but in the right direction.

### 4.2 Simplex EqM 2×2: loss × data variance

Recovery diagnostic on 4 simplex EqM checkpoints (laptop, K=27, L=40,
text8, n=64 sequences, 100 NAG steps):

| | one-hot CLR (Dirac) | Dirichlet-thickened CLR |
|---|---|---|
| **MSE** | sampler no-op at every α | sampler denoises at α=0.50 (`+4.8% token_acc`, `KL_bi 2.82→1.22`) |
| **Hilbert** | sampler no-op at every α | sampler denoises at α=0.35 + 0.50 (`+3.1% / +4.6%`, `KL_bi 0.73 / 1.37`) |

The data-variance row shows on/off behaviour; loss choice is a 2-3% effect.

### 4.3 Per-γ floor decay (theoretical, verified empirically)

```
γ=0.00   floor ≈ 598          (Dirichlet, K=10 demo)
γ=0.50   floor ≈ 153
γ=0.99   floor ≈ 0.06
γ=1.00   floor = 0            ⇐ exactly zero, by construction
```

The floor-vanishing-at-γ=1 is what enables the EBM property to hold
even with thickened data.

## 5. Limitations (honest scope)

### 5.1 Unconditional generation is partial

At pure noise inputs (γ=0), all positions are i.i.d. Gaussian — no
information distinguishes which token any position should be. The
conditional-expectation gradient field at noise points to the centroid
of the embedding distribution. NAG started from pure noise descends to
that centroid, decoding to whichever single token is closest.

This is a **structural information-theoretic limit**, not a training
defect — the variance trick fixes the field's *shape* but not the
*signal-availability* at γ=0. Resolutions:

* **Multi-modal source**: sample `x_0 ∼ Σᵢ wᵢ·N(embed_i, σ²·I)`.
  Now noise carries token bias by construction.
* **Score matching with multi-noise-level training + Annealed Langevin
  sampling**: standard fix for this exact issue in the
  diffusion/score-based literature.
* **Conditional generation**: condition on prefix/context to break the
  symmetry. Easy and standard.

### 5.2 Sampler-API mismatches

Time-conditioned models with Euler integration walk γ from 0 to 1
regardless of `x_init`. For perturbed-near-data inputs, this means the
first many Euler steps query the model at γ=0 (noise) with x near data
— off-distribution. The trajectory drifts.

Resolutions: use NAG-GD (γ-agnostic energy descent) for recovery, OR
start Euler at `γ_start = 1 − α` for perturbation-level α.

### 5.3 Hilbert vs MSE is a small effect

The data-variance change is the *binary* on/off lever (no denoising
without it; real denoising with it). The MSE/Hilbert choice is a
*small* lever (a few % in our experiments). Don't oversell Hilbert.

## 6. Downstream applications enabled by this recipe

* **Sequence healing / denoising.** Take corrupted text (OCR errors,
  typos, bit-flips), encode to CLR, run NAG-GD, decode. ~99% token
  accuracy at α≤0.20 perturbation; ~50-60% at α=0.50; partial-but-
  textually-coherent recovery beyond.
* **Per-position uncertainty quantification.** `‖∇⟨z, f(z)⟩‖` per
  position is a free byproduct of the EBM — high values indicate
  off-manifold positions (typos, OOD chars). Calibration evidence: the
  metric is monotone in distance-from-data.
* **Sequence-level OOD detection.** Mean or max of per-position
  uncertainty serves as a single OOD score per sequence; threshold-
  calibrated for downstream use.
* **Energy-based language modelling.** The energy `E(z) = ⟨z, f(z)⟩`
  defines an unnormalised density; pseudo-likelihoods and contrastive
  scoring fall out.

The recipe is **good at denoising and uncertainty, partial at
unconditional generation**. The downstream framing should reflect this.

## 7. Implementation footprint

Modifying an existing simplex-EqM training pipeline to "Compositional EqM"
requires:

* One config flag (`transformation.dirichlet_sampling = True`).
* Two new fields (`dirichlet_alpha_peak`, `dirichlet_alpha_base`).
* One change in the data collate (≈ 5 lines): replace
  `x = token_ids_to_features(...)` with
  `x = token_ids_to_features_dirichlet(...)` per batch.
* `loss.mode = "hilbert_soft"` (already in the loss registry).

No model-architecture changes. No sampler changes. No new training
infrastructure.

## 8. Calibration notes

* `α_peak`: smallest value with stable ≥99.5% argmax recovery on
  Dirichlet samples. For K=27, this is ≈ 10. Smaller α_peak = more
  smoothing but argmax may flip; larger = closer to deterministic.
* `α_base`: 0.1 is a reasonable default. Smaller = heavier-tailed
  off-target; larger = tighter Dirichlet.
* `loss.hilbert_alpha`: 1.0 in our experiments. Higher α_loss
  approaches strict variation seminorm; lower approaches MSE.
* Other EqM hyperparameters (gradient_lambda, gamma_power,
  ce_min_gamma) carry over unchanged.

## 9. Connection to the broader literature

Mathematically, Compositional EqM is **denoising score matching (DSM)
applied to a Gaussian-mollified Dirac mixture, parameterised via the EqM
energy form, with the Hilbert metric replacing L2 in the regression
loss**.

* DSM (Vincent 2011, Song & Ermon 2019): smoothed-density score gives a
  well-defined gradient field with maxima at data — what we call
  "thickening" is the same thing.
* Stochastic interpolants (Albergo & Vanden-Eijnden 2023): generalise
  FM to allow per-step path noise; our recipe is a particular case
  (noise on the endpoint x_1 only).
* Aitchison compositional analysis: provides the theoretical framework
  for why Hilbert is the natural metric on the simplex.

The contribution of this proposal is the *practical recipe* — combining
these ingredients in a coherent way that enables EqM training on
discrete data without resorting to learned embeddings or auxiliary
losses, and with calibrated downstream applications.

---

## Appendix A — Math summary

Let `α_t = α_base·𝟙 + α_peak·e_t`, `α_0 = Σ_j α_t[j]`, `M = I − (1/K)·𝟙𝟙ᵀ`.

```
Cov(log p | t) = diag(ψ_1(α_t[i])) − ψ_1(α_0)·𝟙𝟙ᵀ                    (Dirichlet)
Cov(CLR(p) | t) = M·diag(ψ_1(α_t[i]))·M                              (after CLR)
Tr(Cov(CLR(p) | t)) = (K−1)/K · Σ_i ψ_1(α_t[i])
floor(γ) = c(γ)² · [σ_source² · K + Tr(Cov(x_1))]                     (MSE floor)
floor(γ=1) = 0                                                         (since c(1)=0)
```

For α_peak=10, α_base=0.1, K=27: `Tr(Cov(CLR)) ≈ 2525`, `floor(γ=0) ≈
2525`, `floor(γ=0.99) ≈ 0.25`, `floor(γ=1) = 0`.

## Appendix B — Critical experimental verifications still needed

* Multi-seed run (n=3) of MSE+Dirichlet and Hilbert+Dirichlet at full
  training budget (compute cluster), with proper γ-matched recovery
  evaluation.
* Ablation on `α_peak`: confirm the floor curve stays below the
  argmax-flip-margin so recovery still works at α≤0.2.
* External baselines: comparison with score-based generative model
  (Song & Ermon EDM-style) on text8 to see if Compositional EqM achieves
  parity on denoising metrics.
