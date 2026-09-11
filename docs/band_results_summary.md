# The band construction for EqM — theory, campaign, and a result that did not reproduce

**Scope.** This document is the write-up source for the paper's band section. It
states (1) why the band exists, (2) how to set it from first principles, (3) what
was run at L=256, and (4) the central finding: **one configuration produced a
large, clean improvement in unconditional generation, and that improvement did not
survive a change of random seed.**

**Reported metrics.** Unconditional generation — n-gram KL (uni/bi/tri) and
`H_ratio`, per `scripts/eval_generation.py` at n=256 samples and 400 sampler steps
— plus the conditional-recovery ladder for the one run that transitioned (§4.1).
OOD detection was also measured on that checkpoint and is **deliberately out of
scope here**. Recovery was run only on the transitioning checkpoint; the four
non-transitioning cells were evaluated for generation alone, so no recovery
comparison across the campaign exists.

**Status of each claim.**

| claim | status |
|---|---|
| The band geometry — `Δ`, `t`, `A_K`, `γ*`, `γ_match`, `t = 1/α` | **EXACT**, and validated against measurement to 4 decimals |
| `γ_lo`'s two requirements and their incompatibility at K=27 | **EXACT** given the definitions; the *tolerances* are judgement calls |
| Band training fixes EqM's unconditional generation | **NOT ESTABLISHED** — observed once in five runs, did not reproduce at seed 43 |
| Band training improves conditional recovery | **NOT ESTABLISHED** — measured only on the transitioning checkpoint, so it inherits §4.2 |
| The band *rescale* alone improves recovery | **ESTABLISHED** — measured on the whole-path control, independent of the transitioning run |

---

## 1. What the band is, and the failure it targets

EqM regresses the conservative gradient `∇_x⟨x, f(x)⟩` onto `c(γ)·(x₀ − x₁)`
along the interpolant `x_γ = (1−γ)x₀ + γx₁`. With the standard `c(γ) = 1 − γ`
the target vanishes at `γ = 1`, so **the equilibrium of the learned energy is the
data vertex**, and sampling is gradient descent onto it.

That is the root of two documented failures (`NOTE_WHY_EBM_INIT_STUCK.md`,
`NOTE_WHY_UNCONDITIONAL_FAILS.md`): the trained energy has a minimum at *every*
vertex, correct or not, so descent from noise commits to a nearby vertex and the
resulting text carries no structure above the letter frequencies.

The band changes where the equilibrium sits:

```
c(γ) = max(0, 1 − γ/γ*) / λ            (models/eqm.py, decay_strategy="band")
γ ~ U[γ_lo, γ_hi],  with γ_hi = γ*
```

`c` now vanishes at `γ*`, so the fixed point of the descent is the **shell**
`‖x_{γ*}‖ ≈ 0.62` rather than the vertex at `‖x₁‖ = 12.27` — a factor of 20
inside it (12.2723 / 0.6151 = 19.9). The model is never asked to reach a vertex, and never trained at one.

Decoding still works because **argmax is scale-invariant**: the answer is read
off the shell at 1/20th the radius at no cost. `γ*` is chosen as precisely the
radius at which that argmax is right often enough. Note the ramp is never
supervised at its own root — the draw is half-open, so `c = 0` is a limit of the
fitted line, not a training example.

---

## 2. Setting the band from first principles

Full derivations in `docs/band_geometry_theory.md`; this is the summary a results
section needs.

### 2.1 Everything collapses to `(K, t)`

For a label-smoothed one-hot, the CLR gap between the correct coordinate and any
competitor is exactly

```
Δ = log((1 − ε + ε/K)·K/ε) = 12.5061          (K=27, ε=1e-4)
```

and the interpolant's argmax is correct iff `g_j − g_c < t` for every competitor,
where

```
t(γ) = γΔ / ((1−γ)σ)
```

`t` is **the decision boundary itself in noise units** — signal gap over
per-coordinate noise. γ, σ and ε reach the accuracy only through `t`, so `(K, t)`
is a sufficient statistic. That is why **σ and ε are not independent design
levers**: they cancel in every criterion below.

Supporting constants, all exact and all confirmed by measurement:

| quantity | formula | value | measured |
|---|---|---|---|
| `‖x₁‖` | `Δ√((K−1)/K)` | 12.2723 | **12.2723** |
| `‖x₀‖` | `σ√(K−1)` | 0.5099 | 0.5043 (1.1%, finite-sample) |
| `‖x_{γ*}‖` | `√(((1−γ)‖x₀‖)² + (γ‖x₁‖)²)` | 0.6151 | — |

### 2.2 `γ*` from decodability `[EXACT]`

Conditioning on the correct coordinate decouples the K−1 competitors, collapsing
a K-dimensional probability to one integral:

```
P_correct = A_K(t) = ∫ φ(x)·Φ(x + t)^(K−1) dx
```

`A_K(0) = 1/K` exactly (the integrand is `d/dx[Φᴷ/K]`), `A_K` is strictly
increasing, so `A_K⁻¹` is unique and the design formula splits cleanly:

```
t*    = A_K⁻¹(target accuracy)        1-D root find, depends on K ALONE
γ_dec = σ·t* / (Δ + σ·t*)             closed form; σ and ε enter only here
```

| target accuracy | `t*` | `γ_dec` |
|---|---|---|
| 50% | 1.973 | 0.01553 |
| 90% | 3.422 | 0.02664 |
| **95%** | 3.840 | **0.02979** |
| 99% | 4.634 | 0.03573 |

**The hand-tuned `γ* = 0.030` is the 95%-decodability point to within 0.7%.** The
empirical choice, restated exactly: *place the equilibrium where 95% of tokens
decode correctly.* This transfers to any `(K, ε, σ)` without re-tuning, and `t*`
grows only as `O(√log K)` — from 3.840 at K=27 to 5.945 at K=50257.

`A_K` is **per-position**. At 95% and L=256 the expected number of wrong
characters per window is 12.8 and `P(clean window) ≈ 2×10⁻⁶`. "95% decodable" is
a statement about tokens, not samples.

### 2.3 `γ_lo`: two analytic requirements that do not intersect at K=27

Unlike `γ*`, the lower edge is over-determined. Two criteria, both closed-form at
every K:

* **Requirement S (support).** Sampling starts at `x₀`, i.e. `γ = 0`. If
  `γ_lo > 0` that initial state lies **outside** the trained band by
  `κ = t_lo/√(K−1)` noise radii. S asks `t_lo ≤ κ√(K−1)` for a tolerable κ.
* **Requirement N (signal).** Below some margin the Bayes-optimal posterior is
  near-uniform, so the regression target is dominated by an irreducible
  (unlearnable) component. With `ρ(t)` the Bayes R², the `c²`-weighted
  irreducible share is `W = ∫c²(1−ρ) / ∫c²`. N asks `W ≤ W_max`.

At `γ* = 0.030`:

| γ_lo | `t_lo` | κ | `ρ(t_lo)` | `W` |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.000 | 89.2% |
| 0.005 | 0.628 | 0.123 | 0.017 | 81.6% |
| 0.01208 | 1.529 | 0.300 | 0.170 | 61.4% |
| 0.019 | 2.422 | 0.475 | 0.513 | 35.1% |

**They have no feasible intersection at K=27** (ratio 0.26), and no hyperparameter
reconciles them:

* **σ and ε cancel** — both requirements live in `t`.
* **γ\* only dilutes.** Raising it does drive `W` below 30%, but the decode
  accuracy there is 1.0000, so the reduction comes purely from adding
  trivially-solved states. Restricted to the informative band, `W` **floors at
  55.0%** even as `γ* → ∞`.
* **K is the only real lever.** Requirement N grows like `√(2 ln K)` while S's
  ceiling grows like `√(K−1)`; the feasible set opens at **K ≈ 1024**.

So at text8's vocabulary, compatibility is not a setting to find but a tolerance
to accept: the two meet only at κ ≈ 0.475, i.e. accepting that the sampler
initialises half a noise radius outside anything the field was trained on.
**Which requirement actually binds is an empirical question** — §3 tests it.

### 2.4 Validating the geometry against measurement `[EXACT + MEASURED]`

Unconditional generation never needs this section — it starts inside the band by
construction. The rescale below exists for tasks that begin from data-scale
inputs, and it is retained here for one reason: **it is the sharpest empirical
test of §2.1–2.2's geometry.** It converts the analytic model into a prediction
about an already-measured quantity, and the prediction holds to four decimals.

A data-scale input sits 20–100× outside the band, so reading the field there is
extrapolation. The remedy is a positive homothety `x ↦ g·x` — bijective and
argmax-preserving — which moves the *data into the model's domain* instead of
extrapolating the model.

A scalar rescale cannot rotate, so the angle to `x₁` is fixed by the corruption
strength α alone and the landing γ is not a free choice. Solving the angle match:

```
γ_match(α) = σ / (σ + α·Δ)        and        g(α) = γ_match(α)
```

The rescale factor **is** the landing γ, `√(K−1)` cancels, and the whole
hand-tabulated conversion collapses to one closed form. Simulation puts this
within 0.2% of the true angle-matching γ.

Substituting into `t` gives the identity that ties §2.2 and §2.4 together:

```
t(γ_match(α)) = 1/α          [EXACT]
```

A corruption ladder in α is therefore a sweep over `t`, read backwards — and that
is falsifiable. If a perturbed input lands at margin `1/α`, its own argmax
accuracy must be `A_K` evaluated there. Correcting for the fact that the
perturbation is not `V_d`-projected (a factor `√(K/(K−1)) = 1.01905`), the
prediction matches the measured per-token accuracy to **±0.0005** across
α ∈ {0.3, 0.5, 0.6, 0.7, 0.8}:

| α | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|
| predicted `A_K(1.019/α)` | 0.8961 | 0.5231 | 0.4027 | 0.3211 | 0.2648 |
| measured | 0.8957 | 0.5229 | 0.4023 | 0.3214 | 0.2653 |

Together with `‖x₁‖` = 12.2723 predicted and 12.2723 measured, this is the
evidence that the analytic band model is the right one. It is independent of any
training outcome.

---

## 3. The campaign at L=256

Five runs, all EqM, d1024/10L/16H, B=8, 10 epochs × 10,000 windows, lr 3e-4,
seed 42 unless noted. Generation evaluated at n=256 samples, 400 NAG-GD steps.

| cell | γ_lo | γ\* | clip | seed | KL_uni | KL_bi | KL_tri | H_ratio | transition |
|---|---|---|---|---|---|---|---|---|---|
| **published** | 0.005 | 0.030 | 1.0 | 42 | 0.0097 | **0.265** | 1.744 | 0.975 | **epoch 7** |
| seed replicate | 0.005 | 0.030 | 1.0 | **43** | — | *pending* | — | — | **none** (probe ≥1.44 through ep8) |
| ladder κ=0.30 | 0.01208 | 0.030 | 1.0 | 42 | 0.0084 | 1.394 | 4.994 | 0.991 | none |
| γ\* package | 0.005 | 0.03573 | 1.191 | 42 | 0.0172 | 1.453 | 4.843 | 0.963 | none |
| both changes | 0.0 | 0.03573 | 1.191 | 42 | 0.0218 | 1.512 | 4.837 | 0.952 | none |

Reference at the same budget and length: **Dirichlet FM 0.009 / 0.959 / 4.088**;
**EqM whole-path U[0,1] 0.018 / 1.582 / 5.042**.

Per-epoch probe (n=16 samples), bigram KL:

```
published    g_lo .005  g* .030   2.224 2.019 1.557 1.463 1.647 1.507 0.522 0.407 0.378 0.363
seed 43      g_lo .005  g* .030   1.972 2.325 1.456 1.457 1.742 1.470 1.444 1.525  …
ladder .0121 g_lo .0121 g* .030   2.155 1.778 1.452 1.322 1.430 1.446 1.390 1.446 1.404 1.468
gamma* pkg   g_lo .005  g* .0357  2.008 1.666 1.732 1.464 1.580 1.568 1.553 1.499 1.627 1.512
both         g_lo 0     g* .0357  2.028 2.684 1.895 1.641 1.677 1.586 1.657 1.569 1.626 1.500
```

**All five runs are indistinguishable through epoch 6** — every trajectory sits in
1.3–1.7. One of them then falls off a cliff at epoch 7 and stays down. The rest
never move.

---

## 4. The finding

### 4.1 The configuration that worked

`γ_lo = 0.005, γ* = 0.030, seed 42` produced, at the identical budget, the only
competitive unconditional-generation result the equilibrium route has ever given:

| | KL_uni | KL_bi | KL_tri | H_ratio |
|---|---|---|---|---|
| **band, seed 42** | **0.0097** | **0.265** | **1.744** | 0.975 |
| Dirichlet FM (same budget) | 0.009 | 0.959 | 4.088 | — |
| EqM whole path `U[0,1]` | 0.018 | 1.582 | 5.042 | — |

**3.6× better bigram KL than the budget-matched Dirichlet FM baseline, and 6.0×
better than whole-path EqM** — on the axis where the equilibrium route had
previously produced nothing above the letter frequencies. Trigram KL improves by
2.3× as well (1.744 vs 4.088), so the gain is genuine higher-order structure and
not a unigram-matching artifact; `H_ratio` 0.975 confirms the samples are not collapsed.

### 4.1.1 Recovery on the same checkpoint

The same run was also scored on conditional recovery — repairing a perturbed
window — with the input rescaled into the band per §2.4. Headline metric is
`Δ@α = token_acc − token_acc_perturbed`, the accuracy the field actually restores.

| α | `t = 1/α` | tok_acc | tok_acc perturbed | **Δ@α** | control Δ@α |
|---|---|---|---|---|---|
| 0.3 | 3.333 | 0.9357 | 0.8957 | **+0.0399** | −0.0216 |
| 0.5 | 2.000 | 0.6452 | 0.5229 | **+0.1223** | +0.0433 |
| 0.6 | 1.667 | 0.5248 | 0.4023 | **+0.1225** | +0.0619 |
| 0.7 | 1.429 | 0.4387 | 0.3214 | **+0.1173** | +0.0716 |
| 0.8 | 1.250 | 0.3742 | 0.2653 | **+0.1089** | +0.0751 |
| 1.0 | 1.000 | 0.2917 | 0.1943 | **+0.0975** | +0.0743 |

The control is the published whole-path `EqM_OneHot` arm read through the *same*
rescale. Δ peaks near α = 0.5–0.6, i.e. at `t ≈ 1.7–2.0`, where §2.2 puts
per-token decodability at 39–51% — the band field does most of its repair work
exactly where the input is maximally ambiguous, and less at both ends, where the
answer is either already visible (α = 0.3, 90% decodable) or largely gone.

**The gain decomposes into two independent halves**, and they have different
evidential status:

```
whole-path EqM, native readout        Δ@0.5 = −0.094      (paper, tab:gen)
whole-path EqM, band-rescaled input   Δ@0.5 = +0.043      +0.137 from the RESCALE
band-trained, band-rescaled input     Δ@0.5 = +0.122      +0.079 from band TRAINING
```

The first step is a property of the **rescale alone** — an argmax-preserving
homothety applied at inference to an already-trained model, no retraining — and it
is measured on the control, so it does not depend on the run that transitioned.
The second step is attributable to band *training* and is measured only on that
run; it therefore inherits §4.2 in full.

*Caveat.* These were taken at the driver's tabulated rescale factors, which match
only the radius and not the angle; they are off the exact `g(α) = σ/(σ+αΔ)` by
−8.7% at α = 0.3 and +1.6 to +3.8% elsewhere (`docs/band_geometry_theory.md`
§5.3). The α = 0.3 row is the one materially affected.

### 4.2 It does not reproduce

Four independent attempts to reproduce or extend the effect all fail, including
the one that changes nothing but the seed:

* `γ*` 0.030 → 0.03573 (the exact 99% point) at fixed γ_lo → **1.453**
* `γ_lo` 0.005 → 0.01208 at fixed γ\* → **1.394**
* both changes together → **1.512**
* **`seed` 42 → 43, nothing else** → probe 1.444 at epoch 7, 1.525 at epoch 8,
  where the original was already at 0.522 and 0.407

The seed replicate was verified programmatically to differ from the published cell
in `training.seed` alone. Its final n=256 eval was still running at the time of
writing; the probe has passed the entire window in which the original transitioned.

### 4.3 What this does and does not mean

**It means the 0.265 cannot be reported as a property of the configuration.** One
run in five transitioned, and the same configuration at another seed did not.
Everything else measured on that checkpoint inherits the same status, which is
why this report confines itself to generation.

**It does not settle which kind of negative this is.** Two readings remain open:

* *One-off.* The transition is an artifact of that trajectory and there is nothing
  to explain.
* *Real but unreliable bistability.* The field genuinely can escape the unigram
  basin, and roughly one run in five finds the escape. This is consistent with the
  data: four flat trajectories and one that jumped, with the jump arriving after
  the point where all five look identical.

The second is still a finding — an escape mechanism that exists but is not
dependable — and the two are distinguished only by more seeds. Nothing about the
run's *behaviour* looks pathological: H_ratio 0.975 and KL_uni 0.0097 are healthy,
and the failing runs are not collapsed either. The ladder rung at γ_lo = 0.01208
in fact has the **best** unigram KL (0.0084) and entropy match (0.991) of any
cell — a model that learned the letter frequencies beautifully and nothing above
them. That is the unigram ceiling the band was designed to escape.

---

## 5. What survives regardless

The geometry is independent of whether any training run transitions:

1. **`γ*` has a closed form**, and the hand-tuned value was the 95% decodability
   point to 0.7%. The band's upper edge is not a hyperparameter to search.
2. **`γ_lo` is over-determined at small K.** Support and signal have no feasible
   intersection below K ≈ 1024, and no hyperparameter reconciles them — γ\*
   appears to help only by dilution. This is a structural statement about applying
   band-EqM to small vocabularies.
3. **The rescale is exact and parameter-free**: `g(α) = γ_match(α) = σ/(σ+αΔ)`,
   with `t = 1/α`, validated against measurement to ±0.0005.

---

## 6. Further research

The campaign leaves four concrete, cheap questions.

1. **Seeds, and the bistability question.** Run the published configuration at
   seeds 44–48. If the escape rate is ~1/5, band-EqM has a genuine but unreliable
   escape from the unigram basin, and the object of study becomes *what
   distinguishes the trajectories that escape*. Per-epoch `r<.33` (the ratio of
   learned to target gradient magnitude) is already logged and is the obvious first
   place to look for a precursor.
2. **Separate "the field is worse" from "the sampler cannot reach it."**
   Unconditional generation starts at `x₀`, i.e. at `γ = 0`, so for `γ_lo > 0` it
   is scored from a state the field never saw — it tests Requirement S and the
   objective at once, and cannot separate them. Scoring the same checkpoints from
   an initial state placed *inside* the band would bypass S entirely; a
   configuration that improves under the second reading while degrading under the
   first would localise the failure in the **sampler** rather than the training
   objective. The generation-only campaign reported here cannot distinguish these.
3. **A band-native sampler.** Requirement S is a constraint only because
   unconditional sampling must start at `γ = 0`. Initialising instead at
   `(1−γ_lo)σε + γ_lo·clr(unigram draw)` is data-free but lands *inside* the band,
   which would remove the S/N conflict rather than trading it off — and it is a
   change to inference alone, requiring no retraining.
4. **The K ≈ 1024 crossover, tested.** The prediction that the S/N conflict
   dissolves at larger vocabularies is analytic and untested. A BPE-tokenised run
   at K ≈ 1024–4096 would test the band's central design constraint where the
   theory says it should finally be satisfiable, and would tell us whether the
   text8 character alphabet was simply the wrong place to try this.

---

*Sources: `sweeps/band_L256.yaml` (all five cells), `runs/band_L256_ep10_d10k*/`
(results), `scripts/band_geometry.py` (every constant here is computed by it, and
the `t = 1/α` identity is asserted in it), `docs/band_geometry_theory.md` (full
derivations), `docs/band_eqm.md` (mechanism walkthrough).*
