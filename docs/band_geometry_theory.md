# Band geometry — the exact theory

*Written 2026-08-28. Companion to [`band_eqm.md`](band_eqm.md), which describes
what the band **is** and why it was introduced. This document derives the
**closed forms**: where the band should sit, how wide it should be, how it moves
with vocabulary size `K`, CLR smoothing `ε` and source noise `σ`, and why
renormalisation is required for every read of the field that does not start from
noise.*

**Status labels used throughout.** `[EXACT]` — derived in closed form or by
quadrature to machine precision. `[MEASURED]` — read off a run in this
repository. `[APPROX]` — a stated approximation, with its error quantified.
`[CONJECTURE]` — consistent with the data but not established.

---

## 1. Setup and notation

EqM trains on a straight interpolant between a noise sample and a data point:

```
x_γ = (1 − γ)·x₀ + γ·x₁          γ ∈ [0,1];  γ=0 pure noise, γ=1 clean data
```

with the flow target `u_tgt = c(γ)·(x₀ − x₁)` (`models/eqm.py:208`).

| symbol | meaning | value for text8 |
|---|---|---|
| `K` | vocabulary size | 27 |
| `ε` | CLR label smoothing (`TransformationConfig.label_smoothing`) | 1e-4 |
| `σ` | source noise scale (`cfg.eqm.source_sigma`) | 0.1 |
| `γ*` | equilibrium — the zero of `c(γ)` (`cfg.eqm.gamma_star`) | 0.03 |
| `[γ_lo, γ_hi]` | trained band (`cfg.eqm.gamma_lo/hi`) | **[0, γ*] = [0, 0.03]** (published runs in §8: [0.005, 0.03]) |
| `α` | corruption strength in the recovery protocol | 0.3 … 1.0 |
| `φ`, `Φ` | standard normal **density** and **CDF**: `φ(x)=e^(−x²/2)/√(2π)`, `Φ(x)=∫_{−∞}^x φ` | — |
| `Δ` | CLR log-ratio gap between the correct coordinate and any competitor (§2.1) | 12.5061 |
| `t` | standardised margin `γΔ/((1−γ)σ)` — the decision boundary in noise units (§4.1) | — |

`x₀ = σ·randn` centred on `V_d` (zero-mean across K), so it lives in the
`(K−1)`-dimensional zero-sum subspace where the CLR features live.

### 1.1 The equilibrium is the zero of `c(γ)` `[EXACT]`

With `decay_strategy: band` (`models/eqm.py:373`):

```python
c_gamma = torch.clamp(1.0 - gamma / gstar, min=0.0)
```

so `c(γ) = max(0, 1 − γ/γ*)`, vanishing exactly at `γ*`. The fixed point of the
descent is therefore the state `x_{γ*}`, not a simplex vertex. Two consequences
that matter later:

- `c(0) = 1` is the **maximum** target weight, and it sits where the input
  carries **no** information about `x₁`. With `γ_lo = 0` that sliver is trained,
  deliberately, at a measured cost of 42% of the target mass (§6.2).
- `γ_hi = γ*` **by design** (2026-08-28): the band top *is* the zero of `c`, and
  everything above it has `c = 0`. `γ_hi` and `γ*` are independent config fields
  (`_c_gamma` reads `gamma_star`; the γ draw reads `gamma_lo/hi`) and could be
  decoupled, but tying them keeps the construction to **one** parameter — see
  §6.1 for what that costs and §6.3 for what it forecloses.

---

## 2. Exact scalar geometry

### 2.1 The CLR log-ratio gap `Δ` `[EXACT]`

`x₁` is the CLR of a label-smoothed one-hot: `p_hi = 1−ε+ε/K`, `p_lo = ε/K`.
Writing `Δ = log(p_hi/p_lo)`, the CLR components are

```
hi coordinate :  (K−1)·Δ/K           = 12.043    (K=27, ε=1e-4)
lo coordinates:  −Δ/K   (×(K−1))     = −0.4632
```

**`Δ = log((1 − ε + ε/K)·K/ε)` is exact.** The frequently written `Δ ≈ log(K/ε)`
is prose shorthand and is *not* used in any computation here; for text8 the two
agree to 4 significant figures (12.5060 vs 12.5062).

**Signs are structural.** `Δ ≥ 0` for every `ε ∈ (0,1]` — `p_hi = p_lo` only at
`ε = 1` — so the hi coordinate is `≥ 0` and the `K−1` lo coordinates are `≤ 0`,
necessarily: CLR is zero-sum, and `hi + (K−1)·lo = 0` identically (verified to
machine precision for ε from 1e-6 to 1). Both vanish only in the uniform limit
`ε → 1`, where `‖x₁‖ = 0` and the data carries no information.

This matters for §4: the correct-vs-competitor gap is `hi − lo = (K−1)Δ/K + Δ/K
= Δ` **exactly**, so the negative lo coordinates supply the final `Δ/K` of the
separation. That is why the decodability gap is a clean `γ·Δ` and not
`γ·(K−1)Δ/K`.

### 2.2 Norms `[EXACT, verified against measurement]`

```
‖x₁‖ = Δ·√((K−1)/K)                        = 12.2723
‖x₀‖ = σ·√(K−1)                            =  0.5099
‖x_γ‖ = √( ((1−γ)‖x₀‖)² + (γ‖x₁‖)² )        (x₀ ⊥ x₁ in expectation)
```

The predicted `‖x₁‖ = 12.2723` reproduces the measured `embed_norm_mean =
12.2723` **to four decimals** `[MEASURED]` (printed by `recovery_check.py`).
This is the single strongest validation that the analytic model is the right
one — ε, K and the CLR construction alone determine the data norm.

`‖x₀‖`: analytic 0.5099 vs 0.5043 measured in `band_eqm.md` (1.1%; finite-sample).

**Orthogonality caveat.** `x₀ ⊥ x₁` holds *in expectation*, not per sample: the
cross term has standard deviation `~σ‖x₁‖` per position. It averages out over
L=256 × n=256 but is not exact for any single window.

### 2.3 The perturbed input `[EXACT]`

The recovery protocol perturbs a clean embedding with per-coordinate noise of
standard deviation `α·‖x₁‖`, so

```
‖z_α‖ = ‖x₁‖·√(1 + K·α²)          [RMS: this is √(E‖z_α‖²), not E‖z_α‖]
```

**Which expectation this is.** Taking `‖z_α‖² = ‖x₁‖² + 2α‖x₁‖⟨x₁,ξ⟩ +
α²‖x₁‖²‖ξ‖²` and using `E⟨x₁,ξ⟩ = 0`, `E‖ξ‖² = K` gives
`E‖z_α‖² = ‖x₁‖²(1+Kα²)` — a **mean square**. The mean norm is about 0.9% lower
by Jensen (simulated `E‖z_α‖ = 33.859` vs RMS 34.165 at α=0.5), and the
*per-position* norm has a spread of ≈13.5%, so this is a distributional
statement, not a per-sample one. It concentrates over a window: at L=256 the
window-mean norm is within ~0.85% of the RMS. §5.2 uses the RMS consistently on
both sides of its angle match, which is why the 0.9% offset cancels there.

Confirmed `[MEASURED]`: the stored `sigma_perturb` is 6.136 at α=0.5 and 3.682
at α=0.3, i.e. exactly `α·12.2723`.

**The data/band scale gap is 22.8×** (12.272 vs 0.539). This is the entire
reason renormalisation exists.

---

## 3. Three γ-scales — and which one is operative

Three distinct thresholds live in this geometry. They are routinely conflated;
they are not the same number, and only the second is the right design target.

| scale | question it answers | text8 value |
|---|---|---|
| `γ_eq` | where do the **norms** of noise and data cross? | 0.0399 |
| `γ_dec` | where does **argmax decoding** succeed? | 0.0298 (95%) |
| `γ_match(α)` | where does a **rescaled α-perturbed input land**? | 0.0157 (α=0.5) |

### 3.1 Norm parity `γ_eq` `[EXACT]` — the wrong criterion

Setting `(1−γ)‖x₀‖ = γ‖x₁‖`:

```
γ_eq = σ√K / (σ√K + Δ)                     = 0.0399
```

**This is not the right target for `γ*`.** It compares norms, but the data
component is concentrated in *one* coordinate while the noise is spread
isotropically over K. Decoding therefore succeeds well before norm parity: at
`γ_eq` argmax accuracy is already 99.75%, so choosing `γ* = γ_eq` would place the
equilibrium deep in the saturated regime and spend band width on states that were
already trivially decodable.

### 3.2 Decodability `γ_dec` `[EXACT]` — the operative criterion

See §4. `γ*` should be set here.

### 3.3 Rescale landing point `γ_match(α)` `[EXACT]`

See §5.2.

---

## 4. Exact decodability

### 4.1 Derivation

Decoding is `argmax` over the K CLR coordinates. At interpolant `x_γ`:

- correct coordinate mean: `γ·(K−1)Δ/K`
- every other coordinate mean: `−γ·Δ/K`
- so the **gap is `γ·Δ`**, independent of K given Δ.

The noise is `(1−γ)x₀`, centred on the zero-sum subspace. With
`Var(xᵢ) = σ²(1−1/K)` and `Cov(xᵢ,xⱼ) = −σ²/K`:

```
Var(xᵢ − xⱼ) = 2σ²(1−1/K) + 2σ²/K = 2σ²      [EXACT — centring cancels]
```

so the zero-sum projection does **not** perturb pairwise differences. Writing
`x₀ᵢ = σ(gᵢ − ḡ)` with `g` iid standard normal, the correct-argmax event is
`gⱼ < g_c + t` for all `j ≠ c`, where

```
t(γ) = γΔ / ((1−γ)σ)
```

**What `t` is.** The standardised margin: the gap between the correct
coordinate's mean and any competitor's mean (`γΔ`, independent of K), expressed
in units of **one** coordinate's noise standard deviation `(1−γ)σ`. It is
dimensionless, and it is the only channel through which γ, σ and ε reach `A_K` —
which is why `A_K` is a function of `K` and `t` alone.

**Two conventions exist and must not be mixed.** `t` above is the
*single-coordinate* normalisation, which is what conditioning on `g_c` requires.
The *pairwise* normalisation `z = t/√2` uses the standard deviation of a
difference `xᵢ − xⱼ` (variance `2σ²`, §4.1). Every formula in this document uses
`t`; §4.4 records what happens when the two are conflated.

**Why conditioning decouples the competitors.** The event
`{gⱼ < g_c + t for all j ≠ c}` is written in terms of the **raw** `g`, which is
iid standard normal. That is the whole point of the previous step: the centring
`ḡ` was already cancelled when `Var(xᵢ − xⱼ) = 2σ²` removed it from every pairwise
comparison. The distinction matters. The centred vector `x₀` has correlated
coordinates (`Cov = −σ²/K`), and conditioning on one *of those* would leave the
rest dependent; conditioning on the iid `g_c` does not.

So condition on `g_c = x`. The K−1 events `{gⱼ < x + t}` are then independent,
each of probability `Φ(x + t)`, giving

```
P(correct | g_c = x) = Φ(x + t)^(K−1)
```

(`Φ` is the standard normal CDF, so `Φ(x+t)` is the chance one competitor
falls below the threshold, and the K−1 of them are independent given `g_c`).
Averaging over the density of `g_c` — which is `φ`, the standard normal
density — collapses the K-dimensional probability to a **one-dimensional
integral**:

```
P_correct(γ) = A_K(t(γ)),      A_K(t) = ∫ φ(x)·Φ(x + t)^(K−1) dx
```

The coordinates are exchangeable, so the answer does not depend on which class is
correct: `A_K(t)` is the per-position accuracy for every token. And `A_K` depends
only on `K` and `t` — it knows nothing about σ, ε or γ individually.

**Properties of `A_K` (why the inversion is well-posed).**

1. **`A_K(0) = 1/K`, exactly.** Since `d/dx[Φ(x)ᴷ/K] = Φ(x)^(K−1)φ(x)`, the
   integrand is an exact derivative and the integral telescopes:
   `∫φΦ^(K−1) = [Φᴷ/K]` from `−∞` to `∞` `= 1/K`. Chance level at zero margin,
   as it must be. This is a complete check on the formula — verified to 8
   decimals at K = 2, 27, 100, 1024 — and it needs no simulation.
2. **Strictly increasing:**
   `A_K′(t) = (K−1)∫φ(x)Φ(x+t)^(K−2)φ(x+t) dx > 0` for all `t`.
3. **Limits:** `A_K(t) → 1` as `t → ∞`, `→ 0` as `t → −∞`.

So `A_K` is a strictly increasing bijection onto `(0,1)`, and `A_K⁻¹(acc)` exists
and is unique for every target in `(1/K, 1)`. The root find cannot fail or land
on a wrong branch. At K=27 the slope at the design points is
`A_K′(3.840) = 0.0893` and `A_K′(4.634) = 0.0227` — small but far from zero, so
`t*` is well-conditioned: a ±0.001 tolerance on the target accuracy fixes `t*` to
about ±0.01, i.e. `γ*` to about ±0.3%.

**Inverting `t(γ)`.** The margin is a Mobius function of γ and inverts in three
lines:

```
t = γΔ / ((1−γ)σ)   ⟹   tσ(1−γ) = γΔ   ⟹   tσ = γ(Δ + tσ)   ⟹   γ = σt/(Δ + σt)
```

`γ(t) = σt/(Δ+σt)` is increasing with `γ(0)=0` and `γ(∞)=1`, so it is a
bijection `[0,∞) → [0,1)` and each target accuracy yields exactly one `γ_dec`.
Substituting gives the design formula:

```
t*    = A_K⁻¹(target accuracy)                  (1-D root find, depends on K ALONE)
γ_dec = σ·t* / (Δ(K,ε) + σ·t*)                   (closed form, restores σ and ε)
```

**The split is the useful part.** The nonlinear step sees only `K`, so it is
tabulated once per vocabulary (§4.3); changing σ or the label smoothing moves
only the second line, which is arithmetic. Nothing needs re-tuning or re-running.

**Numerical recipe.**

* *Quadrature:* integrate over `[−12, 12+t]`. `φ` is below 1e-32 outside, and
  shifting the upper limit by `t` keeps the region where `Φ(x+t)^(K−1)` has not
  yet saturated inside the interval.
* *Large K:* evaluate `Φ^(K−1)` as `exp((K−1)·log Φ)` (`scipy.stats.norm.logcdf`).
  Direct powering underflows to 0 in the left tail for `K ≳ 10⁴` and silently
  biases the integral downward.
* *Root find:* bracket `[1e-6, 40]` and use Brent (`scipy.optimize.brentq`,
  `xtol=1e-12`, as in `scripts/band_geometry.py`). `A_K(1e-6) ≈ 1/K < acc` and
  `A_K(40) ≈ 1 > acc` guarantee a sign change for any target in `(1/K, 1)`.

**`A_K` is per-position, not per-sequence.** This is the easiest number in the
document to misread. At the 95% design point and L=256, the expected number of
mis-decoded characters per window is `0.05 × 256 = 12.8`, and the probability that
a *whole* window decodes cleanly is `0.95²⁵⁶ ≈ 2×10⁻⁶`. Even at 99% it is
`0.99²⁵⁶ ≈ 7.6%`, with 2.6 expected errors. `γ*` is a statement about tokens —
which is the right unit, since the equilibrium is per-position — but "95%
decodable" does not mean "95% of samples are clean text".

**How `t*` grows with K.** The root find has to be redone per vocabulary, but the
answer barely moves:

| K | `t*(95%)` | `t*(99%)` | `√(2 ln(K−1))` | `√(K−1)` |
|---|---|---|---|---|
| 27 (text8) | 3.840 | 4.634 | 2.553 | 5.099 |
| 128 | 4.386 | 5.149 | 3.113 | 11.269 |
| 1024 | 4.999 | 5.738 | 3.723 | 31.984 |
| 4096 | 5.360 | 6.089 | 4.079 | 63.992 |
| 50257 (GPT-2) | 5.945 | 6.663 | 4.653 | 224.179 |

`t*` is set by the scale of the maximum of K−1 standard normals, so it grows like
`O(√(log K))`: over a 1861× increase in vocabulary it rises only 3.840 → 5.945, a
factor of 1.55. (The `√(2 ln(K−1))` column is the asymptotic growth *rate*, not a
usable approximation — it converges far too slowly to substitute for the root
find at these K.) Over the same range `√(K−1)` grows 44×.

Those two rates are what §6.5's K-crossover turns on. Note carefully that they
attach to **different edges** of the band: `t*` sets `γ_hi`, while `√(K−1)` bounds
`γ_lo` through Requirement S. The quantity that races the S ceiling in §6.5 is
`t_N(W)`, the signal requirement on the *lower* edge — not `t*`.

### 4.2 Verification `[EXACT vs MEASURED]`

Quadrature against a 200,000-sample Monte-Carlo simulation, K=27, ε=1e-4, σ=0.1:

| γ | exact `A_K(t)` | simulated |
|---|---|---|
| 0.005 | 0.1125 | 0.1150 |
| 0.010 | 0.2617 | 0.2617 |
| 0.016 | 0.5215 | 0.5229 |
| 0.020 | 0.6970 | 0.6984 |
| 0.028 | 0.9249 | 0.9257 |
| 0.030 | 0.9524 | 0.9526 |
| 0.0399 | 0.9975 | 0.9975 |

Agreement to four decimals. **No simulation is needed.**

### 4.3 The design table `[EXACT]`

| target accuracy | `t*` | `γ_dec` |
|---|---|---|
| 50% | 1.973 | 0.01553 |
| 90% | 3.422 | 0.02664 |
| **95%** | 3.840 | **0.02979** |
| 99% | 4.634 | 0.03573 |

**The hand-tuned `γ* = 0.030` is the 95%-decodability point to within 0.7%.**
The empirical choice, restated exactly: *place the equilibrium where 95% of
tokens decode correctly.* This transfers to any (K, ε, σ) without re-tuning.

### 4.4 Why the earlier heuristic drifted `[APPROX]`

The extreme-value estimate `√(2 ln(K−1)) = 2.553` — the *mean* of the max of K−1
standard normals — is ambiguous in this application, and the ambiguity is larger
than its intrinsic error:

| use of the heuristic | `t` | γ | true accuracy |
|---|---|---|---|
| taken directly as `t` | 2.553 | 0.0200 | **69.7%** |
| with an ad-hoc `√2` | 3.611 | 0.0281 | **92.6%** |
| **exact, 95% target** | **3.840** | **0.02979** | 95% |

Two separate defects. First, `√(2 ln(K−1))` is already in single-coordinate
units, so multiplying by `√2` is not justified — the conditioning on `g_c`
already accounts for the correlation. It happens to compensate, moving 69.7% to
92.6%, which is why the `0.0281` figure looked reasonable. Second, even used
correctly it is a **mean, not a quantile**: it ignores the spread of both the
maximum and of `g_c`, so it cannot target a stated accuracy at all.

Against the exact `t*` the residual error is **5.8% in `t`** (3.611 vs 3.840) and
5.7% in γ — *not* the ~50% an earlier draft of this section claimed by comparing
2.553 against 3.840 across the two conventions. The exact integral removes the
ambiguity entirely, which is the reason to prefer it.

The drift is entirely this approximation — **not** the `log(K/ε)` shorthand,
which was never used numerically.

### 4.5 Numerical note

Superseded by §4.1's **Numerical recipe**, which gives the quadrature limits, the
`logcdf` route for large K, and the root-find bracket, together with the
well-posedness argument (`A_K` is a strictly increasing bijection, so the root is
unique and cannot be missed).

### 4.6 The band is the decodability transition `[EXACT + MEASURED]`

| γ | decode accuracy | role |
|---|---|---|
| 0 = `γ_lo` | 3.7% (chance) | no information; `c(γ)` weight is **maximal** — the sampler's own init (§6.2) |
| 0.005 | 11% | ~3× chance; the floor the §8 runs used |
| 0.0155 | 50% | |
| 0.020 | 70% | char-level OOD localisation peaks here `[MEASURED]` |
| 0.030 = `γ*` | 95% | sequence-level OOD peaks here `[MEASURED]` |
| 0.040 | 99.8% | saturated — nothing left to learn |

**The band `[0, γ*]` spans chance → 95% decodability**, i.e. the entire
informative part of the path plus the source itself. Above `γ*` the task is
already solved, so there is nothing left to learn there — which is exactly where
`c(γ)` vanishes.

**Context caveat.** This curve is *context-free* single-position decodability.
The model is a transformer over L=256 positions and exploits n-gram structure,
so its effective threshold is **lower** than these numbers. The curve is a lower
bound on network capability, not a prediction of it.

---

## 5. Renormalisation

### 5.1 Why it is required, and what it is not

Generation starts at `x_init = σ·randn`, norm 0.520 — *inside* the band by
construction. **Unconditional generation is the band's native operation and needs
no rescale.** Recovery and OOD start from data-scale inputs (norm 12.27, or
`‖z_α‖` when perturbed): 23–100× outside anything the field has seen. Reading the
energy there is extrapolation.

The rescale is a positive homothety `x ↦ g·x`: bijective, argmax-preserving, and
it moves the *data into the model's domain* rather than extrapolating the model.
Machine precision is **not** a constraint — band 0.54 vs data 12.27 is 23×, both
order 1e0, against float32's ~7 digits and 1e-38 subnormals.

But the cardinality of the interval does no work: a learned field is pinned only
where training put mass, to within training error. Analytic continuation would
require exactness, which a network does not provide.

### 5.2 One degree of freedom, two coordinates `[EXACT]`

A scalar rescale **cannot rotate**. Alignment with the data direction is fixed by
α alone:

```
cos(g·z_α, x₁) = ‖x₁‖/‖z_α‖ = 1/√(1 + Kα²)          — invariant under scaling
cos(x_γ,   x₁) = γ‖x₁‖/‖x_γ‖                        — set by γ
```

**Derivation.** Both states split into a component **along** `x₁` (signal) and one
**orthogonal** to it (noise); the whole argument is that pair of numbers.

*(a) The perturbed input.* By §2.3, `z_α = x₁ + α‖x₁‖·ξ` with `ξ` per-coordinate
standard normal. Then

```
⟨z_α, x₁⟩ = ‖x₁‖² + α‖x₁‖·⟨ξ, x₁⟩ = ‖x₁‖²        (E⟨ξ,x₁⟩ = 0)
‖z_α‖     = ‖x₁‖·√(1 + Kα²)                       (§2.3)
⟹ cos(z_α, x₁) = ‖x₁‖² / (‖z_α‖·‖x₁‖) = ‖x₁‖/‖z_α‖ = 1/√(1 + Kα²)
```

The implied tangent is `tan θ = √(1/cos²θ − 1) = α√K`. **That `α√K` is a
tangent, not a component norm** — do not read it as an RMS. The actual RMS
components of `z_α` are

```
∥ (RMS) = ‖x₁‖·√(1 + α²)        since  ⟨z_α,x̂₁⟩ = ‖x₁‖(1 + αu),  u ~ N(0,1)
⊥ (RMS) = α‖x₁‖·√(K−1)          the perturbation less its x̂₁ component
```

(which do reconcile: `‖x₁‖²(1+α²) + α²‖x₁‖²(K−1) = ‖x₁‖²(1+Kα²)` ✓, and
simulation gives `⊥ = 18.778` against `α‖x₁‖√(K−1) = 18.773` at α=0.3, versus
`α‖x₁‖√K = 19.131` — 1.9% out). `tan θ = α√K` instead pairs the **mean** parallel
component `‖x₁‖` with the **RMS** total, which is the convention the cosine above
uses.

**Convention, stated once.** Every line in §5.2 is a *ratio of expectations* —
`E⟨·,x₁⟩` over `√(E‖·‖²)` — not `E[cos]`. The two differ, but only slightly: at
α = 0.5 the formula gives 0.35921 against a simulated `E[cos] = 0.35768`, −0.4%.
What matters is that **the same convention is used on both sides of the match**,
so the bias is common and largely cancels in the solution. Solving numerically for
the γ whose simulated mean angle equals `z_α`'s:

| α | empirical `γ` matching the angle | `σ/(σ+αΔ)` | error |
|---|---|---|---|
| 0.3 | 0.02595 | 0.02596 | +0.03% |
| 0.5 | 0.01577 | 0.01574 | −0.16% |
| 0.8 | 0.00991 | 0.00990 | −0.18% |

So `γ_match` is good to **0.2%**, better than either cosine is individually —
which is why the `[EXACT]` label survives even though the intermediate cosines are
ratio-of-expectations. (A `√(K−1)`-based variant of the derivation gives 0.02644 /
0.01604 / 0.01008, a consistent +1.7 to +1.9% — it is the wrong pairing.)

*(b) Why the rescale drops out.* For any `g > 0`,

```
cos(g·z_α, x₁) = g⟨z_α,x₁⟩ / (g‖z_α‖·‖x₁‖) = cos(z_α, x₁)
```

`g` cancels between numerator and denominator. That is the exact sense in which a
positive homothety "cannot rotate": it slides a state **along its own ray**.

*(c) The band state.* `x_γ = (1−γ)x₀ + γx₁` with `⟨x₀,x₁⟩ = 0` in expectation
(§2.2), so

```
⟨x_γ, x₁⟩ = γ‖x₁‖²
⟹ cos(x_γ, x₁) = γ‖x₁‖² / (‖x_γ‖·‖x₁‖) = γ‖x₁‖/‖x_γ‖
```

with components `∥ = γ‖x₁‖` and `⊥ = (1−γ)‖x₀‖`.

**Why exactly one γ per α.** Laying the two side by side:

| state | ∥ `x₁` | ⊥ `x₁` | radius | angle to `x₁` |
|---|---|---|---|---|
| `g·z_α` | `g‖x₁‖` | `g·α√K‖x₁‖` | set by `g` | **fixed by α** |
| `x_γ` | `γ‖x₁‖` | `(1−γ)‖x₀‖` | set by γ | set by γ |

The rescale multiplies **both** components by the same `g`, so it moves the radius
and leaves the angle untouched. The band's angle, by contrast, is a function of γ.
So the angle equation must be solved for γ **first and alone** — there is no
freedom left in it — and `g` is then whatever makes the radii agree. A band state
has two coordinates that both depend on γ; scaling controls one.

**Matching the angle is matching the noise-to-signal ratio.** Two vectors share an
angle with `x₁` iff their `⊥/∥` ratios agree, which gives the matching equation in
one line and with no squaring:

```
α√K = (1−γ)‖x₀‖ / (γ‖x₁‖)
```

Equivalently, equating the cosines and clearing denominators:

```
γ²‖x₁‖²(1+Kα²) = ((1−γ)‖x₀‖)² + γ²‖x₁‖²      subtract γ²‖x₁‖² from both sides
⟹ γ²‖x₁‖²Kα²  = ((1−γ)‖x₀‖)²                 both sides ≥ 0, take roots
⟹ γ‖x₁‖√K·α   = (1−γ)‖x₀‖                    collect γ
⟹ γ(‖x₀‖ + α‖x₁‖√K) = ‖x₀‖
⟹ γ_match(α)  = ‖x₀‖ / (‖x₀‖ + α‖x₁‖√K)
```

Substituting `‖x₁‖√K = Δ√((K−1)/K)·√K = Δ√(K−1)` and `‖x₀‖ = σ√(K−1)`, every term
carries `√(K−1)` and it **cancels exactly**:

```
γ_match(α) = σ√(K−1) / (σ√(K−1) + αΔ√(K−1)) = σ / (σ + α·Δ)      [EXACT]
```

**The landing point is K-independent.** The vocabulary size enters the derivation
twice — through `√K` in the perturbation's orthogonal component and through
`√((K−1)/K)` in `‖x₁‖` — and leaves entirely. `γ_match` depends only on the ratio
`σ/Δ`, i.e. on the noise scale against the label-smoothing log-odds.

### 5.2.1 The landing margin is exactly `1/α` `[EXACT + MEASURED]`

Feeding `γ_match` back into the standardised margin of §4.1 collapses it
completely. With `1 − γ_match = αΔ/(σ+αΔ)`:

```
t(γ_match(α)) = γΔ / ((1−γ)σ) = [σΔ/(σ+αΔ)] / [αΔσ/(σ+αΔ)] = 1/α      [EXACT]
```

**Equivalently — and this is the cleaner route — the matching equation *is* the
identity.** The last line of §5.2 before collecting γ reads `γΔα = (1−γ)σ`.
Divide both sides by `α(1−γ)σ`:

```
γΔ / ((1−γ)σ) = 1/α        i.e.   t = 1/α
```

No substitution needed: `t = 1/α` and the angle-match condition are the same
equation written two ways. That also makes the identity independent of §5.2's
ratio-of-expectations convention — whatever `γ_match` is, its `t` is `1/α`.

**`t·α = 1`.** The corruption strength and the margin it lands on are exact
reciprocals — `σ`, `Δ` and `K` all cancel. Verified to six decimals for
α ∈ {0.3, 0.5, 0.6, 0.7, 0.8, 1.0}. This closes §4 and §5 into a single statement:
**the recovery ladder is a sweep over `t`, read backwards.**

It is also falsifiable, because it predicts something already measured. If the
perturbed input lands at margin `1/α`, its *own* decodability must be `A_K(1/α)`,
which is exactly what `recovery_check.py` stores as `token_acc_perturbed_band`:

| α | `t = 1/α` | `A_K(1/α)` | `A_K(1.019/α)` | measured | error |
|---|---|---|---|---|---|
| 0.3 | 3.333 | 0.8857 | **0.8961** | 0.8957 | +0.0003 |
| 0.5 | 2.000 | 0.5095 | **0.5231** | 0.5229 | +0.0002 |
| 0.6 | 1.667 | 0.3917 | **0.4027** | 0.4023 | +0.0004 |
| 0.7 | 1.429 | 0.3124 | **0.3211** | 0.3214 | −0.0003 |
| 0.8 | 1.250 | 0.2578 | **0.2648** | 0.2653 | −0.0005 |

The raw `A_K(1/α)` is systematically ~1% low, and the discrepancy is **not slop**
— it is the `V_d` projection, and it is exact. The band's noise `x₀` is projected
onto the zero-sum subspace; `z_α`'s perturbation (§2.3) is **not**, being added to
all K coordinates independently. So `z_α`'s own margin is

```
Δ / (α‖x₁‖) = Δ / (α·Δ√((K−1)/K)) = (1/α)·√(K/(K−1)) = 1.01905/α      (K=27)
```

slightly larger than the band state it is matched to — the two agree in *angle*,
which is what the rescale can control, but differ by `√(K/(K−1))` in noise
normalisation. Using that corrected margin, prediction meets measurement to
**±0.0005 across all five α**.

### 5.3 The rescale factor in closed form `[EXACT]`

**Identity: the rescale factor *is* the landing γ.**

```
g(α) = γ_match(α) = σ / (σ + α·Δ)
```

*Proof.* Angle-matching (§5.2) gives `(1−γₘ)‖x₀‖ = γₘ‖x₁‖√K·α`, so

```
‖x_γₘ‖² = (γₘ‖x₁‖√K·α)² + (γₘ‖x₁‖)² = γₘ²‖x₁‖²(1 + Kα²)
⟹ ‖x_γₘ‖ = γₘ·‖z_α‖        (since ‖z_α‖ = ‖x₁‖√(1+Kα²))
⟹ g = ‖x_γₘ‖/‖z_α‖ = γₘ
```

So the entire hand-tabulated `g_for_alpha` table collapses to one closed form —
no norms, no `√K`, no reference radius. Computed by
`scripts/band_geometry.py`.

**What the driver's table gets wrong.** `drive_band_L256.sh:g_for_alpha` matches
only the *radius*, against a fixed `‖x_γ‖ = 0.539`:

| α | exact `g = σ/(σ+αΔ)` | driver `g` | driver error |
|---|---|---|---|
| **0.3** | 0.02596 | 0.02370 | **−8.7%** |
| 0.5 | 0.01574 | 0.01600 | +1.6% |
| 0.6 | 0.01315 | 0.01342 | +2.0% |
| 0.7 | 0.01129 | 0.01165 | +3.2% |
| 0.8 | 0.00990 | 0.01027 | +3.8% |
| 1.0 | 0.00793 | 0.00830 | +4.6% |

The tabulated constants are within ~2% only at α = 0.5–0.6 and drift to 4.6% by
α=1.0 — and are **8.7% low at α=0.3**, which is the one rung where the band arm
loses to the budget-matched Dirichlet FM (+0.040 vs +0.051, §8). `[CONJECTURE]`
that these are connected; the test is cheap, since the fix is a corrected
constant and not a retrain (§9).

> **Correction (2026-08-28).** An earlier version of this section titled itself
> "the driver's constants are accidentally right" and tabulated `γ_match` using
> `‖x₁‖ = 12.2723` where the derivation calls for `Δ = 12.5061` — they differ by
> `√((K−1)/K)`, 1.9%. That inflated `γ_match` (0.02644 vs 0.02596 at α=0.3) and
> made the driver look better than it is at large α and worse at α=0.3. The
> agreement is real but weaker than claimed, and it is an approximation, not a
> coincidence: at small γ the band radius is dominated by the fixed noise floor
> `‖x₀‖`, so matching radius and matching angle nearly coincide.

### 5.4 Transport picture `[EXACT geometry + MEASURED Δ]`

The rescale drops the input at `γ_match(α)`; the field transports it to `γ*`.

| α | γ_match | transport `γ* − γ_match` | measured Δ_recovery |
|---|---|---|---|
| 0.3 | 0.02596 | 0.00404 | +0.040 |
| 0.5 | 0.01574 | 0.01426 | +0.122 |
| 0.6 | 0.01315 | 0.01685 | +0.123 |
| 0.7 | 0.01129 | 0.01871 | +0.117 |
| 0.8 | 0.00990 | 0.02010 | +0.109 |
| 1.0 | 0.00793 | 0.02207 | +0.098 |

α=0.3 has 3.5× the shortest transport and the weakest Δ. Past α=0.6 the decline
tracks task difficulty (token accuracy collapses), not transport.

**Crossover prediction `[EXACT geometry, UNTESTED]`.** `γ_match(α) = γ*` at

```
α_cross = σ(1−γ*) / (γ*·Δ) = 0.2585
```

Below α ≈ 0.26 the rescaled input lands **above** the fixed point, so the descent
should push it *away* from the data and recovery should go **negative**. The
ladder starts at α=0.3, just above the line. **One rung at α=0.15 or 0.2 on the
existing checkpoint tests the entire transport mechanism** — no retraining.

### 5.5 The zero-transport corollary is REFUTED `[MEASURED]`

The transport picture predicted that rescaling to `γ*` itself gives **zero**
transport — `c(γ*) = 0`, so the learned target vanishes there — and that Δ should
therefore collapse toward 0. **Measured, it does not.** Same checkpoint, same
n=256 / 400 steps; only the rescale target changed
(`recovery_gstar030_a{05,06}.json`, g = 0.01827 / 0.01553):

| α | target γ=0.016 (on-manifold) | target γ*=0.03 (radius only) | change |
|---|---|---|---|
| 0.5 | +0.1223 | +0.1092 | −0.0130 (−10.6%) |
| 0.6 | +0.1225 | +0.1076 | −0.0149 (−12.2%) |

Both rungs agree in sign and magnitude, so this is not noise. Three conclusions:

1. **The strict mechanism is wrong.** The field does *not* vanish when the input
   radius matches `γ*`. Radius alone does not determine the network's effective
   `γ`, so "the model infers γ from scale, and `c(γ*)=0` silences it" is not how
   this works. A learned vector field has no obligation to tie its magnitude to a
   perceived interpolation time, and this one does not.
2. **The manifold-matching account (§5.3) survives and is supported.** Scaling
   cannot rotate, so targeting `γ*`'s radius while the angle stays at α's leaves
   the state **off the band manifold** (γ*'s norm, γ_match's direction). A
   consistent ~11% degradation is what that costs. The driver's constants, which
   match radius *and* angle, are the better target — as §5.3 predicted.
3. **The field is robust to the exact rescale target.** A landing ~14% wrong in
   radius costs only ~11% of Δ, so the rescale need not be exact and the energy
   landscape near the band is smooth rather than sharply tuned.

`[CONJECTURE]` The residual sensitivity is *angular*, not radial. Testable by
rescaling to several radii at fixed α and checking whether Δ degrades
symmetrically about `γ_match(α)` rather than falling monotonically toward `γ*`.

---

## 6. Setting the band

### 6.1 `γ*` — the upper end

Set it at a target decode accuracy via §4.1 — never by hand.

**Current design: `γ* = 0.03573`, the exact 99% point** (§4.3). The historical
`0.030` was hand-tuned; the exact formula puts 95% at 0.02979, so 0.030 was the
95% point to within 0.7% by luck rather than construction. Two independent
arguments say the target should be *higher* than 95%:

1. **γ\* is a hard ceiling** (below). 95% caps token accuracy at 95.2% and the
   measured arm already sits at 0.9357 — binding. 99% lifts the cap to 99%.
2. **The realised zero sits below γ\***. EqM is autonomous (`f(x)`, no γ input by
   construction), so `c(γ)` shapes the *target distribution*, not a curve the
   field can represent pointwise. At radius `‖x_γ*‖` the network outputs the
   marginal `E[c(γ)(x₀−x₁) | x]` over the posterior of γ given x, which mixes in
   γ < γ\* where `c > 0`; those contributions cannot cancel, so the learned
   equilibrium is biased **low**. Raising γ\* partially compensates. `[CONJECTURE
   — the offset is unmeasured; see §9]`

What the change moves `[EXACT]`:

| | γ\*=0.030 (empirical) | γ\*=0.03573 (exact, 99%) |
|---|---|---|
| decode ceiling | 95.2% | 99.0% |
| `‖x_γ*‖` | 0.6166 | 0.6588 |
| band width | 1.00× | 1.19× |
| `sample_grad_clip` | 1.0 | **1.191** |
| `α_cross` (§5.4) | 0.2585 | **0.2158** |
| transport at α=0.5 | 0.01426 | 0.01999 (+40%) |

`γ_match(α) = σ/(σ+αΔ)` is **independent of γ\*** (§5.2), so the landing points
and therefore the driver's `g_for_alpha` table are **unaffected** — only the
destination moves, lengthening transport by 40% at α=0.5 and pushing `α_cross`
down so more of the ladder sits in positive-transport territory.

Applied to `band_L256_ep10_d10k_g0`, `band_L256_ep30_d30k_g0` and
`band_L256_ep30_full_d1280L14`. The three already-run cells keep `γ*=0.03`;
every number in §8 is at the old value.

**`γ*` is a hard ceiling on decode accuracy, and it is currently binding**
`[EXACT + MEASURED]`. The equilibrium *is* the state `x_{γ*}`, so the fixed point
the descent converges to decodes correctly with probability `A_K(t(γ*))` — the
design quantile itself. `γ* = 0.030` therefore caps token accuracy at **95.2%**,
no matter how well the field is trained. The measured band arm reaches
`token_acc = 0.9357` at α=0.3, i.e. **it is already at that ceiling**, so the
low-α end of the recovery ladder is limited by γ\* and not by the model. Raising
γ\* to 0.0362 buys a 99% ceiling (§4.3).

This reframes the choice of quantile: it is not a fudge factor but the *target
fidelity of the generative fixed point*. It also rules out setting `γ*` at the
bare 50% threshold (γ=0.0155) — that would place the equilibrium on the decision
boundary and cap accuracy at 50%.

**Tension.** Raising `γ*` lengthens transport (§5.4), lowers `α_cross`, and lifts
the ceiling above — all of which should help recovery; but with `γ_hi = γ*` (§1.1)
raising it also widens the band toward the whole-path regime and erodes the
concentration that gives the band its advantage (KL_bi 0.265 vs 1.582 whole-path,
§8). There is an optimum, not a monotone gain. `[CONJECTURE]` The widening is
modest at the margin: 0.030 → 0.0362 is +21% band width for +3.8 points of
ceiling.

**Implementation gotcha.** `eqm.py`'s own comment: *"the sampler compensates via
`eqm.sample_grad_clip`, which should be scaled by γ\*."* Changing `γ*` without
rescaling `sample_grad_clip` silently detunes the sampler — hence the 1.191 row
above (`1.0 × 0.03573/0.030`).

### 6.2 `γ_lo` — the floor `[DESIGN: 0 · CONTESTED by §6.4]`

**Current design: `γ_lo = 0`**, so the band is `[0, γ*]` and the whole
construction has one free parameter — `γ*` from the decode curve (§6.1) closes
the top, 0 closes the bottom. §6.4 derives an analytic floor that disagrees;
both cases are set out below because neither wins from theory alone.

#### The case for 0 — the endpoint argument

1. **The sampler initialises at exactly γ=0.** Training from 0 means the field
   has seen its own source distribution, so the initialisation — the state every
   trajectory starts from, and where symmetry breaking happens — is
   *in-distribution* rather than extrapolated. A flow whose field is undefined at
   `t=0` takes an arbitrary first step.
2. **`x_{γ=0} = x₀` is stochastic, and that is the diversity budget.** `x₀`
   carries `L×(K−1) = 6656` Gaussian dimensions against the ~1218 bits needed to
   specify 256 tokens. The descent is deterministic *given* `x₀`, exactly as a
   flow-matching ODE is, and that is not a defect.
3. **Predicting `E[x₁]` at γ=0 is correct, not a failure.** With no information
   about `x₁` the L2-optimal output *is* the unigram mean; reproducing the unigram
   where the model knows nothing is the desired behaviour. An earlier draft called
   this "mode-collapse pressure", conflating the correct conditional mean with a
   pathology. The trajectory then acquires token identity as it climbs toward
   `γ*` — which is what the band is for.
4. **No diversity failure exists** `[MEASURED]`. At `γ_lo = 0.005`, samples are
   8/8 unique, mean pairwise character agreement **0.0854** against real text8's
   **0.0722**, `H_ratio = 0.975`. The entropy of `x₀` survives the descent, so the
   case for training γ=0 cannot rest on rescuing diversity.

**Is `E[x₁]` an unescapable attractor?** `[EXACT]` It is a fair worry: the γ=0
field is `∇E ≈ x₀ − μ`, whose descent `x ← (1−η)x + ημ` has a fixed point at
`μ` — a *single* state shared by every position, whose argmax is the most frequent
token. If the trajectory reached it, generation would collapse to the unigram
mode. **It cannot reach it, and the band is exactly why.**

`c(γ) → 0` at `γ*` brakes the descent at radius `‖x_γ*‖ = 0.6166`, while
`‖E[x₁]‖ = 2.392` — **3.88× further out**. Taking the worst case, in which the
field never sharpens and keeps pointing at `μ` for the entire trajectory
(`x(a) = (1−a)x₀ + a·μ`, with `x₀ ⊥ μ`), the brake engages at

```
a = 0.192      state = 0.808·x₀ + 0.192·μ,   ‖x(a)‖ = 0.6166 = ‖x_γ*‖
```

so the descent is halted **19.2% of the way to `μ`**, which sits 5.2× beyond the
distance it is permitted to travel. The attractor exists only in the *unbraked*
linearisation of the γ=0 field; `c(γ)`'s zero at `γ*` removes it. This is the
same brake that defines the equilibrium (§1.1) — it is not an extra safeguard,
it is the construction doing its job.

Two things soften even the 19.2% figure. `E[x₁|x_γ]` **sharpens** from `μ` toward
the specific token as γ rises (decode accuracy 3.7% → 95%, §4.6), so the field
stops pointing at `μ` long before the brake; and the transformer conditions on all
L=256 positions, lowering the effective threshold below the context-free curve
(§4.6 context caveat). Empirically, at `γ_lo = 0.005`, there is no unigram-mode
collapse to be found: KL_uni 0.010 and `H_ratio` 0.975 `[MEASURED]`.

**So the residual risk of `γ_lo = 0` is a *bias*, not a trap** — more training
mass on the μ-drift could tilt early trajectories toward the unigram mode before
context sharpens them. That is a soft, measurable effect (it would show up as
`H_ratio` moving toward 1 and `KL_uni` rising), which is precisely what the A/B
below reads. `[CONJECTURE]`

#### The case against 0 — gradient SNR (§6.4)

The floor question is really *how much of the target is learnable*, and
decodability answers the wrong question. §6.4 derives the Bayes R²

```
ρ(γ) = Var(E[x₁|x_γ]) / Var(x₁)      a function of (K, t) alone
```

and it lags decodability badly at the bottom: at γ=0.005, **11% of tokens decode
but only 1.7% of the target variance is learnable**. Weighting `1−ρ` by the mass
`c(γ)²` that carries it gives the irreducible share `W` of the training signal:

| γ_lo | ρ | W at γ*=0.03573 |
|---|---|---|
| **0** (current design) | 0.000 | **84.4%** |
| 0.005 (historical) | 0.017 | 75.1% |
| 0.0155 | 0.324 | 42.3% |
| 0.019 (§6.4's rule, W=30%) | 0.493 | 30.0% |

Two things follow. **The band is noise-dominated at every setting anyone has
run** — 0 versus 0.005 is an 8-point shift inside an already-saturated regime, so
a large effect from that A/B would be surprising. And **§6.4's rule points an
order of magnitude higher than either**, at `γ_lo ≈ 0.019`.

This also supersedes the framing that used to live here: the old "42% of the
squared-target mass sits in `[0, 0.005)`" counted the sliver's *weight* but
assumed it was entirely wasted and everything above it useful. Weighting by `ρ`
shows the waste is far larger and extends well past the sliver.

#### Why the disagreement is not resolvable from theory

The two cases are not commensurable. §6.2's rests on the **support** of the
learned field — whether the sampler starts inside the training distribution, a
property no amount of gradient efficiency substitutes for. §6.4's rests on
**estimator variance** — how much of the gradient budget buys signal. A floor of
0 maximises the first and minimises the second; 0.019 does the reverse.

What would resolve it: §6.4's prediction that the abrupt KL_bi transition (1.507
→ 0.522 at epoch 7, §8) is an SNR effect, so raising `γ_lo` should make it arrive
**earlier and more reliably**. If it does, gradient SNR is the operative
constraint and the floor should move up; if the transition instead degrades or
vanishes, the initialisation-support argument is doing the real work.
`[CONJECTURE]`

**Status.** `band_L256_ep10_d10k_g0` was intended to isolate `γ_lo`, but as
launched it also carries `γ* = 0.03573` (§6.1), so it measures the **combined**
recipe. Arms `band_L256_ep10_d10k_g0_gs030` (γ_lo only) and
`band_L256_ep10_d10k_g005_gs99` (γ* only) complete the 2×2 against the published
cell. Reference to beat: KL_bi 0.265, Δ@0.5 +0.122. Historical note: the original
`γ_lo = 0.005` was chosen as ~3× chance decodability, which §6.4 identifies as the
ρ ≈ 2% point; every number in §8 comes from that configuration.

### 6.3 What `γ_hi = γ*` forecloses `[EXACT]`

Because `c(γ) = max(0, 1 − γ/γ*)` is **clamped**, every state above `γ*` has zero
drift: the region beyond the equilibrium is a **flat plateau**, not a basin. For
the deterministic NAG-GD sampler this is exactly right — the descent arrives and
stops. It has one consequence worth stating, because it constrains any stochastic
variant:

> A Langevin/SDE sampler injecting noise at the equilibrium would **diffuse away**
> across the plateau, since there is no restoring force to confine it.

A confining equilibrium requires `c` to go *negative* above `γ*` — i.e. dropping
the clamp, which makes the ideal dynamics an Ornstein–Uhlenbeck process about
`γ*`:

```
dγ/dstep = η·c(γ) = η(1 − γ/γ*)          unclamped ⟹ γ* is a true attractor
```

but that requires training above `γ*`, hence `γ_hi > γ*`. **So `γ_hi = γ*` and a
stochastic sampler are mutually exclusive as currently formulated.** `[EXACT]`
The repo already ships the sampler (`eqm.sampler: "sde"` → `sampling/sde.py`,
`x ← x − h·v + √(2αh)·ξ`, α=0 recovering deterministic Euler), so only the clamp
and `γ_hi` stand in the way.

Related: `transformation.sigma_interpolant_max` adds training-time path noise
with `σ(γ) = σ_max·sin(πγ)`, which **peaks at γ=0.5 — far outside the band**.
Inside `[0.005, 0.030]` it never exceeds `sin(π·0.03) = 0.094·σ_max`, so enabling
it as written does almost nothing. A band-aware schedule needs `sin(πγ/γ*)`.

### 6.4 An analytic `γ_lo` — the learnable fraction `[EXACT derivation + MEASURED]`

§6.2's "~3× chance decodability" is a heuristic. There is an exact criterion, and
it comes from the same one-dimensional inversion that fixes `γ*`.

**Reduction.** Given `x_γ`, the source is determined: `x₀ = (x_γ − γx₁)/(1−γ)`.
So the flow target collapses to

```
u = c(γ)·(x₀ − x₁) = [c(γ)/(1−γ)]·(x_γ − x₁)
```

and since `x_γ` is known at prediction time, **the regression is exactly
"predict `x₁` from `x_γ`"**, scaled. Define the learnable fraction as the Bayes
R² of that problem — the scalar prefactor cancels, so it is a property of the
geometry alone:

```
ρ(γ) = Var(E[x₁|x_γ]) / Var(x₁)
```

**It is a function of `(K, t)` only, with the same `t` as §4.1.** Writing
`(x₀)_k = σ(g_k − ḡ)` and `x₁ = Δ(e_c − 1/K)`, the posterior over the token is a
softmax whose argument reduces (the `ḡ` shift is constant in `k` and cancels) to

```
p_k ∝ exp( t·g_k + t²·δ_kc ),        t = γΔ/((1−γ)σ)
```

so with a uniform prior `ρ = (E[Σ_k p_k²] − 1/K)/(1 − 1/K)`, invertible in `t`
exactly like `A_K`. **Validation:** the argmax rate of this posterior reproduces
the independently-derived `A_K(t)` of §4.2 to 3–4 decimals (0.1130 vs 0.1125 at
γ=0.005; 0.4980 vs 0.4985 at 0.0155; 0.9530 vs 0.9524 at 0.030) `[MEASURED]`.

**ρ lags decodability badly at the bottom of the band** — which is why
decodability was the wrong criterion for `γ_lo`:

| γ | decode acc | **ρ (learnable)** |
|---|---|---|
| 0.005 | 11.3% | **1.7%** |
| 0.010 | 26.1% | 10.1% |
| 0.0155 | 49.8% | 32.5% |
| 0.030 | 95.3% | 92.7% |
| 0.03573 | 99.0% | 98.4% |

At the historical `γ_lo = 0.005`, 11% of tokens decode but only **1.7% of the
target variance is learnable**. Inverting ρ gives `γ_lo` directly, and it
retro-fits the hand-tuned value: ρ=1% → 0.00393, **ρ=2% → 0.00533**, ρ=5% →
0.00771, ρ=10% → 0.00998.

**The better criterion — bound the irreducible share of the training signal.**
Weighting ρ by the mass `c(γ)²` it actually carries:

```
W(γ_lo) = ∫ c(γ)²·(1 − ρ(γ)) dγ  /  ∫ c(γ)² dγ        over [γ_lo, γ*]
```

`W` is the fraction of squared-target mass that is **irreducible** — the noise
floor of the regression. `[MEASURED]`

| γ_lo | ρ | W at γ*=0.030 | W at γ*=0.03573 |
|---|---|---|---|
| 0 | 0.000 | **89.2%** | 84.4% |
| 0.005 (historical) | 0.017 | **81.0%** | 75.1% |
| 0.010 | 0.100 | 68.0% | 61.4% |
| 0.0155 | 0.324 | 48.6% | 42.3% |
| 0.0186 | 0.493 | — | 30.0% |

Three consequences, and the first is the important one:

1. **The band is noise-dominated at every setting anyone has run.** The published
   configuration already spends **81%** of its target mass on irreducible
   posterior variance; `γ_lo = 0` moves that to 89.2%. The whole `γ_lo` question
   is an 8-point shift *inside an already noise-dominated regime*, so a large
   effect from the A/B would be surprising. (Earlier framing — "42% of the mass
   is uninformative" — counted only the `[0, 0.005)` sliver and understated it.)
2. **This is gradient SNR, not wrongness.** The irreducible part does not bias the
   target; the Bayes predictor is still what gets learned. It inflates gradient
   *variance*. A regression at ~19% SNR learns slowly and then resolves abruptly,
   which is what the published run's trajectory looks like — flat near KL_bi 1.5
   for six epochs, then 1.507 → 0.522 at epoch 7 (§8). **Prediction: raising
   `γ_lo` should make that transition earlier and more reliable.** Cheaper to test
   than the `γ_lo → 0` direction, and it discriminates the mechanism.
   `[CONJECTURE]`
3. **The wider `γ*` helps SNR.** `W` falls at every `γ_lo` when γ* goes 0.030 →
   0.03573 (89.2%→84.4% at γ_lo=0), because the extra width sits where ρ is high.
   An independent argument for §6.1's change.

**Rule.** Set `γ_lo` by a target `W` (gradient SNR), not by decodability. `W ≤
30%` gives `γ_lo ≈ 0.019` at γ*=0.03573 — far above anything yet run, and the
natural next probe.

**Caveat.** ρ here is *context-free single-position*, the same caveat as §4.6. A
transformer over L=256 exploits n-gram structure, so its effective ρ is higher
and every `W` above is an **upper bound** on the true irreducible share.

### 6.5 The two `γ_lo` requirements, and why they conflict `[EXACT forms]`

§6.4 gives a floor from gradient SNR. The initialisation argument of §6.2 gives a
*ceiling*, and it has an equally closed form — so the two are commensurable after
all. Both live in `t`, which is why they can be compared at all.

**Requirement N (signal).** At least a stated share of the training signal must
be learnable: `W(γ_lo) ≤ W_max` (§6.4). For text8 at `W ≤ 30%`:

```
t_N = 2.40     ⟹   γ_lo ≥ 0.0188
```

**Requirement S (support).** The sampler starts at `x₀ ~ N(0, σ²P)`. The band at
`γ_lo` is a mixture of K Gaussians centred at `γ_lo·x₁⁽ᵏ⁾` with the same
covariance. The displacement between the init's centre and a component centre,
in units of the per-coordinate noise std, is

```
γ‖x₁‖/((1−γ)σ) = t·√((K−1)/K) ≈ t
```

— the same `t` again. The Gaussian's own radius in `K−1` dimensions is `√(K−1)`,
so requiring the init to sit within a fraction `κ` of that radius gives

```
Requirement S:   t_lo ≤ κ·√(K−1)
```

| κ | t_lo | γ_lo |
|---|---|---|
| 0.05 | 0.255 | 0.00203 |
| **0.123** | 0.627 | **0.00499** ← the historical `γ_lo = 0.005` |
| 0.20 | 1.020 | 0.00809 |
| 0.475 | 2.401 | 0.01884 ← what Requirement N demands |

**The historical 0.005 is exactly κ = 0.123** — the init sits 12% of a noise
radius from the nearest trained state. Satisfying N instead puts it at **48%**,
nearly half a Gaussian radius outside the trained region.

**At K=27 the two are incompatible** (`t_S/t_N = 0.26`): no `γ_lo` satisfies both.
That is the quantitative form of §6.2's tension — the choice is which requirement
to violate, and by how much.

**But the conflict is a small-vocabulary artifact.** `σ` and `Δ` cancel in `t`, so
only `K` matters, and `t_S ∝ √K` while the SNR threshold grows only as `√(log K)`:

| K | `t_S` (κ=0.123) | `t_N ≈ √(2 ln K)` | `t_S/t_N` |
|---|---|---|---|
| 27 | 0.63 | 2.57 | **0.24** |
| 256 | 1.96 | 3.33 | 0.59 |
| 1024 | 3.93 | 3.72 | **1.06** ← crossover |
| 50257 | 27.6 | 4.65 | 5.93 |

They cross near **K ≈ 1024**, beyond which both requirements are satisfiable at
once. This is the sharp version of §7.1's conjecture that the thin-band
awkwardness is a text8 artifact — with a specific vocabulary threshold rather
than a hand-wave. `[CONJECTURE]` `√(2 ln K)` is a *proxy* for how the SNR
threshold scales, justified at K=27 (2.57 against the computed 2.40, 7% off) but
not derived; the exact version needs `ρ` recomputed per K, which
`scripts/band_geometry.py --K …` does.

**`κ` is a choice; the form is not.** What is analytic is `t_lo ≤ κ√(K−1)`, and
that is what makes S directly comparable to N. Both are computed by
`scripts/band_geometry.py`, which prints `COMPATIBLE` / `INCOMPATIBLE` and the
ratio for any `(K, ε, σ)`.

---

## 7. Scaling with K, ε, σ

Two different K-dependences, and they differ — this is the practically important
result.

```
γ_eq   = σ√K / (σ√K + Δ)          — retains √K
γ_dec  = σt*/(Δ + σt*),  t*=A_K⁻¹ — √K enters only via A_K and Δ
γ_match(α) = σ/(σ + αΔ)           — √K cancels EXACTLY
```

`γ_eq` grows with `√K/log K` because the noise norm grows as `√K` while the data
norm grows only as `log K`. `γ_match` does not, because the α-perturbation
carries `√K` too.

**Vocabulary** (ε=1e-4, σ=0.1), `γ_eq`:

| K | Δ | γ_eq | vs text8 |
|---|---|---|---|
| 27 | 12.51 | 0.0399 | 1.00× |
| 256 | 14.76 | 0.0978 | 2.45× |
| 1024 | 16.14 | 0.1654 | 4.15× |
| 50257 | 20.04 | 0.5281 | **13.2×** |

**CLR smoothing** (K=27), a fine adjustment — logarithmic, 2.6× over four orders:
ε=1e-6 → 0.0295, 1e-4 → 0.0399, 1e-2 → 0.0618.

**Source σ** (K=27) — the free knob, no data or vocabulary change:
σ=0.05 → 0.0204, 0.1 → 0.0399, 0.5 → 0.1720.

### 7.1 Consequences

**The band widens with vocabulary, so the approach should scale favourably.**
At K=27 the band is a razor-thin shell (3% of the path) — which is why everything
needs renormalising and why γ=1 readings invert. At BPE scale the same recipe
gives a band reaching γ≈0.4, a genuinely thick region. *The thin-band awkwardness
is a text8 artifact, not intrinsic to the method.* `[CONJECTURE]`

**Inputs land relatively deeper as K grows.** Since `γ_eq` grows as `√K` but
`γ_match` only via `log K`, the ratio `γ_match(0.5)/γ_eq` falls: 0.394 at K=27,
0.137 at K=256, 0.019 at K=50257 — progressively more transport headroom.

**ε is as strong a lever as a 10× vocabulary change.** ε: 1e-4 → 1e-3 moves Δ by
−18%, the same magnitude as K: 27 → 256. If ε is ever retuned, every hard-coded
rescale constant is stale. This argues for deriving them, not tabulating them.

---

## 8. Empirical anchors

All from `runs/band_L256_ep10_d10k` (d1024/10L/16H, B=8, L=256, 10k windows,
10 epochs, step 12500) — budget-matched to
`runs/sflm_bench_a100_20g_L256/DirichletFM` (identical architecture, data, epochs;
`train_meta.json` confirms 10k windows / batch 8 / 10 epochs).

**Two Dirichlet FM arms exist at this architecture and they are easy to confuse.**
Both are d1024/10L/16H, B=8, L=256; only the budget differs `[MEASURED]`:

| arm | windows × epochs | KL_uni | KL_bi | KL_tri | Δ@0.3 | Δ@0.5 | Δ@0.7 |
|---|---|---|---|---|---|---|---|
| `DirichletFM` — **budget-matched** | 10k × 10 | 0.009 | **0.959** | 4.088 | +0.051 | +0.090 | +0.102 |
| `DirichletFM_ep30_d30k` — 9× budget | 30k × 30 | 0.008 | **0.208** | 1.250 | +0.105 | +0.193 | +0.190 |
| **band, this run** | 10k × 10 | 0.010 | **0.265** | 1.744 | +0.040 | +0.122 | +0.117 |

The band arm beats its budget-matched peer by 3.6× on KL_bi and reaches 0.265 at
**1/9** the budget of the 30k×30ep arm's 0.208. `band_L256_ep30_d30k_g0` is the
matched head-to-head against that second row.

**Generation** `[MEASURED]` — KL_uni 0.010, **KL_bi 0.265**, KL_tri 1.744,
H_ratio 0.975, vs budget-matched Dirichlet FM KL_bi 0.959 and whole-path EqM
1.582. The band's native, un-rescaled operation, and its strongest result.

**Recovery** `[MEASURED]` — Δ@0.5 = +0.122 vs matched Dirichlet FM +0.090; beats
it at every α ≥ 0.5; loses at α=0.3 (+0.040 vs +0.051). The whole-path EqM arm
*without* rescale is **negative** past α=0.3 (−0.094 at α=0.5), i.e. it destroys
tokens. Full-corpus Dirichlet FM reaches +0.282 — not budget-matched.

**OOD** `[MEASURED]` — read in-band, `γ`-swept, rate 0.15:

| γ | replace seq | replace char (sign-corrected) | shuffle seq |
|---|---|---|---|
| 0.005 | 0.672 | 0.544 | 0.513 |
| 0.020 | 0.952 | **0.743** | 0.770 |
| 0.030 = γ* | **0.967** | 0.579 | **0.822** |
| 1.0 (control) | **0.108** | 0.518 | 0.496 |

Two structural facts: (i) the unrenormalised control **inverts** (0.108); (ii)
localisation peaks at γ=0.020 (70% decodable — the field is still deciding per
position) while sequence detection peaks at γ*=0.030 (95% — settled, so residual
disagreement only aggregates). This mirrors `BayesLinHead`'s two-pass design in
[`bayes_linear_ood.md`](bayes_linear_ood.md), where energy is read at
`t_eval=4.5` and variance at `t_var=7.5`, and the variance signal *inverts* if
read at the wrong point. **One readout point per channel.** `[CONJECTURE]`

**Rate axis — the two channels move oppositely** `[MEASURED]`. Read each channel
at its own γ, sweeping the corruption rate (`replace`, char sign-corrected):

| rate | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| localise — char `‖∇E‖` @ γ=0.020 | **0.757** | 0.748 | 0.743 | 0.738 | **0.730** |
| flag — sequence energy @ γ*=0.030 | **0.755** | 0.909 | 0.967 | 0.994 | **1.000** |

Monotone, in opposite directions, and the mechanism is not subtle: localisation
asks *which* characters are wrong, so the cleaner the surrounding sequence the
more sharply the few remaining corrupt characters stand out — best at low rates,
degrading as corruption becomes the norm. Sequence detection asks *whether the
window is corrupt at all*, so every additional corrupt token is more evidence —
monotonically increasing, saturating at **1.000 by rate 0.30**. `shuffle` shows
the same sequence trend (0.624 → 0.955) with a flat char channel (~0.53).

This is the rate-axis twin of the γ-axis split above, and it reinforces the same
rule: **one readout point per channel**, and additionally *one corruption regime
per channel* — a localiser should be benchmarked sparse, a flagger dense.

**Sign caveat.** The in-band char figures are `1 − AUROC`: per-position `‖∇E‖` is
*lower* at corrupted characters, so the raw AUROC is below chance. The flip is
exact for char and mean-pooling (mean commutes with negation) but **not for
`word_max`** (max of −x = −min of x), which needs a genuine min-pooled re-run.
The flip is post-hoc, justified by being systematic and monotone across five γ in
both corruption schemes; the correct fix is to fix the sign convention a priori
in `band_ood_score.py`.

**No budget-matched OOD comparator exists.** The paper's 0.981 denoiser-NLL comes
from the *full-corpus* Dirichlet FM (d1280/14L, ~28 epochs). Comparing this
10k-window model against it is not like-for-like. Running the denoiser NLL on
`runs/sflm_bench_a100_20g_L256/DirichletFM/epoch_final.pt` with the same
corruption schedule and `_bench_common.word_metrics` pooling would close the gap
cheaply (forward passes only).

---

## 9. Open questions and testable predictions

1. **α_cross** — recovery should go negative below α ≈ 0.26 (§5.4). One rung at
   α=0.15/0.2, no retraining. *The single highest-information experiment here.*
2. ~~**Zero-transport**~~ — **RESOLVED 2026-08-28, prediction REFUTED** (§5.5).
   Δ fell only ~11% (α=0.5: +0.1223→+0.1092; α=0.6: +0.1225→+0.1076), not to 0.
   Follow-up: is the residual sensitivity angular rather than radial? Sweep the
   rescale radius at fixed α and test for symmetry about `γ_match(α)`.
3. **γ_lo = 0** (§6.2) — does the 42%-of-target-mass cost outweigh having the
   sampler's initialisation in-distribution? **Running**
   (`band_L256_ep10_d10k_g0`, identical except `γ_lo`).
4. **Budget-matched head-to-head** — `band_L256_ep30_d30k_g0` (30k × 30ep)
   against `DirichletFM_ep30_d30k`'s KL_bi 0.208 / Δ@0.5 +0.193. **Queued.**
5. **`γ_lo` by gradient SNR** (§6.4) — the W-rule points at `γ_lo ≈ 0.019`
   (W=30%), ~4× the historical floor and far above the `γ_lo = 0` now running.
   It predicts the epoch-7-style transition should arrive **earlier and more
   reliably** at higher `γ_lo`, which discriminates the SNR mechanism from a
   capacity story. One training run, and it is the cheapest test of §6.4.
6. **γ\* ceiling** (§6.1) — γ\*=0.030 caps token accuracy at 95.2% and the arm
   already sits at 0.9357. Does γ\*=0.0362 (99% ceiling) lift the low-α rungs, or
   does the wider band cost more in concentration than it gains? One training run.
7. **Stochastic equilibrium** (§6.3) — unclamping `c` with `γ_hi > γ*` turns γ\*
   into a true attractor and makes the shipped SDE sampler usable. Untested; the
   clamp currently makes noise injection diffuse.
8. **γ\* sweep** — does the concentration/transport tension (§6.1) have an
   interior optimum? Costs one training run per point.
9. **α=0.3 anomaly** — is the 8.7% rescale error (§5.3) the reason it is the only
   losing rung? Re-run with the exact `g = σ/(σ+αΔ) = 0.02596`. No retraining.
10. **Vocabulary scaling** — does the band's advantage grow with K as §7.1
   predicts? Testable at K=256 with a byte- or BPE-level corpus.
11. **Min-pooled word OOD** — the one number the paper's protocol actually
   reports, currently uncomputable from stored data.

---

## 10. Literature

### 10.1 Directly used by this repository (IDs verified in-repo)

- **Equilibrium Flow Matching (EqM)** — Wang & Du, 2025. [arXiv:2510.02300](https://arxiv.org/abs/2510.02300).
  The conservative-gradient formulation `∇_x⟨x, f(x)⟩` and NAG-GD sampling this
  band variant modifies.
- **Dirichlet Flow Matching** — Stark et al., 2024. [arXiv:2402.05841](https://arxiv.org/abs/2402.05841).
  The baseline the band arm is benchmarked against.
- **Statistical Flow Matching** — [arXiv:2405.16441](https://arxiv.org/abs/2405.16441).
  Fisher–Rao `√μ` sphere; the alternative simplex geometry.
- **Flow Matching for Generative Modeling** — Lipman et al. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
  The straight-interpolant construction `x_γ = (1−γ)x₀ + γx₁`.
- **D3PM** — Austin et al., 2021. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006).
  The uniform-process peer for BPC.
- **SEDD** — Lou et al. [arXiv:2310.16834](https://arxiv.org/abs/2310.16834).
- **Spilled Energy in LLMs** — Minut, Dewidar & Masi, ICLR 2026. [arXiv:2602.18671](https://arxiv.org/abs/2602.18671).
  Note the sign typo in Eq. 8 — follow the reference implementation.

### 10.2 Compositional geometry

- **Aitchison, J. (1982).** "The Statistical Analysis of Compositional Data."
  *JRSS-B* 44(2), 139–177. The CLR/ALR transforms and the simplex as a vector
  space — the `Δ√((K−1)/K)` norm in §2.2 is a direct consequence of the CLR
  construction.
- **Egozcue, J.J. et al. (2003).** "Isometric Logratio Transformations for
  Compositional Data Analysis." *Mathematical Geology* 35(3), 279–300. The ILR
  basis used in `geometry.py` for visualisation.

### 10.3 Noise-schedule theory — the closest precedent for §7

**This is the most important external connection.** The result that the optimal
noise schedule shifts with data dimensionality is established in the diffusion
literature, and our `√K` scaling is the same phenomenon in vocabulary rather than
resolution.

- **Kingma et al., Variational Diffusion Models.** [arXiv:2107.00630](https://arxiv.org/abs/2107.00630).
  Parameterising the diffusion process by **signal-to-noise ratio** rather than
  time. `γ_eq` and `γ_dec` are SNR thresholds in exactly this sense, and the band
  is an SNR window.
- **Chen, T., On the Importance of Noise Scheduling for Diffusion Models.**
  [arXiv:2301.10972](https://arxiv.org/abs/2301.10972). Shows the optimal schedule
  **shifts with image resolution** because effective SNR scales with the number
  of correlated dimensions, and proposes rescaling to compensate. Direct
  precedent for §7's claim that the band must move with K.
- **Hoogeboom et al., simple diffusion.** [arXiv:2301.11093](https://arxiv.org/abs/2301.11093).
  Resolution-dependent noise-schedule shifting, same mechanism.
- **Karras et al., Elucidating the Design Space of Diffusion-Based Generative
  Models (EDM).** [arXiv:2206.00364](https://arxiv.org/abs/2206.00364). Training
  noise levels drawn from a deliberately chosen distribution concentrated where
  the task is informative — the same principle as restricting γ to the band, and
  the natural reference for §4.6.

### 10.4 Supporting

- **Szegedy et al. (2016),** "Rethinking the Inception Architecture."
  [arXiv:1512.00567](https://arxiv.org/abs/1512.00567). Label smoothing — the ε
  that sets Δ and hence the entire band location.
- **David, H.A. & Nagaraja, H.N., *Order Statistics* (3rd ed., Wiley, 2003).**
  The max-of-normals asymptotics behind the `√(2 ln n)` heuristic of §4.4. The
  relevant point is that `E[max]` is a *mean*, not a quantile, so it cannot
  target a stated decode accuracy — see §4.4 for the exact replacement.

> **Verification note.** §10.1 arXiv IDs are taken from this repository and are
> internally consistent. §§10.2–10.4 are standard references cited from
> background knowledge and were **not** verified against a live index while
> writing; check the identifiers before they go into a paper bibliography.

---

## 11. Reproducing every number here

The geometry needs no GPU. **`scripts/band_geometry.py` derives every constant
in this document from `(K, ε, σ)`** — run it rather than trusting the tables:

```bash
uv run python scripts/band_geometry.py            # full derivation, text8 defaults
uv run python scripts/band_geometry.py --K 256    # any vocabulary
uv run python scripts/band_geometry.py --yaml     # emit sweep overrides
```

It computes Δ and the norms (§2), inverts `A_K` for `γ*` (§4), Monte-Carlos `ρ`
and integrates `W` for Requirement N (§6.4), evaluates Requirement S and prints
`COMPATIBLE`/`INCOMPATIBLE` with the ratio (§6.5), and emits the exact
`g(α) = σ/(σ+αΔ)` ladder (§5.3). Each function is marked EXACT (closed form or
quadrature) or MONTE CARLO, with the derivation in its docstring.

This closes what earlier versions of this section listed as the next step —
`gamma_star`, `gamma_lo` and the `g_for_alpha` table are no longer hard-coded,
so a vocabulary or smoothing change is correct by construction. Three sets of
hand-tabulated constants replaced by one principle.

The measured anchors come from
`runs/band_L256_ep10_d10k/{eval.json, recovery_bandmap_a*.json, band_ood.json}`
and `runs/sflm_bench_a100_20g_L256/DirichletFM/`.
