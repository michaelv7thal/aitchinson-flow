# The band setup for EqM — how it actually works

*Written 2026-08-28 against `sweeps/band_L256.yaml:band_L256_ep10_d10k`,
`src/aitchinson_flow/models/eqm.py`, `scripts/recovery_check.py` and
`scripts/band_ood_score.py`. Every number below is either read off the code or
measured; the measured ones are marked.*

The band is three changes that all follow from one geometric fact. Take the fact
first, because the rest is bookkeeping.

---

## 1. The fact: the interpolant is noise-dominated almost everywhere

EqM trains on a straight interpolant between a noise sample and a data point,

```
x_γ = (1 − γ)·x₀ + γ·x₁        γ ∈ [0,1],  γ=0 pure noise, γ=1 clean data
```

where `x₁` is the label-smoothed CLR feature of a token and `x₀ = σ·randn`
centred on `V_d` (`σ = cfg.eqm.source_sigma = 0.1`). The two have wildly
different scale. Measured, per position, K=27:

| quantity | value | where it comes from |
|---|---|---|
| `‖x₁‖` | **12.2723** | CLR of a label-smoothed one-hot (measured; the run log prints `embed_norm_mean=12.2723`) |
| `‖x₀‖` | **0.5043** | `σ·√(K−1)` = 0.1·√26 = 0.5099 (measured 0.5043) |

That is a **24× disparity**. Since the two terms are orthogonal in expectation
(both zero-mean across K), `‖x_γ‖² = ((1−γ)‖x₀‖)² + (γ‖x₁‖)²`, and the data term
only catches up with the noise term when

```
γ·12.27 ≈ 0.50    ⟹    γ ≈ 0.041
```

**So for γ above ~0.05 the interpolant is already essentially clean data, and
below ~0.005 it is essentially pure noise.** The entire transition — every γ at
which the model sees a genuinely ambiguous mixture — is squeezed into roughly
`γ ∈ [0.005, 0.05]`. Sampling γ uniformly on [0,1], as standard EqM does, spends
~95% of training on inputs that are trivially clean.

The band is just: *train only in that transition window.*

```yaml
eqm.gamma_lo: 0.005
eqm.gamma_hi: 0.03
```

Measured shell radii inside the band:

| γ | `‖x_γ‖` | `c(γ)` |
|---|---|---|
| 0.005 | 0.5055 | 0.833 |
| 0.016 | 0.5337 | 0.467 |
| 0.030 | 0.6122 | 0.000 |

The band is a thin spherical shell of radius **≈0.51–0.61**, against a data
manifold sitting out at radius 12.27. Hold on to that; §4 is entirely about it.

---

## 2. Change one: where γ is drawn

`eqm.py:201`

```python
gamma = torch.rand(B).pow(s.gamma_power)      # gamma_power = 1.0 here → uniform
lo, hi = s.gamma_lo, s.gamma_hi               # 0.005, 0.03
if (lo, hi) != (0.0, 1.0):
    gamma = lo + (hi - lo) * gamma            # uniform on [0.005, 0.03]
```

With `gamma_power: 1.0` the `.pow()` is the identity, so γ is **uniform on the
band**. (The repo's usual `gamma_power=0.5` trick — upweight γ≈1 because "that
is where the signal lives" — is switched off here, and rightly so: inside the
band *every* γ is a signal regime, so there is nothing to reweight toward.)

Defaults are `(0.0, 1.0)`, which makes the remap inert — the band is opt-in and
every non-band arm is unaffected.

---

## 3. Change two: where the equilibrium sits

This is the substantive change. The FM target is

```
u_tgt = c(γ)·(x₀ − x₁)
```

and `c(γ)` is what decides the fixed point of the descent. `eqm.py:_c_gamma`:

| strategy | `c(γ)` | target vanishes at |
|---|---|---|
| `linear` (standard EqM) | `1 − γ` | γ = 1, i.e. **the data vertex** |
| `band` | `max(0, 1 − γ/γ*)` | γ = γ*, i.e. **`x_γ*`, a point on the shell** |

```yaml
eqm.decay_strategy: band
eqm.gamma_star: 0.03      # == gamma_hi
```

Because `γ* = 0.03 = gamma_hi`, the target magnitude **decays linearly to exactly
zero at the top of the band**: `c(0.005) = 0.833`, `c(0.03) = 0`. The model is
being told "the equilibrium is the top edge of the band", not "the equilibrium
is the clean token".

This is the whole idea. Standard EqM puts a minimum at every vertex of the
simplex — which is why its unconditional sampling fails (`NOTE_WHY_UNCONDITIONAL_FAILS.md`,
and the paper's "a minimum at every vertex"). The band never asks the field to
have a minimum at a vertex at all. It asks for a minimum on a shell of radius
0.61, where the neighbourhood structure is still smooth.

The normalisation is deliberate: `c(γ)` is scaled so `c(0)=1`, matching the
`linear` strategy, so the **training-loss scale is unchanged** and losses stay
comparable across arms. The code comment flags the consequence — the sampler
must compensate via `eqm.sample_grad_clip`, "which should be scaled by γ*".

---

## 4. Change three: the rescale, and why it is legitimate

Here is the problem the band creates. The field was only ever trained on inputs
of radius ≈0.5. A real candidate window — clean text, or text you want to repair
or screen — has radius 12.27, or worse. Feeding it straight in is **24× outside
anything the model has seen**. It is out of distribution by construction.

So before reading the field, the candidate is scaled onto the shell:

```
z ← g·z          g = ‖x_γ‖ / ‖z_α‖
```

`drive_band_L256.sh:g_for_alpha`, with `‖x_γ‖ = 0.539` and
`‖z_α‖ = 12.27·√(1 + K·α²)`, K=27. Verified against the implementation:

| α | `‖z_α‖` | `g` (computed) | `g` (driver table) |
|---|---|---|---|
| 0.3 | 22.729 | 0.02371 | 0.0237 |
| 0.5 | 34.165 | 0.01578 | 0.0160 |
| 1.0 | 64.939 | 0.00830 | 0.0083 |

**Why this does not cheat.** `g` is a *single positive scalar* applied to the
whole window. It cannot move an argmax: `argmax(g·z) = argmax(z)` for `g > 0`.
So the perturbed input's token predictions are identical before and after, and
`Δ@α = token_acc − token_acc_perturbed` is still measured against the
*unrescaled* input. The rescale buys the model an in-distribution input; it
gives away no information about the answer.

Two implementation notes:

- `--band-mode scale` is the pure rescale `z ← g·z` (`recovery_check.py:450`).
  `--band-mode interp` instead mixes in fresh source noise,
  `z ← (1−g)·x₀ + g·z`. The runs use `scale`.
- A coincidence worth knowing at α=0.5: `(1−g)·σ = g·α·12.27` (0.0984 vs
  0.0982). At that one rung the rescale lands exactly where the band's own
  source noise would put it, so `scale` and `interp` nearly coincide.

The driver prints `embed_norm_mean` at every ladder rung precisely so this table
can be falsified: if it is not ≈12.27, the `g` constants are stale and every
recovery number is suspect. In the L=256 run it printed **12.2723**.

---

## 5. Sampling: NAG-GD on the conservative gradient

Unchanged by the band, but the constants are band-scaled. `eqm.py:508`:

```python
grad = clip(∇_x ⟨x, f(x)⟩)
for _ in range(max_steps):
    x_last = x
    x = x - eta * grad
    grad = clip(∇_x ⟨x, f(x)⟩ evaluated at  x + mu*(x - x_last))   # Nesterov lookahead
```

| knob | value | note |
|---|---|---|
| `sample_eta` | 0.005 | vs the 0.1 default — steps must be small relative to a 0.5-radius shell |
| `sample_mu` | 0.9 | Nesterov momentum |
| `sample_grad_clip` | 1.0 | per-position L2 cap; stops the cold-start spike being amplified by momentum |
| `sample_max_steps` | 400 | the `eval_fixed` protocol |

The energy is **`⟨x, f(x)⟩`** and the field used is its gradient — the same
object as the training target. Train and sample must use the same field; a past
bug where sampling used raw `f(x)` is in `SESSION_SUMMARY.md`.

> **Dead knob.** The cell sets `eqm.sample_gamma: 1.5`, but EqM is **autonomous**
> — `time_conditioning: off`, as the method requires — and `_compute_grad` only
> consults `sample_gamma` when time conditioning is on. In these runs it does
> nothing. Don't tune it.

**Autonomy is the method, and it has a consequence for `c(γ)`.** The field is
`f(x)`, with no γ input: the equilibrium formulation is precisely that sampling
is descent on a *static* field. So `c(γ)` shapes the **training target
distribution** but is not a curve the network can represent pointwise — it is
trained against the marginal

```
E[ c(γ)·(x₀ − x₁) | x_γ = x ]        over all (γ, x₀, x₁) consistent with x
```

Hence "the equilibrium is at `γ*`" (§3) is a statement about the *target design*.
Where `‖∇E‖` actually vanishes is an emergent property of that marginalisation,
and there is no analytic route to it — it has to be measured on the checkpoint.
Two reasons to expect the true zero to sit **short of** `γ*`:

1. `γ` is drawn on the **half-open** `[γ_lo, γ*)` (`torch.rand` → `[0,1)`), so the
   model never once receives a zero target: over 2M draws, max γ = 0.02999999 and
   `count(c == 0) = 0` `[MEASURED]`. The zero is a limit, not a training signal.
2. Landing a recovery input exactly on radius `‖x_γ*‖ = 0.6166` should collapse Δ
   toward 0 if the field vanished there. Measured: Δ falls only from **+0.122 to
   +0.109** at α=0.5 `[MEASURED]`, i.e. the field is still doing most of its work
   at the nominal equilibrium.

---

## 6. The OOD read

`band_ood_score.py` applies the same idea to detection: map the candidate into
the band, *then* read the field.

```
x = γ·clr(ids)                       # --noise-draws 0, pure rescale
x = γ·clr(ids) + (1−γ)·σ·ε           # with noise draws
```

and scores per-position `‖∇E‖` and sequence energy, corrupt vs clean. It sweeps
`--gammas 0.005,0.01,0.016,0.02,0.03,1.0`, where **γ=1.0 is the data-scale
control** — the un-rescaled read the paper already has, i.e. the "minimum at
every vertex" reading.

**Protocol rule: read the model inside the band, always.** The rescale is a
positive homothety — bijective, argmax-preserving, and it maps the candidate
into the domain the field was trained on. A band-trained model evaluated at
γ=1.0 is being asked about states of norm 12.27 when it has only ever seen
~0.51, and it answers accordingly: its sequence AUROC there is **0.108**, i.e.
*inverted*. That number is an extrapolation artifact, not a result, and it must
not be reported as the arm's score.

This is not special pleading for the band arm. The whole-path control reads
**better in-band (0.853) than at its own native data scale (0.683)** even though
γ=1.0 is in-domain for it. Renormalisation improves detection for both arms, so
it is a statement about where the energy is informative, not a trick.

---

## 7. What it bought, at L=256 (d1024/10L, 10k windows × 10 epochs)

Generation, `KL_bi` (lower is better):

| arm | KL_uni | KL_bi | KL_tri |
|---|---|---|---|
| **band, this run** | **0.010** | **0.265** | **1.744** |
| Dirichlet FM, matched budget | 0.009 | 0.959 | 4.088 |
| EqM det-CLR, whole path | 0.018 | 1.582 | 5.042 |
| Discrete FM (flagged collapsed) | 0.001 | 1.466 | 5.141 |

Recovery, `Δ@α` with the rescale (higher is better):

| α | band | DirFM matched | DirFM full corpus | EqM whole path, **no** rescale |
|---|---|---|---|---|
| 0.3 | +0.040 | +0.051 | +0.139 | −0.000 |
| 0.5 | **+0.122** | +0.090 | +0.282 | −0.094 |
| 0.6 | **+0.123** | — | — | — |
| 0.7 | **+0.117** | +0.101 | +0.289 | −0.158 |
| 0.8 | **+0.109** | +0.094 | +0.184 | −0.125 |
| 1.0 | +0.098 | — | — | +0.071 (L=40) |

The sign flip in the last column is the clearest statement of what the band
does. The same architecture on the whole path, without the band and rescale,
**destroys** tokens at every damage level past 0.3. With them, it repairs.

It beats the matched-budget Dirichlet FM from α=0.5 up, and loses to the
full-corpus one everywhere.

---

## 8. Detection: two channels, opposite rate behaviour

Read in-band (§6), the energy supports **two different detectors**, and they are
not competing readings of one signal — they answer different questions and move
in opposite directions as corruption density rises.

`replace`, at each channel's own readout point (char figures sign-corrected,
see the caveat below):

| corruption rate | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| **localise** — char `‖∇E‖` @ γ=0.020 | **0.757** | 0.748 | 0.743 | 0.738 | **0.730** |
| **flag** — sequence energy @ γ*=0.030 | **0.755** | 0.909 | 0.967 | 0.994 | **1.000** |

**Why they move oppositely.** Localisation asks *which* characters are wrong: the
cleaner the surrounding sequence, the more sharply a few remaining corrupt
characters stand out against it, so the signal is strongest at low rates and
degrades as corruption becomes the norm. Sequence detection asks *whether this
window is corrupt at all*: every additional corrupt token adds evidence, so it
strengthens monotonically with rate and **saturates at 1.000 by rate 0.30**.

The same split appears on the γ axis (§4.6 of `band_geometry_theory.md`):
localisation peaks at γ=0.020, where the field is still deciding per position,
and sequence detection peaks at γ*=0.030, where positions have settled and only
residual disagreement aggregates. **One readout point per channel** — the same
two-pass structure `BayesLinHead` uses (`docs/bayes_linear_ood.md`), where
reading the variance at the energy's point inverts the signal.

Against the whole-path control at the sequence channel, band training wins:
**0.967 vs 0.853** at rate 0.15.

**Sign caveat.** Raw per-position `‖∇E‖` AUROC is *below* chance (0.243–0.47), so
the char figures above are `1 − AUROC`: the gradient is *lower* at corrupted
characters. The flip is systematic and monotone across five γ and both
corruption schemes, and it is exact for char and mean-pooling, but **not for
`word_max`** (max of −x = −min of x), which needs a genuine min-pooled re-run.
The correct fix is to settle the sign convention a priori in
`band_ood_score.py`.

## 9. What is not yet established

The samples are not words —

> `'ecoat aalsqsarut w tha aults lore d ng pomngnton oofnanenfo wem aliti e ne es e'`

but this arm is a **10-epoch, 10k-window (~2.5M character) d1024/10L model**,
against a full-corpus target trained on ~90M characters for ~28 epochs — roughly
1/36 the data passes. At *matched* budget it beats Dirichlet FM by 3.6× on KL_bi
(0.265 vs 0.959) and beats it on recovery from α=0.5 up, so lexical failure at
this budget is a budget statement, not a structural one.

The same caution applies to

```
grad_at_gen  0.389        # ‖∇E‖ at the model's own samples
grad_at_gt   4.718        # ‖∇E‖ at real corpus text
```

Real text is not a critical point of the learned energy. That is what an
*undertrained* energy looks like as much as a mis-specified one, and this run
cannot separate the two. **No budget-matched OOD comparator exists**: the
paper's 0.981 denoiser-NLL comes from the full-corpus Dirichlet FM, so the
localisation gap is currently measured against a model with 36× the training.

What this run does *not* show, contrary to an earlier reading of it: any
diversity failure. Samples are 8/8 unique with mean pairwise character
agreement 0.0854 against real text8's 0.0722, and `H_ratio` is 0.975 — the
deterministic descent preserves the entropy of x₀ (6656 Gaussian dimensions
against the ~1218 bits needed for 256 tokens).

---

## 10. The knobs, in one place

```yaml
eqm.gamma_lo:         0.005   # band floor — below this, pure noise
eqm.gamma_hi:         0.03    # band ceiling
eqm.gamma_star:       0.03    # == gamma_hi: c(γ) hits 0 here, so the
                              #   equilibrium is x_γ*, not the data vertex
eqm.decay_strategy:   band    # c(γ) = max(0, 1 − γ/γ*)
eqm.gamma_power:      1.0     # identity → γ uniform on the band
eqm.gradient_lambda:  1.0     # c(γ) is divided by this
eqm.sample_eta:       0.005   # small steps for a radius-0.5 shell
eqm.sample_grad_clip: 1.0     # scale with γ* if you change it
eqm.sample_max_steps: 400
eqm.sample_gamma:     1.5     # DEAD unless time_conditioning != "off"
eqm.source_sigma:     0.1     # sets ‖x₀‖ = σ√(K−1); train and sample must match
```

Changing `gamma_hi`, `gamma_star` or `source_sigma` invalidates the `g_for_alpha`
table in the drivers, because it moves either the shell radius `‖x_γ‖` or the
data radius. Recompute `g = ‖x_γ‖/‖z_α‖` before trusting any recovery number.

---

## 11. Where to look

| what | where |
|---|---|
| γ draw + band remap | `models/eqm.py:201` |
| `c(γ)` and the equilibrium | `models/eqm.py:_c_gamma`, `band` branch |
| NAG-GD sampler | `models/eqm.py:508` |
| conservative gradient | `models/eqm.py:_compute_grad` |
| rescale table | `drive_band_L256.sh:g_for_alpha` |
| rescale application | `scripts/recovery_check.py:445` |
| OOD band mapping | `scripts/band_ood_score.py:_map_into_band` |
| dev-scale cell | `sweeps/band_L256.yaml:band_L256_ep10_d10k` |
| full-scale cell | `sweeps/band_L256.yaml:band_L256_ep30_full_d1280L14` |
