# Findings — γ-bucket flow_loss asymmetry in Compositional EqM training

Investigation triggered while running `RUNBOOK_COMPOSITIONAL_EQM_TEST.md`.
Author: Claude session, 2026-05-10. Branch `capstone-project` @ `25ac5fe`.

## 1. Observation

In the per-epoch tqdm postfix for the compositional EqM cells (e.g.
`comp_mse_seed42`, 5 epochs, `loss=mse`, Dirichlet `α_peak=10`,
`α_base=0.1`), the γ-bucket diagnostics emitted by `_eqm_loss`
(`src/aitchinson_flow/models/eqm.py:298-304`) split as:

| epoch | g<.33 | g<.66  | g<1    | flow_loss |
|------:|------:|-------:|-------:|----------:|
| 1     | 11.37 | 0.768  | 1.355  | 6.076     |
| 2     |  7.71 | 0.155  | 0.180  | 3.726     |
| 3     |  7.69 | 0.089  | 0.119  | 3.740     |
| 4     |  7.18 | 0.062  | 0.090  | 3.449     |
| 5     |  6.82 | 0.053  | 0.080  | 3.303     |

`g<.33` (low γ, x_γ ≈ noise) plateaus high (~6.8); `g<.66` and `g<1`
collapse near zero. Same pattern in `comp_mse_seed43` (epoch 1: g<.33=10.86,
g<.66=0.758, g<1=1.404 → epoch 2: 7.44 / 0.162 / 0.191).

The **first-impression read** would be that the model "found a trivial
solution" in the high-γ buckets and "failed to learn" in g<.33. That read
is wrong — both effects are predicted by the proposal's math. But the
bucket diagnostic, as printed, is misleading on its own. Below: why, and
what to do about it.

## 2. Root cause — c(γ)·(x₀−x₁) target shape

`_eqm_loss` regresses the conservative gradient `∇⟨x_γ, f(x_γ)⟩` against

```
u_tgt = c(γ) · (x₀ − x₁)
```

with the default schedule `c(γ) = 1 − γ`
(`Config.eqm.decay_strategy = "linear"`, `eqm.py:308-314`).

`PROPOSAL_COMPOSITIONAL_EQM.md` Appendix A makes the irreducible per-γ
variance floor explicit:

```
floor(γ) = c(γ)² · [σ_source²·K + Tr(Cov(x₁))]
```

For Dirichlet CLR with `α_peak=10`, `α_base=0.1`, `K=27`:
`Tr(Cov(CLR)) ≈ 2525`. Therefore:

| γ regime | c(γ)  | u_tgt magnitude | floor (raw) | per-element after L=40·D=27 averaging |
|---------:|------:|:----------------|------------:|--------------------------------------:|
| γ → 0    | 1.0   | full            | 2525        | ≈ 2.3                                 |
| γ = 0.5  | 0.5   | half            |  631        | ≈ 0.6                                 |
| γ → 1    | 0.0   | zero            |    0        | 0                                     |

The bucket-averaged numbers we see (g<.33 ≈ 7, g<.66 ≈ 0.05, g<1 ≈ 0.08)
match the *shape* of the floor curve exactly — not the absolute scale,
because (a) `loss_fn` aggregates over batch×length×dim with conventions
that differ from the raw trace, (b) the Dirichlet-induced source noise
`σ²_source·K` is negligible vs Cov(x₁), and (c) γ-importance sampling
(`gamma_power=1.5` by default for MSE cells) skews the in-bucket γ
distribution toward bucket boundaries.

Two separate mechanisms produce the asymmetry:

### 2.1 High γ — target is mechanically zero

`c(γ→1) = 0` ⇒ `u_tgt → 0`. The MSE-optimal model is `f*(x_γ) =
E[u_tgt | x_γ]`, which at γ=1 is exactly 0. The model converges to
"output zero gradient at clean Dirichlet samples," and **that is the
intended equilibrium / EBM property** (proposal §3): clean data sit at
local minima of the implicit energy.

So `g<1 → 0` is not a trivial-solution pathology; it is the design.
What looks like "found a trivial solution fast" is the schedule
trivialising the target faster than the model needs to learn anything.

### 2.2 Low γ — irreducible variance floor

At γ→0, `x_γ ≈ x₀` is pure Gaussian noise. The model only sees
`(x_γ, γ)`; it cannot recover `x₁` from `x_γ` alone. The best it can do
is predict `E[u_tgt | x_γ ≈ noise] = c(γ)·(x₀ − E[x₁])`, where `E[x₁]`
is the corpus marginal of the smoothed CLR. The residual variance —
`c(γ)²·Tr(Cov(x₁))` — is the proposal's central design feature: it
**forbids** the model from collapsing flow_loss to zero everywhere
(which deterministic CLR allows, producing spike basins, see `SESSION_SUMMARY.md`).

So `g<.33 ≈ 6.8` is not a learning failure; it is the per-token Dirichlet
variance manifesting as the predicted variance floor at the γ regime
where it cannot be averaged away.

### 2.3 Why the user's intuition is half-right

> "g<.33 at least did not learn anything arbitrary."

Yes — g<.33 carries real signal (the conditional-mean component of
`u_tgt`), and the residual is genuinely irreducible. But the
*magnitude* of the loss in that bucket is dominated by the irreducible
floor, not by what the model has learned. **You cannot read training
progress in g<.33 as "is the model still off."** You have to look at:

* **Δ(g<.33) across epochs** (epoch 1: 11.37 → epoch 5: 6.82 — the
  model is reducing the bias component of the loss) — relative reduction
  ≈ 40 %, consistent with closing the gap to the variance floor.
* **Recovery accuracy at small α** (the headline metric in the runbook).
  Bucket loss does not directly answer "does the field steer perturbed
  samples back to attractors?"

## 3. Implications for the runbook's interpretation

* For **Compositional MSE** cells, `g<.33` plateauing in the 6–8 range
  and `g<.66`, `g<1` collapsing to <0.2 is the predicted shape and is
  not a bug.
* For **deterministic-CLR reference** cells (`comp_ref_det_mse`,
  `comp_ref_det_hilbert`), expect *all three buckets* to drop near zero
  within ~2 epochs (Tr(Cov(x₁)) = 0 ⇒ floor(γ) = 0 everywhere). If they
  don't, that's a bug in the reference. If they do (the predicted
  outcome), the *training* side of the proposal is validated: Dirichlet
  thickening prevents the deterministic-CLR collapse-to-zero pathology.
* The headline test of the proposal is **recovery@α=0.50**, not
  flow_loss. A cell can have higher per-bucket flow_loss and *better*
  recovery — that is the spike-basin failure mode the proposal is
  designed to avoid.

## 4. Possible remedies (if a more uniform per-bucket diagnostic is
desired)

These are ordered roughly by intrusiveness; only (1)–(2) are pure
diagnostics that don't change training dynamics.

1. **Normalise the bucket diagnostic by `c(γ)²`** before printing.
   `out["g<.33"]` etc. would then report a relative MSE
   `‖grad_g − u_tgt‖² / ‖u_tgt‖²`, which removes the schedule-induced
   asymmetry. This is a 5-line change to the diagnostic block in
   `_eqm_loss` (`eqm.py:297-304`); it changes nothing about the
   gradient path.

2. **Print a "floor-relative" gap**: `bucket_loss − floor(γ)` using the
   closed-form floor from Appendix A with the configured Dirichlet
   parameters. Same diagnostic improvement as (1), and makes "is the
   model at the floor or above it?" obvious.

3. **Hilbert seminorm in place of MSE** (`loss.mode = "hilbert_soft"`,
   already on the sweep matrix as `comp_hilbert_seed{42,43,44}`). The
   Hilbert seminorm is invariant under additive shifts in CLR (it lives
   on the (K−1)-dim simplex tangent), so it ignores the all-ones
   direction. The trace of the simplex-tangent covariance is `(K−1)/K`
   times the CLR trace ⇒ floor reduces by ~4 %. The proposal's §3.1
   predicts this gives a small recovery-band broadening (recovery
   active at α=0.35 with Hilbert vs. only α≥0.50 with MSE). The full
   sweep tests this directly.

4. **`gamma_power` < 1** (push samples toward γ=1). Cosmetically lowers
   the bucket-averaged g<.33 by sampling fewer of those γ values — but
   the per-γ floor is unchanged and the model sees less of the
   variance-floor regime, which is the regime that prevents basin
   collapse. Hides the diagnostic; doesn't fix what it's revealing.
   Worth testing only if recovery@small-α is unaffected.

5. **`gamma_power` > 1.5** (more samples at low γ). Forces more capacity
   onto the variance-floor regime. May tighten basins (more training
   gradient where the field has slope) at constant compute. Currently
   MSE cells use the config default `gamma_power=1.5` and Hilbert cells
   set `gamma_power=1.0` — a small grid `{1.0, 1.5, 2.0}` would
   disambiguate which knob does what.

6. **Affine c(γ) = (1−γ) + ε** instead of linear. Avoids the trivial
   `u_tgt ≡ 0` at γ=1 by lower-bounding the target magnitude. Trades
   the "zero gradient at the data manifold" property — which is what
   makes the EBM interpretation clean — for a more uniform bucket
   diagnostic. Probably bad; the design wants `c(1)=0`.

7. **Decrease the Dirichlet variance** (increase `α_peak`). Reduces
   the floor, makes g<.33 fall faster. The Phase 0 sweep
   (`runs/dphase0_dirichlet_*` heritage; reproducible via
   `scripts/check_dirichlet_data.py`) shows `α_peak=200` cuts
   `Tr(Cov(x₁))` by ~25 % vs. `α_peak=10`. Recovers more of the
   deterministic-CLR regime — and reintroduces the spike-basin failure
   mode in proportion. Wrong direction for the proposal.

8. **Conditional context (pass token_ids as `h_ctx`)**. Lets the model
   see `x₁` via context, killing the variance floor outright at low γ.
   Turns the model into a conditional denoiser; loses unconditional
   generation. Fundamentally different model — out of scope for this
   ablation but a real architecture lever if the task changes.

## 5. Recommendation for the current sweep

* **Do not adjust the training recipe based on the bucket diagnostic
  alone.** Wait for the recovery@α numbers (step 3 of the runbook). The
  bucket asymmetry is predicted; the recovery metric is what
  distinguishes "variance floor doing its intended job" from "model
  failing to learn the gradient."
* When tabulating (step 4), the relevant comparisons are:
  - g<.33 plateau **for the Dirichlet cells** vs **for the
    deterministic refs**. The refs should drop near zero; the Dirichlet
    cells should hold near 6–8.
  - **Δ@.50 sign**: positive for compositional cells, ≈0 for
    deterministic refs.
  - **Hilbert vs. MSE Δ@.50** with seed std. Proposal §5.3 expects ~2–3 %
    in Hilbert's favour.
* If, after the recovery numbers come in, the conclusion is that
  Compositional MSE *also* underperforms on recovery (not just on the
  bucket diagnostic), the most informative follow-up is the
  `gamma_power` grid `{1.0, 1.5, 2.0}` at fixed loss, not a c(γ)
  schedule change.

## 6. The "non-trivial but bad" expectation

A natural follow-up question once you accept that the high-γ bucket is
trivially zero by schedule and the low-γ bucket sits at a true variance
floor: **does the sampler then produce non-trivial output?**

There are two distinct "trivial" failure modes to distinguish:

* **Mode collapse / no-op sampling** — the field is too peaked or too
  flat for descent to do useful work; perturbed inputs stay perturbed,
  and unconditional samples either repeat one token or hit one corpus
  prototype. Diagnosed by Δ@α (= clean recovery − perturbed-baseline
  recovery) ≈ 0, plus very low entropy `H_gen`.
* **Spike-basin overfit** — the `comp_ref_det_*` failure mode the
  proposal targets. Deterministic CLR has Tr(Cov(x₁)) = 0 ⇒ floor(γ) = 0
  *everywhere*, so MSE drives the field into a tiny degenerate energy
  bowl around each training prototype. Unigram KL is *low* (model fits
  the corpus) but the basins are too narrow to denoise from any real
  perturbation; Δ@.50 ≈ 0.

What the Compositional recipe targets is the in-between regime: **the
sampler should produce non-trivial output that is bad at higher-order
structure but recognisably text-like at the unigram level**, with
Δ@.50 strictly positive. The first three MSE seeds already show this
shape:

| metric                  | observed (3 MSE seeds) | trivial-collapse signature | spike-basin signature |
|-------------------------|------------------------|----------------------------|-----------------------|
| `KL_uni`                | ≈ 0.66                 | ≫ 1 (one token dominates)  | ≈ 0.05                |
| `KL_bi`                 | ≈ 5.9                  | ≈ 0 (one bigram dominates) | ≈ 2.0                 |
| `H_ratio = H_gen/H_gt`  | ≈ 1.16                 | ≪ 1                        | ≈ 1                   |
| sample look             | character soup, full alphabet | single repeated token | training-set prototypes |
| Δ@.50 (predicted)       | > 0                    | ≈ 0                        | ≈ 0                   |

So the bucket diagnostic is loud about the *training-side* variance
floor, but the *sampling-side* observable already shows the Compositional
recipe is not trivial: full alphabet coverage, slightly-over-corpus
entropy, and unigram statistics within the right ballpark. The bigram
KL is high — confirming "bad but not trivial." This is the hypothesis
the recovery-α sweep is set up to verify quantitatively.

The reference cells (`comp_ref_det_mse`, `comp_ref_det_hilbert`, last
two cells in the sweep) are the contrast. Per the runbook's §7 table
they are predicted to land at `KL_uni ≈ 0.05`, `KL_bi ≈ 2.0`,
`acc@.50 ≈ 1.00`, `Δ@.50 ≈ 0.00` — i.e. KL looks *better* than the
Compositional cells (overfitting helps the corpus-comparison
metrics), but `Δ@.50 ≈ 0` means descent is a no-op. **That row is the
proposal's claim.** If we observe that pattern in the runbook output,
the bucket-asymmetry observation in §1–2 above is closed: it is the
mechanism by which the trained field avoids the spike-basin sink.

## 7. Auxiliary-loss remedy — supplying a clean signal where the FM target is noise-dominated

A natural fix to the asymmetry described in §2: keep the variance floor
(it is what prevents spike-basin collapse), but supplement the FM
regression with a **clean** auxiliary loss in the low-γ regime, where
the FM target is dominated by irreducible Dirichlet noise. The
auxiliary loss should measure something the model *can* improve
indefinitely (token accuracy / cross-entropy / KL), while the FM loss
keeps training the gradient field on the simplex.

The codebase already has this scaffolding (`models/eqm.py:230-293`):

```
pred_x1 = x_γ − λ·grad_g                        # "implied-x1" reconstruction
log_probs = log_softmax(pred_x1, dim=-1)        # treat as categorical logits
ce  = NLL(log_probs, token_ids)                 # token CE
bg  = NLL(bigram_log_p, bigram_ids)             # bigram CE (currently off)
tg  = NLL(trigram_log_p, trigram_ids)           # trigram CE (currently off)
total_loss = flow_loss + λ_ce·ce + λ_bg·bg + λ_tg·tg
```

But — and this is the load-bearing detail — they are **only applied
where γ ≥ `ce_min_gamma` = 0.5**. The mask in
`models/eqm.py:240` reads:

```python
ce_mask = gamma >= s.ce_min_gamma   # default 0.5
```

So in the *current* sweep:

| γ regime  | FM signal               | CE signal       |
|----------:|:------------------------|:----------------|
| γ < 0.5   | noise-dominated (g<.33 ≈ 7) | **none**     |
| γ ≥ 0.5   | trivially small (g<1 ≈ 0.05) | full λ_ce=0.5 |

This is exactly backwards relative to the proposal in §7's first
sentence: CE pressure is applied where the FM signal is already
trivial, and *not* applied where the FM signal is buried in noise.

### 7.1 Concrete options

(A) **Lower `ce_min_gamma`** to 0.0 (or to 0.05–0.1 to avoid pure-noise
inputs). Existing CE pathway extends to all γ. **This is a one-config-
knob change.**

What this provides at each γ regime, with `pred_x1 = x_γ − λ·grad_g`
as the categorical logits:
* γ ≈ 1: x_γ already encodes the token; CE just sharpens the basin.
  (current behaviour)
* γ ≈ 0.3: x_γ has roughly equal contributions from x_0 and x_1; the
  token IS recoverable from x_γ. CE here trains a genuine
  discriminative gradient at the regime where g<.33 currently
  plateaus at the variance floor.
* γ → 0: x_γ ≈ noise; the only target reachable from "no information
  about which token" is the corpus unigram marginal. CE there pushes
  the implied-x1 reconstruction toward `log p(token)` (the marginal),
  which is a useful prior, not a per-sample target. Worth verifying
  that this doesn't fight the FM regression — the two have the same
  fixed point at the marginal.

Risk: at γ → 0, `pred_x1 = x_γ − λ·grad_g ≈ −λ·grad_g`. CE on this
forces `grad_g` to carry the token logits. That is *exactly* the same
direction the FM loss is pushing (E[u_tgt | x_γ] ≈ −c(γ)·E[x_1 | x_γ]),
so they should reinforce, not conflict.

(B) **γ-attenuated `λ_ce(γ)` schedule** instead of a hard mask. e.g.
`λ_ce(γ) = λ_max · clip(γ/γ₀, 0, 1)^p` with `γ₀=0.3, p=1`. This is
the user's framing: "attenuate it with a lambda factor". Smoother
than the hard mask, and exposes a per-γ knob. One loss-function
edit (≤10 lines) in `_eqm_loss`. Would need a small grid (`p ∈
{0.5, 1.0, 2.0}`) to find the right roll-off.

(C) **Discriminative classifier head** (separate from velocity head).
Take the encoder hidden `h_enc` (already produced when `bigram_head`
is active, `models/eqm.py:213`), pass through a small linear → K
classifier, CE against `token_ids` at *all* γ without going through
`pred_x1`. Gradient flows back through the backbone but bypasses
`grad_g`. Provides a "the encoder must encode token-discriminative
features at every γ" pressure. More invasive (new module, new config
field) but cleanly decouples the CE pathway from the FM target's
noise floor.

(D) **Bigram CE** (`λ_bg > 0`, currently 0). Already implemented
(`models/eqm.py:272-279`). Attacks the *other* visible failure of
the trained Compositional MSE samples — `KL_bi ≈ 5.9`, essentially
random adjacency. Not directly addressing the variance floor but
arguably the more impactful follow-up because the variance floor is
working as intended (samples have correct unigram statistics) while
bigram structure is genuinely missing. The sweep YAML does not
exercise this knob; a 4-cell ablation `λ_bg ∈ {0, 0.1, 0.5, 1.0}` at
fixed `loss=mse, ce_min_gamma=0.5` would isolate it.

(E) **EBM-style ELBO on the discrete tokens**. Closed-form
`log p_γ(token_k | x_γ) ∝ −E(x_γ + ε·CLR(token_k))` with K=27 forward
passes per training step. Train via NLL of true token. Most
principled (it is literally the energy-based likelihood the recipe
implies), but the most expensive (27× forward passes per step). Worth
considering only if (A)/(B)/(D) under-perform.

### 7.2 Suggested order of trials

1. **(A) Drop `ce_min_gamma` to 0.0**, keep `λ_ce = 0.5`. One knob,
   re-run a single Compositional MSE seed at the same budget. If
   `g<.33` drops materially and `Δ@.50` improves, the variance-floor
   bucket *was* training-starved.
2. **(B) Smooth γ-schedule** if (A) destabilises early training (e.g.
   if forcing CE at γ=0.05 produces unbounded `pred_x1` magnitudes
   from undertrained `grad_g`).
3. **(D) Bigram CE** independently of (A)/(B), targeting the
   `KL_bi ≈ 5.9` failure visible in samples, not the bucket
   asymmetry.

(C) and (E) are larger architecture changes; defer until (A), (B), (D)
data is in.

### 7.3 What to check on the runs

For each variant, the diagnostic to read:
* `g<.33` trajectory across epochs — did it drop below the predicted
  variance floor? (If so the model is now compressing the
  variance-floor regime via the CE pathway.)
* `Δ@.50` from `recovery_check.py` — the sampler-side observable. The
  whole point of CE there is to make the field steer perturbed
  inputs back to the right token, not just any text-like state.
* `KL_bi` from `eval.json` — sample bigram structure, the second
  observable failure of the current setup.

A successful (A) should show: `g<.33` ↓, `Δ@.50` ↑, `KL_bi` ≈
unchanged. A successful (D) should show: `g<.33` ≈ unchanged,
`KL_bi` ↓, `Δ@.50` mildly ↑.

## 8. Non-linear paths — Dirichlet-marginal interpolants

The current EqM uses the linear CLR-space interpolant
(`models/eqm.py:201`):

```
x_γ = (1−γ)·x_0 + γ·x_1,   x_0 ~ σ_source·N(0, I)|_{V_d},
                            x_1 = CLR(Dir(α_t))
```

This is the FM "linear path" / "rectified-flow" choice. It has two
geometric weaknesses for simplex data:

* **Off-manifold for γ ∈ (0, 1)**: x_γ is a Gaussian-mixed-with-CLR
  point in R^K with sum-zero. It is *not* the CLR of any probability
  vector for intermediate γ (because the Gaussian noise breaks the
  log-ratio structure). The model is asked to learn velocities at
  off-simplex points it will never see during sampling.
* **Forces the variance-floor / target-collapse asymmetry of §2**: u_tgt
  scales with c(γ), which the linear path requires to vanish at γ=1
  for the marginals to be correct.

Both can be addressed by replacing the linear path with a path whose
intermediate marginals stay on the simplex. The user's framing — "learn
a chain of Dirichlets approximating x_1" — is exactly the Dirichlet
Flow Matching (DFM-Stark) family.

### 8.1 Existing literature (concrete pointers)

* **Stochastic Interpolants** (Albergo, Boffi, Vanden-Eijnden 2023,
  arxiv 2303.08797). Generalises x_γ = α(γ)·x_0 + β(γ)·x_1 + σ(γ)·z
  with designable (α, β, σ) subject only to the endpoint conditions
  α(0)=β(1)=1 and α(1)=β(0)=0. The current code is the special case
  α(γ)=1−γ, β(γ)=γ, σ(γ)=0 (rectified flow). The paper's optimal
  drift `b(x_γ, γ) = E[α'(γ)·x_0 + β'(γ)·x_1 | x_γ]` is the FM target
  in our notation; choosing (α', β') that don't both vanish at γ=1
  directly removes the c(γ)→0 trivialisation of §2.1. The paper is
  path-agnostic about the data manifold — for the on-simplex
  property, must combine it with a simplex-respecting ρ_0 (e.g.
  uniform Dirichlet) as in §8.3 below. Cheap follow-up cell:
  cosine schedule (α(γ)=cos(πγ/2), β(γ)=sin(πγ/2)) under the same
  conservative-gradient parametrisation. ≤30 LOC in `_eqm_loss` plus
  a new `path_mode` config field.
* **Dirichlet Flow Matching** (Stark, Jing, Wang, et al., NeurIPS
  2024). Trains FM on the simplex via a conditional probability path
  p_t = Dir(α(t)·e_token + ε·𝟙) interpolating from a near-uniform
  Dirichlet at t=0 to a sharply peaked Dirichlet at t=1. Velocity field
  pushes samples toward the data; sampling stays on the simplex
  throughout.
* **Rectified Flow / Reflow** (Liu, Gong, Liu 2022). After a first FM
  training, "straighten" the learned trajectories by re-pairing
  (x_0, x_1) along the model's own samples. Reduces curvature, allows
  fewer NFE at sampling. Not directly Dirichlet, but a path-design
  technique that composes with any of the above.

### 8.2 What this codebase already has

The `DFM` baseline (`src/aitchinson_flow/models/`, registered via
`build_model`) implements **Gat et al. 2024 Discrete Flow Matching**:

```
p_t = κ_t · δ(x_1) + (1 − κ_t) · U(K)
```

This is a degenerate Dirichlet-style path: a *mixture* of point-mass
on the data and uniform-over-K. It is on-simplex by construction (a
simplex point is a convex combination of simplex points) and CE on the
denoiser is the natural loss because the marginals are explicit.

So one valid non-linear-path follow-up is to **directly compare the
EqM-on-CLR (linear path) against DFM (mixture path) on the same
recovery metric**. This sweep does not include DFM — adding 1–3 DFM
cells with matched seeds would pin down whether the variance-floor
asymmetry is specific to the linear-CLR interpolant or generic to the
text8 simplex problem.

### 8.3 A "smooth Dirichlet path" as a concrete experiment

Halfway between the current EqM-linear and DFM-mixture: keep the
EqM/conservative-gradient parametrisation, but specify the marginal
along the path as a **smoothly varying Dirichlet**. Concretely:

```
α_γ = (1 − γ)·α_uniform + γ·α_peaked
       (α_uniform  ≈ α_base · 𝟙,   α_peaked = α_base·𝟙 + α_peak·e_token)
p_γ ~ Dir(α_γ),                          x_γ = CLR(p_γ)
```

At γ=0: p_γ ~ Dir(α_base·𝟙) — a near-uniform Dirichlet over the
simplex, which acts as a non-Gaussian "noise" source that *lives on
the simplex*.
At γ=1: p_γ ~ Dir(α_base·𝟙 + α_peak·e_token) — the peaked Dirichlet
at the data point (same as the current x_1).

Properties:
* **All x_γ are CLRs of valid probability vectors** — model never
  trains on off-simplex points.
* **No c(γ)→0 trivialisation needed**: u_tgt = ∂_γ x_γ stays of order
  ‖CLR(α_peaked) − CLR(α_uniform)‖ throughout, modulated by the
  α-schedule's derivative. The bucket asymmetry of §2 should largely
  vanish.
* **Variance floor still present** (good — it is the load-bearing
  property): at any γ < 1, Dir(α_γ) has nonzero spread; per-token
  Cov(x_γ) > 0. So spike-basin collapse is still prevented.
* **Costs**: requires sampling Dir(α_γ) per training step (already
  cheap in PyTorch via `torch._standard_gamma`), and computing u_tgt
  via the chain rule on `CLR(Dir(α_γ))`. The latter is non-trivial
  closed form but can be done by autograd through the reparametrised
  Dirichlet sample (the "implicit reparametrisation" used in
  CHALAlgorithms or by stop-grad / score-function tricks). Adds
  perhaps 100–200 LOC and one new `Config.transformation` mode.

### 8.4 A "chain of Dirichlets" — the discretised version

The continuous Dirichlet-marginal path above is the cleanest
formulation, but the user's phrasing suggests a **discretised**
version: T fixed snapshots `Dir(α_0), Dir(α_1), …, Dir(α_T)` with the
model trained to predict transitions `p_{γ_k} → p_{γ_{k+1}}`.

This is essentially a **Markov chain on simplex marginals** with
learnable α_k. Two options:

* **Fixed schedule, learned transitions**: pick α_k by a known
  schedule (geometric, cosine, linear in entropy); the model learns
  only the velocity field. This is the discretised analogue of §8.3.
* **Learned schedule**: parameterise the α_k themselves and optimise
  via a path-length / variance-floor objective. Risk: collapses to
  trivial extremes (all α_k near α_uniform or all near α_peaked).
  Constraints (monotone entropy, fixed endpoints) keep it sane.

The chain-of-Dirichlets has a nice interpretation in light of §3.1 of
the proposal: each Dir(α_k) is a curvature-controlled Aitchison-ball
around the data; the model learns to descend through nested balls,
which is precisely what Hilbert-metric sampling does in the limit.
This is consistent with the existing finding that **Hilbert + Dirichlet
shows the most uniform per-bucket loss** (visible already in the first
Hilbert cell of the current sweep: g<.33=12.4 / g<.66=7.8 / g<1=7.3,
versus MSE's 11.0 / 0.78 / 1.4) — the Hilbert seminorm is the metric
that respects this nested-Dirichlet path geometry.

### 8.5 Stochastic interpolant — variance floor without Dirichlet

A natural follow-up to §8.1: the variance floor of §2.2 is the
load-bearing ingredient that prevents spike-basin collapse, but it does
*not* have to come from `Cov(x_1)`. The Stochastic Interpolants
framework gives an alternative source: an additive noise term σ(γ)·z
on the path itself.

```
x_γ = α(γ)·x_0 + β(γ)·x_1 + σ(γ)·z,    z ~ N(0, I) ⊥ (x_0, x_1)
```

with the same boundary conditions α(0)=β(1)=1, α(1)=β(0)=0, plus
σ(0)=σ(1)=0 so the endpoint marginals stay at ρ_0 and ρ_1. The drift
target picks up an additional `σ'(γ)·z` term, which is per-(x_γ, γ)
random — exactly the variance-floor mechanism from §2.2, but with the
randomness living in the path noise rather than in the data thickening.

**Implication for the one-hot / deterministic-x_1 question.** A
non-linear *deterministic* path (e.g. cosine schedule, σ ≡ 0) on
deterministic-CLR data has *zero* per-(x_γ, γ) target variance — same
spike-basin pathology as the linear path on deterministic-CLR. The
non-linearity does not by itself add variance. What adds variance is
the σ(γ)·z term. With σ(γ) > 0 in the interior of (0, 1), one can
recover the proposal's anti-spike-basin property *without* Dirichlet
thickening of x_1.

| recipe | x_1 stochasticity | path stochasticity | variance floor (γ ∈ (0,1)) |
|---|---|---|---|
| current Compositional (`comp_*`) | Dirichlet, large | none (σ=0) | c(γ)²·Tr(Cov(x_1)) ≈ peaked at γ=0 |
| det-ref (`comp_ref_det_*`) | none | none (σ=0) | 0 (spike basins) |
| cosine path, det-ref | none | none (σ=0) | 0 (still spike basins) |
| **stochastic-interpolant + det-ref** | **none** | **σ(γ)>0** | **σ'(γ)²·dim, shape designable** |

Properties of the stochastic-interpolant + det-ref recipe:

* x_1 is just label-smoothed CLR of the token — no Dirichlet sample
  per training step, no `α_peak`/`α_base` tuning.
* The variance-floor *shape* across γ is a free design choice (via
  σ(γ)). Common choices: σ(γ) = ε·sin(πγ) (peaked at γ=0.5, zero at
  endpoints), σ(γ) = ε·γ(1−γ) (similar), or learnable σ(γ).
* Compatible with the existing EqM conservative-gradient
  parametrisation; the `_eqm_loss` change is ~30 LOC (add z sampling
  and σ-dependent target term).

Trade-offs vs. Dirichlet thickening:

* **Loses the Aitchison-tangent specialisation** of §3.1 of the
  proposal. Dirichlet variance lives in the (K−1)-dim simplex
  tangent; isotropic Gaussian z does not. The Hilbert-metric
  advantage may shrink in this regime — Hilbert was tailored to a
  metric on the simplex tangent, while now half the variance is in
  the perpendicular all-ones direction. Could be partially recovered
  by projecting z onto V_d (subtracting its mean), preserving the
  zero-mean-CLR constraint.
* **Does not fix the high-γ "easy bucket"** — σ(1)=0 is still
  required, so the floor still vanishes at γ=1 just as c(γ=1)=0
  vanished. The high-γ trivialisation is *unavoidable*: at the data
  manifold itself, the field must have zero gradient (that is what
  "the data is at the equilibrium" means). The fix only redistributes
  the floor across the interior of (0, 1); it cannot eliminate it at
  the endpoint.

Suggested cell:

```yaml
- name: comp_stoch_interp_seed42
  overrides:
    training.model_name: EqM
    training.epochs: 5
    training.seed: 42
    text8_dataset.max_train_windows: 10000
    loss.mode: mse
    transformation.dirichlet_sampling: false   # deterministic CLR x_1
    transformation.path_mode: cosine_stochastic   # NEW config field
    transformation.sigma_schedule: "eps_sin_pi_gamma"
    transformation.sigma_max: 0.5
```

`path_mode = cosine_stochastic` would imply `α(γ) = cos(πγ/2)`,
`β(γ) = sin(πγ/2)`, and σ(γ) per `sigma_schedule`. This is a clean
test of the hypothesis: **does the proposal's anti-spike-basin
mechanism depend on Dirichlet thickening, or only on the existence of
*some* per-step variance floor at γ < 1?**

Predicted outcomes:

* If the recipe needs *Dirichlet* specifically (the Aitchison-tangent
  story matters): the stochastic-interpolant cell underperforms
  Compositional MSE on Δ@.50 despite having a non-zero variance floor.
* If the recipe needs only *some* variance floor: the
  stochastic-interpolant cell matches or beats Compositional MSE,
  with the bonus of a simpler data pipeline.

### 8.5.1 Implementation status (2026-05-10 session)

The §8.5 cell has been wired into the codebase as a follow-up
ablation to the runbook sweep. Changes (one commit's worth, no public
API):

* `src/aitchinson_flow/config.py` — `TransformationConfig` gains two
  fields, both default to no-op:
  - `sigma_interpolant_max: float = 0.0`
  - `sigma_interpolant_schedule: str = "sin"`
* `src/aitchinson_flow/models/eqm.py` — `_eqm_loss` adds an optional
  block (≈12 LOC) that, when `sigma_interpolant_max > 0`, samples
  `z ~ N(0,I)|_{V_d}` and updates
  `x_γ ← x_γ + σ(γ)·z`,  `u_tgt ← u_tgt + σ'(γ)·z`.
  Default behaviour (no σ) is unchanged.
* `sweeps/compositional_eqm_test.yaml` — 4 cells added at the end:
  `comp_stoch_smoke` (1-epoch wiring check) and `comp_stoch_seed{42,43,44}`
  (3-seed full-budget run at `sigma_interpolant_max=2.0`,
  `dirichlet_sampling=false`, `loss=mse`).

The orchestrator (`scripts/run_sweep.py`) reads the YAML once at
startup and is idempotent on `eval.json` presence, so re-launching the
sweep after the current run finishes will skip the existing 9 cells
and execute only the 4 new ones.

### 8.6 Suggested experiment ladder

If one wanted to test the path-design hypothesis without committing to
a full re-implementation:

1. **Add 1–3 DFM cells** to the sweep at matched seeds. Cheapest test:
   answers "is the asymmetry path-specific or problem-specific?"
2. **(§8.3) Implement the smooth-Dirichlet-marginal path** as a new
   `LossConfig.path_mode = "dirichlet"` (or `transformation.path_mode`)
   alongside the existing linear path. Reuse the existing EqM
   conservative-gradient parametrisation, just replace `x_γ` and
   `u_tgt` construction. Run at the same matrix as `comp_*` cells.
3. **(§8.4 fixed schedule)** if (2) shows a clear win, then ablate
   schedule choices (geometric vs. cosine vs. learned-α_k) as a
   follow-up.

Step 1 is low-effort (a few lines of YAML, no code). Step 2 is the
bulk of the work but is also the cleanest test of the hypothesis.
Step 3 is fine-tuning — defer until (1) and (2) data are in.

## 9. Anchor: where this analysis lives in the code

* `src/aitchinson_flow/models/eqm.py:184-306` — `_eqm_loss`, including
  the c(γ) target construction (line 206) and the bucket diagnostic
  (lines 297-304).
* `src/aitchinson_flow/models/eqm.py:308-345` — `_c_gamma`. The default
  schedule used by all sweep cells is `linear` ⇒ `c(γ) = 1 − γ`.
* `PROPOSAL_COMPOSITIONAL_EQM.md` §3, Appendix A — variance-floor
  derivation and the `floor(γ) = c(γ)²·Tr(Cov(x₁))` formula.
* `scripts/check_dirichlet_data.py` — Phase 0 numbers used above (run
  again with `--alpha-peak 10 --alpha-base 0.1` to reproduce the
  `Tr(Cov(x₁)) ≈ 2525` claim).
