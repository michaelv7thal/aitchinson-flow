# NAG-GD Sampler Diagnosis — text8 EqM (5-epoch baseline)

> Hypothesis under test: the current default sampler `(η=0.1, μ=0.9, 200 steps,
> return-final)` overshoots the energy minimum, so `model.sample()` returns an
> iterate well past the basin it found earlier in the run.
>
> Verdict: **confirmed and fixed.** Per-position gradient clipping at `‖·‖ ≤ 1`
> plus best-iterate tracking turns a 12× overshoot into a 2.6× overshoot,
> drops unigram-KL **0.073 → 0.049 (−33 %)** and bigram-KL **2.00 → 1.69
> (−16 %)** at the same compute (200 NAG steps, B=32).

## TL;DR

| Variant | g_min | g_final | overshoot | KL_uni | KL_bi | Δ KL_uni | Δ KL_bi |
|---------|------:|--------:|----------:|-------:|------:|---------:|--------:|
| **NAG η=0.1 μ=0.9 200 (current)**    | 0.26 |  3.03 | 2.95 | 0.0727 | 2.00 | — | — |
| NAG η=0.1 μ=0.9 200 +best            | 0.26 |  3.03 | 2.95 | 0.0667 | 1.81 | −8 % | −10 % |
| NAG η=0.05 μ=0.9 200 +best           | 0.23 |  0.79 | 0.57 | 0.0524 | 1.70 | −28 % | −15 % |
| NAG η=0.1 nesterov-μ 200 +best       | 0.23 |  3.14 | 3.10 | 0.0606 | 1.75 | −17 % | −12 % |
| NAG η=0.1 adaptive-restart 200 +best | 0.27 |  3.05 | 3.12 | 0.0671 | 1.80 | −8 % | −10 % |
| NAG η=0.1 η-decay τ=50 200 +best     | 0.26 |  1.35 | 1.17 | 0.0679 | 1.80 | −7 % | −10 % |
| **NAG η=0.1 μ=0.9 clip=1.0 200 +best** ★ | 0.19 |  0.50 | **0.31** | **0.0487** | **1.69** | **−33 %** | **−16 %** |
| plain GD η=0.05 500 +best            | 0.23 |  2.87 | 2.91 | 0.0483 | 1.70 | −34 % | −15 % |
| plain GD η=0.10 500                  | 0.26 |  2.96 | 3.61 | 0.0862 | 3.14 | +18 % | +57 % |

The headline finding is the gradient-norm trajectory of the current default:

```
step   0: ||grad|| = 4.61   (noise init)
step   1: ||grad|| = 9.68   ← NAG cold-start spike, gradient field is steep
step   2: ||grad|| = 9.51
step  10: ||grad|| = 4.67
step  20: ||grad|| = 1.81
step  50: ||grad|| = 0.32   ← reaches the basin
step 100: ||grad|| = 0.67   ← already climbing back
step 150: ||grad|| = 2.04
step 199: ||grad|| = 3.15   ← what model.sample() returns today
```

The model finds the basin at step 50–60 and then NAG carries the iterate back
out. The user's intuition that "NAG is overshooting" was right; the gradient
trajectory makes it visible.

## 1. Setup

- Checkpoint: `checkpoints/baseline_5ep/epoch_final.pt` — EqM, 5 epochs, the
  "all-fixes" config from `SESSION_SUMMARY.md` (conservative-grad sampling,
  matched train/sample σ=0.1, aux CE on implied-x1, γ-importance sampling).
- Diagnostics:
  - `scripts/legacy/diagnose_overshoot.py` — B=32, 7 sampler variants + grad-norm
    trajectory.
  - `scripts/legacy/diagnose_overshoot_v2.py` — adaptive restart, μ schedule,
    η decay, gradient clip.
- Reference corpus: `text8` train split (K=27, L=40 windows).

## 2. What we measured per config

- `g0` — initial mean per-position ‖∇E‖ at the noise init.
- `g_min` / `@step` — best mean ‖∇E‖ found during the run + the step it occurred.
- `g_final` — ‖∇E‖ at the final iterate (what `model.sample()` returns today).
- `overshoot = max(traj[g_min:]) − g_min` — by how much the energy gradient
  bounces back up after reaching its minimum. Headline overshoot signal.
- `H_uni`, `KL_u`, `KL_b` — argmax-decoded unigram entropy, unigram KL vs
  train, bigram KL vs train.
- All metrics computed off the *best iterate* (`+best`) when applicable, so
  the row reflects what a sampler-with-best-iterate-tracking would return.

## 3. Iteration 1 — vanilla NAG / plain GD sweep

```
config                                   g0   g_min  @step  g_final  overshoot  H_uni    KL_u    KL_b
NAG eta=0.1  mu=0.9 200 (current)     4.606  0.2590     57   3.0266    2.9541   2.676   0.0727  2.004
NAG eta=0.1  mu=0.9 200 +best         4.606  0.2590     57   3.0266    2.9541   2.637   0.0667  1.805
NAG eta=0.1  mu=0.5 200 +best         4.606  0.2442     51   3.1606    2.9605   2.648   0.0642  1.779
NAG eta=0.05 mu=0.9 200 +best         4.606  0.2334    113   0.7940    0.5742   2.699   0.0524  1.704
plain GD eta=0.10  200                4.606  0.2560     49   3.1349    2.9620   2.706   0.0702  1.898
plain GD eta=0.10  500                4.606  0.2560     49   2.9599    3.6149   2.875   0.0862  3.139  ← worst
plain GD eta=0.05  500                4.606  0.2277    102   2.8716    2.9086   2.835   0.0897  2.371
```

Reads:

1. **The overshoot is intrinsic to the energy field, not just to NAG momentum.**
   Even plain GD (μ=0) reaches `g_min ≈ 0.26` at step ~50 then climbs back to
   `g_final ≈ 3.0`. Running plain GD longer (200 → 500 steps) makes things
   *worse* (KL_b 1.90 → 3.14). Beyond ~100 steps the iterate drifts off the
   basin attractor along weak repulsive directions; this is consistent with a
   5-epoch undertrained model whose `⟨x, f(x)⟩` field has saddles in the
   places where it should have strict minima.

2. **Best-iterate tracking is a free win.** Returning the lowest-‖∇E‖ iterate
   instead of the trajectory's endpoint cuts unigram-KL 8 % and bigram-KL
   10 % at zero extra compute — the same forward passes, just one more
   `min`-clone.

3. **Lowering η to 0.05 keeps NAG inside the basin.** Overshoot drops 2.95 →
   0.57; KL_uni reaches 0.052 (already below the in-training probe value of
   0.065 reported at epoch 5).

4. **NAG cold-start spike.** Gradient norm doubles at step 1 (4.6 → 9.7).
   Lookahead on the very first step has no momentum (`x_prev = x`), so the
   spike is a pure consequence of `f(x)`'s steepness near the noise init —
   the field is locally steeper than `1/η`, so a single Euler step lands
   somewhere with a *larger* gradient. This made gradient clipping the most
   promising iter-2 candidate.

## 4. Iteration 2 — overshoot-targeted variants

```
config                                          g0   g_min  @step  g_final  overshoot restart  H_uni    KL_u    KL_b
NAG eta=0.1 mu=0.9 200 +best                  4.606  0.2590     57   3.0266    2.9541       0  2.637  0.0667  1.805
NAG eta=0.1 adaptive-restart 200              4.606  0.2703     55   3.0484    3.1165     137  2.635  0.0671  1.804
NAG nesterov-mu eta=0.1 200                   4.606  0.2349     55   3.1389    3.0989       0  2.669  0.0606  1.753
NAG eta_decay tau=50 mu=0.9 200               4.606  0.2639     67   1.3544    1.1737       0  2.633  0.0679  1.800
NAG eta=0.1 mu=0.9 clip=1.0 200    ★          1.000  0.1889    142   0.4960    0.3070       0  2.718  0.0487  1.692
adaptive-restart + eta_decay=50               4.606  0.2715     71   1.4085    1.2010     125  2.637  0.0667  1.802
plain GD eta=0.05 500                         4.606  0.2277    102   2.8716    2.9086       0  2.724  0.0483  1.700
```

### What worked, and what didn't

- **Gradient clipping (`‖∇E‖_pos ≤ 1.0`) is the cleanest fix.** It eliminates
  the cold-start spike directly (g0 = 1.00 instead of 4.61), keeps NAG inside
  the basin once it lands there (overshoot 0.31 vs 2.95), and reaches the
  *lowest* `g_min` of any iter-1+iter-2 method (0.189 vs 0.23–0.27). It also
  reaches that minimum *latest* (step 142 vs 50–60), meaning it's still
  productively descending where unclipped methods are bouncing. Best KL on
  both unigram and bigram axes.

- **Plain GD η=0.05 × 500 steps + best matches the clipped NAG on KL** but
  takes 2.5× more compute (500 vs 200 steps). The KL parity comes from
  picking out the basin iterate around step 100 — without `+best` the
  trajectory drifts to `g_final = 2.87`. Verdict: gradient clip is strictly
  cheaper for the same outcome.

- **Adaptive restart (O'Donoghue & Candès) doesn't help.** It triggers on
  137 / 200 steps — i.e. the lookahead direction and the gradient
  systematically conflict. With this much disagreement, restart degenerates
  to plain GD; KL is unchanged from the +best baseline. Strong signal that
  the field has weak/inconsistent descent direction across many positions.

- **Nesterov classic μ schedule (μ_t = t/(t+3))** modestly helps unigram
  (0.061) by softening the cold-start ramp, but doesn't fix overshoot
  downstream. Worth bundling with grad-clip if any further hardening is
  needed; not worth standalone.

- **Step-size decay (η_t = η_0 / √(1 + t/50))** reduces overshoot 3.0 → 1.2
  but doesn't improve KL because the iterate at the minimum is still found
  around step 60 — and `+best` already returns that one regardless of what
  η does later. Verdict: redundant once `+best` is in place.

- **`adaptive-restart + η-decay` combo doesn't beat either component.**

## 5. Recommended sampler defaults

Concretely change `EqM.sample` in `src/aitchinson_flow/models/eqm.py` to:

1. **Track the lowest mean-‖∇E‖ iterate and return that** when
   `return_best=True` (default).
2. **Per-position L2-clip the gradient at `clip_norm=1.0`** before each NAG
   update.
3. Keep `η = 0.1`, `μ = 0.9`, `max_steps = 200` (no need to lower η — the
   clip handles the cold-start, and 200 steps is plenty once the basin is
   reached around step 140).

And in `EqM` config:

```python
sample_eta:        0.1     # unchanged (gradient clip handles the steep field)
sample_mu:         0.9     # unchanged
sample_max_steps:  200     # was 500 — empirically no benefit beyond 150 once clipped
sample_grad_clip:  1.0     # NEW — per-position L2 cap on ∇E
sample_return_best: True   # NEW — return the lowest-‖∇E‖ iterate
```

Expected effect on the in-training unigram-KL probe: **0.065 → ~0.048**
(B=64, same compute). The headline number on `evaluate_eqm.py` panel C and
on `scripts/quick_eval.py` should drop similarly.

### Validation — `scripts/quick_eval.py` after the patch

After applying the recommended changes (`grad_clip=1.0`, `return_best=True`,
`max_steps=200`) to `EqM.sample`, running `scripts/quick_eval.py` end-to-end at
B=64 / 200 steps reproduces the diagnostic-predicted improvement:

```
=== Argmax decode (64 samples) ===
unigram_kl = 0.0530       (was 0.073 — −27 %)
H_gen      = 2.712         (was 2.677 — 95.2 % of corpus entropy)
H_ref      = 2.848
bigram_kl  = 1.6474        (was 2.00  — −18 %)

=== Gradient norm (lower = on energy minimum) ===
||grad|| at GT  : 0.1776   ← reference: where the data sits in the energy field
||grad|| at gen : 0.1915   ← samples now sit ≈ as close to a minimum as GT
```

The `‖grad‖ at gen ≈ ‖grad‖ at GT` parity is the cleanest before/after
signal: under the old sampler the generated samples sat at `‖grad‖ ≈ 3.0`
(13× larger than GT); after the patch they sit at `‖grad‖ ≈ 0.19` (within
8 % of GT). The sampler now reliably lands on energy minima.

Argmax-decoded samples are also visibly more diverse — no longer the same
template repeated across the batch:

```
'h hve iuneosfett ooriruwttiseen  aigmie '
'nn   rigda tciotieuosvintt latsw hi wapt'
'rrhct idti etorairs neeh tos ar  srgeur '
'e oeetitdei aethtcvsi   tt rarih te wbbs'
```

Still not English (that's an upstream training-budget issue, not a sampler
issue — see §6), but per-position character distributions now look like
text8 across batch positions, not template-locked.

## 6. What this does *not* fix

- **Character-soup quality.** Even the best iter-2 sampler still produces
  `' ntresit u twtuo   snlpaetepan'`. Sampler fixes can only make us land
  *on* per-position attractors; bigram-level coherence requires either more
  training, an n-gram-aware loss, or time-conditioning so the model knows
  where it is in the flow (`SESSION_SUMMARY.md` §5.5).
- **Reconstruction-BPD tautology** documented in §5.3 of `SESSION_SUMMARY.md`.
  Independent of sampling.

## 7. Open questions for the next training run

- **Why does plain GD overshoot at all?** The conservative-grad field
  *should* be a descent direction near a minimum; the climb after step ~50
  suggests the model's `⟨x, f(x)⟩` at 5 epochs has weak repulsive directions
  even inside the basins. Worth re-running this diagnostic on a 25-epoch
  checkpoint and checking whether plain GD becomes monotone — if so, the
  energy field has stiffened enough that gradient clipping becomes redundant.

- **Train-time gradient clip?** The training step does
  `optimizer.zero_grad(); loss.backward(); clip_grad_norm_(model.params(),
  1.0); step()`. That clips the *parameter* gradient, not the energy
  gradient `∇_x ⟨x, f(x)⟩`. A well-calibrated energy field at sample time
  shouldn't need additional clipping; the fact that we benefit from one
  hints at training-side smoothness regularisation (curvature penalty,
  spectral norm of the velocity head) being a useful next experiment.

- **B=32 → B=256 sanity.** All numbers above are from B=32 because of
  iteration speed on the diagnostic GPU. The in-training probe runs at
  B=64. A one-shot B=256 run with the recommended sampler (clip=1.0,
  +best) would tighten the KL estimate to a publication-ready precision.

## 8. How to reproduce

```bash
# Iter 1 — sweep + grad-norm trajectory
python -u scripts/legacy/diagnose_overshoot.py    checkpoints/baseline_5ep/epoch_final.pt
# Iter 2 — overshoot-targeted variants
python -u scripts/legacy/diagnose_overshoot_v2.py checkpoints/baseline_5ep/epoch_final.pt
```

Each run is ≈ 4 min on a single ~8 GB consumer GPU (RTX PRO 1000 Black).
