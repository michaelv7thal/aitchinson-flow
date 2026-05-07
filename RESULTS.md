# EqM vs DFM on text8 — Capstone results

A synthesis of the sweeps run for the [TRAINING_PLAN](TRAINING_PLAN.md). Focuses on
*why* each lever moved or didn't move bigram coherence — the per-run hypothesis
notes live in [`runs/DECISION_LOG.md`](runs/DECISION_LOG.md).

## TL;DR

- **Goal**: EqM (continuous flow-matching on the simplex with conservative-gradient
  sampling) on text8 (K=27, L=40), reaching plan target **KL_bi ≤ 0.50** without
  changing the conservative-grad framework.
- **Best EqM result**: KL_bi = **1.38** at parity compute (`data_50k_ep5`). 2.8× over
  target.
- **DFM control at parity compute**: KL_bi = **0.148** — clears the target, samples
  read as English-ish ("ine al cane tere", "two thowereh", "be thelaclad lip"). DFM
  beats EqM by **9.4×** on bigram KL at identical backbone, data, and epochs.
- **Strongest single lever for EqM**: data variety > epoch scaling.
  At parity compute, 5× more unique windows ⇒ −16 % bigram KL, while 5× more
  epochs on 10k windows ⇒ only −17 % bigram KL.
- **Three clean architectural negatives**: factorised bigram NLL, γ-conditioning
  (add and concat), and the plan's recommended γ=1 sample-time fallback.
- **Where the bottleneck actually lives**: EqM's sampler is a fixed-γ energy
  descent; DFM's sampler is an Euler integrator over t. The two architectures use
  time conditioning for fundamentally different reasons. EqM cannot benefit from
  γ conditioning the way DFM does without rebuilding the sampler.

## What was run

Eleven training runs across six phases. Each row is one full training cycle plus
the canonical scorecard from [`scripts/eval_full.py`](scripts/eval_full.py)
(256 samples × 200 NAG steps; bigram and trigram KL added vs the original tooling).

| Run | Phase | Cfg | KL_uni | KL_bi | KL_tri | H_ratio | \|∇E\|gen / \|∇E\|gt |
|---|---|---|---:|---:|---:|---:|---:|
| baseline_5ep | 0 | 5 ep × 10k windows (default) | 0.051 | 1.99 | 7.22 | 0.96 | 0.227 / 0.179 |
| ep10_default | 1 | 10 ep × 10k | 0.033 | 1.78 | 6.76 | 0.95 | 0.135 / 0.129 |
| ep25_default | 1 | 25 ep × 10k | 0.015 | 1.65 | 7.21 | 1.00 | 0.056 / 0.089 |
| **data_50k_ep5** ★ | 3 | 5 ep × 50k | 0.035 | **1.38** | **5.96** | 0.99 | 0.224 / 0.103 |
| data_50k_ep10 | 3 | 10 ep × 50k | 0.023 | 1.56 | 6.05 | 1.00 | 0.125 / 0.093 |
| data_200k_ep2 | 3 | 2 ep × 200k | 0.207 | 2.48 | 6.08 | 0.93 | 0.674 / 0.319 |
| ng_bg05_data50k | 5 | + λ_bigram=0.5 | 0.059 | 1.77 | 5.89 | 0.95 | 0.404 / 0.241 |
| ng_bg10_data50k | 5 | + λ_bigram=1.0 | 0.127 | 1.80 | 6.11 | 0.92 | 0.493 / 0.262 |
| tc_add_data50k γ=0.7 | 4 | + γ-add (best γ) | 0.838 | 3.35 | 7.55 | 0.74 | n/a |
| tc_concat_data50k γ=0.5 | 4 | + γ-concat | 0.81 | 10.28 | 14.53 | 0.36 | n/a |
| **dfm_data50k_ep5** | 8 | DFM at parity | **0.007** | **0.148** | **1.40** | 0.98 | n/a |

Plan target line: KL_bi ≤ 0.50, KL_uni ≤ 0.10. Only DFM crosses it.

Phases skipped:
- Phase 2 (backbone scaling) — skipped per plan rule after Phase 1's KL_bi at ep25
  was 1.65, in the "epoch scaling not the lever" regime.
- Phase 6 (γ-sampling and loss ablations) — not informative until an architectural
  fix unblocks the bottleneck.
- Phase 7 (training-time stability) — energy-field convergence trends across runs
  are non-monotone; no clear pathology to regularise against.
- Phase 9 (long final run) — paused until a winning architecture from the deferred
  next-session work.

## Why each lever did or didn't move bigram coherence

### Phase 1 — Epoch scaling (cheapest path → plan rule says abandon)

5 / 10 / 25 epochs at 10k windows: KL_bi 1.99 → 1.78 → 1.65. Each doubling of
epochs buys roughly 0.03–0.20 nats of bigram KL, asymptoting around 1.6.
Linear extrapolation of the trajectory to the 0.50 target needs ~10⁴ epochs.

**Why scaling alone fails**: at fixed 10k windows the model rapidly memorises the
per-position character distribution (KL_uni 0.05 → 0.015) and then refines the
local energy basins, but it has no joint structure to discover beyond what already
saturates at ep10. The energy gradient at GT *decreases below* the gradient at gen
samples (0.089 / 0.056 at ep25 vs 0.179 / 0.227 at ep5) — the field is becoming
sharper around per-position attractors that don't align with text8 joints. Per
plan §Phase 1's decision rule (KL_bi at ep25 ≥ 1.5), epoch scaling is not the
bottleneck.

### Phase 3 — Data ceiling (the strongest lever found)

The single biggest improvement came from giving the model more unique windows
without giving it more passes over the data:

- 10k × 25 ep (ep25_default): KL_bi 1.65, KL_uni 0.015
- 50k × 5 ep (data_50k_ep5): **KL_bi 1.38**, KL_uni 0.035 — same total samples seen
- 50k × 10 ep (data_50k_ep10): KL_bi 1.56 — 2× exposure regresses joints
- 200k × 2 ep (data_200k_ep2): KL_bi 2.48 — undertrained per-position

**Why variety beats repetition**: at parity compute, more distinct training points
expose the model to more of the bigram statistics text8 actually contains. Each
10k-window slice is heavy-tailed in joint frequencies — most digraph types appear
0–1 times. With 50k windows, the model sees enough instances of medium-frequency
digraphs to fit them.

**Why too much repetition hurts**: at 10 epochs × 50k, the model starts solidifying
per-position attractors that don't generalise to the joints — same overfitting
signal as Phase 1. The 5-pass mark is the sweet spot empirically; under that, the
model can't converge per-position (data_200k_ep2 at 2 passes had KL_uni=0.21 and
gibberish samples).

**Compute wall**: data_full @ 25 ep would be ~290 GPU-hours at this MIG's
throughput, infeasible. The "use all the data" recommendation in the plan can't be
realised at the available throughput.

### Phase 5 — Factorised n-gram NLL (architectural no-op)

The plan suggested adding a bigram NLL term on the implied-x1 reconstruction:

```python
log_p_bigram = log_probs[:, :-1, :, None] + log_probs[:, 1:, None, :]
bg_nll = F.nll_loss(log_p_bigram.reshape(-1, K*K), bg_target.reshape(-1))
```

ng_bg05 (λ=0.5): KL_bi went **up** 28 % (1.38 → 1.77). ng_bg10 (λ=1.0): KL_uni
0.13 (above plan's 0.10 distortion cap), KL_bi 1.80. Both clearly worse.

**Why it can't work as written**: the head emits per-position factorised log-probs
`log p(c)`. The "joint" `log p(a) + log p(b)` is mathematically just the
factorised joint over independent positions. NLL on the observed digram becomes:

    -log p(a) − log p(b)

— equivalent to running unigram CE twice, with interior positions counted in two
adjacent pairs and edge positions in one. It has the same expressive power as the
existing CE (Eq. 7 of the EqM loss), just with a different per-position weighting.
Critically: there's no way for it to penalise an *invalid* digraph that the
factorised marginals would assign high probability to.

To get a real bigram lever the head has to emit `R^{K²}` joints from
`h[t] ⊕ h[t+1]`, not factorise. That's not what the plan's snippet does.

### Phase 8 — DFM head-to-head (the capstone result)

At identical backbone (d=1024, 8 layers), identical data (50k windows), and
identical epochs (5), DFM lands at KL_bi=0.148, EqM at KL_bi=1.382 — a 9.4×
gap. DFM samples are recognisably English-ish; EqM samples are character soup
with correct unigram statistics.

DFM's only architectural advantage: a sinusoidal `t` embedding added to per-token
hidden states (DFMBackbone.forward takes `t` as input). Backbone shape, attention
backend, optimizer, and data are matched.

This isolates the bottleneck almost certainly to *time conditioning + the way the
sampler uses it*.

### Phase 4 — Time conditioning for EqM (the surprising negative)

Implemented per plan §Phase 4: optional sinusoidal γ embedding into
`TransformerBackbone.forward`, in "add" or "concat" modes; sample-time falls back
to γ=1 (the data-manifold endpoint, plan's recommended choice).

Both modes catastrophically collapsed:

| Variant | sample-γ | KL_uni | KL_bi |
|---|---|---:|---:|
| tc_add | 1.0 (plan default) | 1.32 | 4.02 |
| tc_concat | 1.0 (plan default) | 1.02 | 12.93 |

#### Why γ=1 fails

At γ=1 the FM target `c(γ=1) · (x0 - x1) = 0`. The model is therefore trained to
output velocity ≈ 0 at γ=1 (and the training data confirms it: at γ near 1 the
flow loss is the smallest of any bin: g<1 = 0.001 in the training history).
The conservative gradient `∇_x ⟨x, f(x; γ=1)⟩` is then ≈ 0 — the energy field
is flat. NAG-GD has no descent direction; the sampler degenerates to whatever
small bias the residual has, which collapses to a few unigram modes.

This is the single most important lesson from Phase 4: **the plan's recommended
γ=1 sample-time fallback is not just suboptimal, it is degenerate** because of
the c(γ) decay factor in the FM target.

#### Re-evaluation at non-degenerate γ

Re-evaluating the same checkpoints with sample-time γ swept from 0.05 to 0.99
(no retraining, just changing the evaluator):

| sample γ | KL_uni | KL_bi | KL_tri |
|---|---:|---:|---:|
| 0.05 | 0.26 | 4.59 | 8.86 |
| 0.20 | 0.26 | 3.90 | 8.09 |
| 0.50 | 0.52 | 3.62 | 7.74 |
| 0.70 | 0.84 | **3.35** | 7.55 |
| 0.90 | 1.18 | 3.78 | 7.91 |
| 0.99 | 1.29 | 4.04 | 8.18 |

The best γ for KL_bi is around 0.7. **Every choice still loses to tc_off (KL_bi
1.38)** by a large margin.

#### Why γ conditioning hurts EqM even when γ is set sensibly

Three reasons stack up:

1. **Distribution mismatch at sample time.** At training, the model only ever
   sees `x_γ = (1 − γ)·x0 + γ·x1` — a non-trivial mixture of noise and data — at
   each γ. At sample time the iterate starts as pure σ-noise (≈ x0). For any
   γ > 0 we feed the model an x that doesn't match what it saw at training for
   that γ. DFM avoids this because its sampler explicitly walks t from 0 to 1,
   so each forward call is at a t whose training input distribution matches the
   current iterate.

2. **The energy field is γ-dependent**, but the sampler runs at a single fixed γ.
   The training objective doesn't constrain the field's geometry at any γ to
   point toward the data manifold; it only constrains the field's *value*
   (`f(x_γ; γ)`) to match the FM target. Sampling at fixed γ then descends a
   field whose minima have no required geometric relationship to text8.

3. **The conservative-gradient framework adds another layer of indirection.** EqM
   regresses the gradient of `⟨x, f(x; γ)⟩`, not `f` directly. The energy field's
   `γ`-dependence enters quadratically through the dot product, so even small
   variations in `f` across γ produce mismatched gradients at sample time.

Together: DFM uses γ to traverse an integration path, EqM uses it (or would use
it) only to query a static field. The two needs are different; the same
conditioning mechanism doesn't serve both.

## Why DFM works — and what EqM is missing

DFM beats EqM by 9.4× on bigram KL at parity compute (Phase 8). The two
architectures are configured to be as similar as possible — same backbone shape,
same data, same epochs, same optimizer. The gap therefore localises sharply.
This section unpacks where the gap lives.

### One-line answer

**DFM's sampler is the inverse of its training path; EqM's sampler is a
single-shot search over a static field that the training never directly
optimises for sampling.** Time conditioning, the `c(γ)` decay factor, the
input-distribution mismatch, and the conservative-gradient indirection all
follow from this difference.

### Side-by-side anatomy

| Aspect | DFM | EqM |
|---|---|---|
| What the model emits | Denoiser logits `p_{1|t}(x1\|x_t)` (categorical over K, per position) | Velocity field `f(x_γ)` in R^K |
| Loss | Direct CE on observed token at random t | Regression of `∇_x⟨x, f⟩` to the FM target c(γ)·(x0−x1), plus aux CE anchor |
| Source distribution | Uniform over K | σ-Gaussian on V_d (zero-mean simplex coords) |
| Probability path | `p_t = κ_t·δ(x1) + (1−κ_t)·U(K)`. Every t ↔ a known mix | `x_γ = (1−γ)·x0 + γ·x1`, where x0 ~ σ·N(0, I) |
| Time conditioning | Sinusoidal `t` embedding into the backbone | (Optional) sinusoidal `γ` embedding — Phase 4 |
| **Sampler** | **Euler integrator: walk t from 0 to 1 in nfe steps, query model at each t** | **NAG-GD: descend a fixed energy field `⟨x, f(x)⟩` from a noise init** |
| Sample-time conditioning input distribution | Always matches training: at step i the iterate `x_t` is corrupted at level t exactly as during training | Mismatched: at *any* γ > 0 the iterate is pure σ-noise, not the trained `(1−γ)x0 + γx1` mix |
| Inference cost | O(nfe) forwards, no autograd | O(max_steps) forwards + autograd-grad each step |
| Decoding | Argmax of logits at t=1 | Argmax of CLR log-softmax of final iterate |

### Why DFM's setup makes sampling easy

1. **The training objective directly supervises the sampler step.** At training,
   DFM is asked: given a partial corruption x_t (which is `(1−κ_t)·U + κ_t·δ(x1)`),
   what is x1? The Euler sampler then asks the same question at each step. There
   is *no gap* between what the model is trained to predict and what the sampler
   needs.

2. **The sampler walks a path, so each step has its own training signal.** At
   step i, the iterate's distribution is the (i/nfe) progress along the
   probability path — and that's exactly the marginal the model trained on at
   t = i/nfe. No distributional mismatch at any step.

3. **Time conditioning is a path index, not a hyperparameter.** DFM uses `t` to
   tell the model *where on the path* the current iterate sits. The sampler
   knows where it is; the model knows what to predict for that location. They
   are coordinated by construction.

4. **The output is a discrete categorical distribution.** DFM emits softmax
   logits over K characters per position. The sampler can take a Categorical
   draw, and the joint distribution of the L-token output is whatever the
   model's per-position logits factorise to. Bigram structure emerges because at
   high t the input x_t already constrains the joint via the few clean tokens
   that survive the corruption.

### Why EqM's setup makes sampling hard

1. **The field has no committed sample-time γ.** EqM trains f to satisfy a
   regression target *at every γ ∈ (0, 1)*, but the sampler descends one
   particular energy realisation. Two issues stack up:

   - The training signal at any single γ is much weaker than DFM's signal at
     any single t (because the FM target c(γ)·(x0−x1) is an average direction
     across many (x0, x1) pairs, not a per-sample target like DFM's CE).
   - Whatever γ the sampler picks, the input x ≈ σ·noise is OOD relative to
     what the model saw at training for that γ. Phase 4 made this concrete:
     no γ choice in [0.05, 0.99] beat the un-conditioned baseline.

2. **The conservative gradient is two layers of indirection from the data.**
   The training objective is `loss(∇⟨x, f⟩, target)`. So:

       data ← target ← ∇⟨x, f(x)⟩ ← f(x)

   Even if `f` learned the FM regression perfectly, the gradient of `⟨x, f(x)⟩`
   at sample time depends on `f`'s *local Jacobian*, which is not directly
   constrained by the loss. A field whose value matches the target everywhere
   along the simplex doesn't necessarily have minima at the data manifold's
   support points; minima could be anywhere the value happens to vanish under
   the inner-product metric.

3. **The aux CE anchor papers over (2) but only per-position.** The CE branch
   on `pred_x1 = x_γ − λ·grad_g` does pull the gradient field's directional
   structure toward the per-position softmax of x1. This is what stops EqM
   from collapsing to the unigram mode (the "all space" failure documented in
   `SESSION_SUMMARY.md`). But it only constrains *each position
   independently*; nothing in the loss says the joint of pred_x1 across L
   should match the joint of x1.

4. **There is no temporal refinement.** A flow model that walks from noise to
   data inherently spends its early steps roughing out the marginal and its
   later steps refining the joints. EqM's NAG-GD doesn't have phases; it just
   tries to find a local minimum of `⟨x, f(x)⟩`. Whatever joint structure the
   field encodes has to be present *globally*, and the sampler has to land in a
   basin that respects it. Both demands are stronger than what a 5-epoch,
   conservative-grad training run produces.

### The c(γ) trap

A sharp specific consequence of (1)+(2): the FM target is `c(γ)·(x0 − x1)`, where
c(γ) is the schedule's derivative. For the linear decay used by EqM, c(1) = 0.
So at γ=1 the model is trained to output velocity ≈ 0, regardless of x. The
energy `⟨x, f(x; γ=1)⟩` is therefore nearly flat, and `∇⟨x, f(x; γ=1)⟩` is
nearly zero. Setting γ=1 at sample time — *which is what the plan recommends* —
flattens the field the sampler depends on. Phase 4's collapse to a few unigram
modes is exactly this.

DFM has no such pole: its κ schedule is also degenerate at the endpoints, but
the *sampler doesn't sit at one endpoint* — it walks through them. By the time
the Euler integrator is at the t=1 boundary it has already produced an
almost-clean x1 from the previous steps; the final logits just need to match.

### The bigger picture

Phrasing this in one paragraph for the writeup: **DFM and EqM have similar
training-time structure (regress something to a t-indexed target derived from
the probability path), but their sampling mechanisms come from different
mathematical traditions.** DFM's Euler-on-velocity sampler is the natural
inverse of its probability path; the sampler is essentially compiled into the
training objective. EqM's NAG-GD-on-conservative-energy sampler is borrowed
from energy-based models, where the energy is supposed to *be* the data-likelihood
landscape; bolting it onto an FM-regression objective gives an energy field
whose minima are only loosely related to where samples should land.

The most direct path to closing the gap is therefore not better γ conditioning,
not bigger backbones, and not n-gram losses — it's giving EqM a sampler that
walks a γ-path the way DFM does. A small refactor of `EqM.sample` to do an
Euler step on `f(x; γ)` (the velocity, not the energy gradient) would test that
hypothesis cheaply and re-use the trained checkpoints.

### Energy-field health (Phase 7 trigger check)

Plan §Phase 7 asks for a curvature regulariser if `|∇E|gen / |∇E|gt` doesn't
converge to ~5 %. Across all EqM runs this ratio swings either way, never within
that band:

| Run | ratio | shape |
|---|---:|---|
| baseline_5ep | 1.27 | gen above gt — undertrained field |
| ep25_default | 0.63 | gen below gt — over-trained, attractors deeper than GT |
| data_50k_ep5 | 2.18 | gen above gt — field still settling |
| ng_bg10_data50k | 1.88 | gen far from minimum — distorted by n-gram term |

The ratio depends on the training mix, not on a fixed pathology. A curvature
penalty might help but isn't the obvious fix; the diagnostic from Phase 4 makes
it more likely the issue is sampler-side, not training-side.

## Recommendations for next steps

In rough priority order. Items 1–2 are the architectural changes most likely to
move EqM toward DFM's KL_bi.

### 1. Replace EqM's sampler with an Euler integrator over γ

The Phase 4 outcome strongly suggests that the bottleneck is EqM's sampler, not
its conditioning. DFM walks t from 0 to 1; EqM descends a fixed energy field.
Implement a sampler that combines both:

```python
# Pseudocode for an FM-Euler sampler over the conservative-grad model.
x = source_sigma * randn(B, L, K)
for i, gamma in enumerate(linspace(0, 1, nfe + 1)[:-1]):
    h = 1.0 / nfe
    grad = compute_grad(x, gamma=gamma)  # uses the existing conservative grad
    # FM update: x_{t+h} = x_t + h · v(x_t, γ_t), where v is the *stored* velocity,
    # not its gradient. For EqM, v = f(x; γ); we already train this implicitly.
    x = x + h * f(x, gamma=gamma)        # uses raw f, not ∇⟨x, f⟩
```

The conservative-grad regression stays as-is at training time. Only the sampler
changes. If this closes the EqM-DFM gap to within 2×, the capstone narrative
becomes: "the training framework matters less than how you turn the trained
field into samples". This is the single highest-leverage thing to try.

Note: the plan's §6 forbids switching to "raw f(x) descent" *during training*.
Using f(x) inside an Euler sampler at *inference* is a different question; the
training-time conservative-gradient regression still happens.

Compute estimate: ~2 hr to implement and re-evaluate existing checkpoints.

### 2. Non-factorised bigram head

The Phase 5 negative result rules out the plan's spec. A real bigram lever needs
a head that emits `R^{K²}` joints, not factorised log-probs:

```python
class BigramHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.transformer.d_model
        K = cfg.text8_dataset.K
        self.proj = nn.Linear(2 * d, K * K)
    def forward(self, h):  # h: (B, L, d)
        h_pairs = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)  # (B, L-1, 2d)
        return self.proj(h_pairs).reshape(*h_pairs.shape[:-1], K, K)
```

NLL on the observed digram now genuinely supervises the model on joint
structure. This is orthogonal to the sampler change in (1); the two can be
combined.

Compute: ~3 hr to implement + train a single cell at the data_50k_ep5 platform.

### 3. Phase 9 long run

Once one of (1) or (2) closes the gap, take the winning platform and run for
50–100 epochs. text8's ceiling is the data, not the model size. Save
intermediate checkpoints every 25 epochs and rebuild the headline table for the
writeup.

### 4. (Lower priority) Phase 6 ablations

Once the architecture is locked, sweep `lambda_ce`, `gamma_power`,
`ce_min_gamma`, and `decay_strategy`. Most will be no-ops; useful ones graduate
into the platform.

## Lessons that go in the writeup

1. **Variety beats repetition at parity compute.** Three sweep cells lined up to
   show that 50k × 5 ep beats 10k × 25 ep on KL_bi. The model is not capacity-
   bound at 100M params on this data; it's data-coverage-bound for joint
   structure.

2. **Continuous-flow vs discrete-flow is a sampler-architecture difference, not
   just a parameterisation difference.** EqM and DFM use a γ embedding for
   different reasons. DFM's Euler integrator naturally consumes a γ schedule;
   EqM's energy descent has no such schedule. Lifting γ conditioning into EqM
   without rebuilding the sampler doesn't help — and the plan's recommended
   γ=1 fallback actively breaks the model because of the FM target's c(γ)
   decay factor.

3. **The "factorised bigram NLL" trick from the plan is a no-op architecturally.**
   It's a re-weighted unigram CE; cannot capture joints. A real bigram lever
   needs a non-factorised head.

4. **The energy field's "gen / gt gradient norm ratio" is a useful diagnostic**
   (introduced as a plan §Phase 7 trigger check, also reported by `eval_full.py`).
   It distinguishes under-trained from over-fitted fields and exposes Phase 4's
   collapse modes earlier than the KL metrics do.

5. **Compute estimates in the plan were 5× optimistic for this MIG slice.**
   Re-budget any future plan against the measured 14 min / 5 epochs.

## Reproducing the results

```bash
# 1. Re-train the canonical baseline (replaces the missing
#    checkpoints/baseline_5ep/ stub the plan referenced).
python scripts/run_sweep.py --sweep sweeps/phase0.yaml --n 256 --steps 200

# 2. Phase 1 epoch scaling.
python scripts/run_sweep.py --sweep sweeps/phase1.yaml --n 256 --steps 200

# 3. Phase 3 data ceiling (right-sized to fit MIG budget).
python scripts/run_sweep.py --sweep sweeps/phase3.yaml --n 256 --steps 200

# 4. Phase 5 factorised n-gram (negative result).
python scripts/run_sweep.py --sweep sweeps/phase5.yaml --n 256 --steps 200

# 5. Phase 4 time conditioning (negative result).
python scripts/run_sweep.py --sweep sweeps/phase4.yaml --n 256 --steps 200

# 6. Phase 8 DFM head-to-head.
python scripts/run_sweep.py --sweep sweeps/phase8.yaml --n 256 --steps 200
```

`runs/sweep_results.jsonl` aggregates one record per cell. `runs/<name>/eval.json`
has the per-run scorecard plus 8 sample decodes. `runs/DECISION_LOG.md` has the
per-run hypothesis / result / decision entries.
