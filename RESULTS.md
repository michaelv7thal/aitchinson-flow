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
- **Four clean architectural negatives**: factorised bigram NLL, γ-conditioning
  (add and concat), the plan's recommended γ=1 sample-time fallback, and
  3.4× backbone scaling (d=1536/12L is still 11× off DFM's KL_bi).
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
| bb_d256_l4_data50k | 2 | d=256/4L (~1M) | 0.040 | 2.14 | 6.36 | 0.94 | n/a |
| bb_d1536_l12_data50k | 2 | d=1536/12L (~340M) | 0.081 | 1.60 | 5.15 | 0.92 | 0.472 / 0.233 |
| **dfm_data50k_ep5** | 8 | DFM at parity | **0.007** | **0.148** | **1.40** | 0.98 | n/a |

Plan target line: KL_bi ≤ 0.50, KL_uni ≤ 0.10. Only DFM crosses it.

Phases skipped:
- Phase 2 (full 7-cell backbone scaling) — done as a 2-cell mini-sweep
  bracketing extremes (d=256/4L and d=1536/12L) at the data_50k_ep5 platform
  rather than the full sweep, since DFM at d=1024/8L already crossed the
  target and the sampler diagnosis from Phase 4/8 strongly suggested capacity
  was not the lever. Outcome confirmed the diagnosis (see §"Phase 2 mini").
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

### Phase 2 mini — Backbone scaling (capacity is not the bottleneck)

After Phase 8 localised the EqM-DFM gap to sampler architecture, I ran a
2-cell spot-check at the data_50k_ep5 platform to rule out backbone capacity
as a confounder:

| Backbone | Params | KL_uni | KL_bi | KL_tri | H_ratio |
|---|---:|---:|---:|---:|---:|
| d=256 / 4L | ~1M | 0.040 | 2.14 | 6.36 | 0.94 |
| **d=1024 / 8L** (default) | ~100M | 0.035 | **1.38** | 5.96 | 0.99 |
| d=1536 / 12L | ~340M | 0.081 | 1.60 | **5.15** | 0.92 |

Three things to read:

1. **Smaller is worse, not better.** The 1M-param model under-fits the joints
   (KL_bi 2.14 vs 1.38). The plan's "less is more" hypothesis (#4 in §2,
   "100M is over-parameterised") is wrong at 50k windows.

2. **Bigger is *also* slightly worse.** 3.4× the parameter count produces
   KL_bi 1.60 (5 % higher), KL_uni 0.081 (130 % higher), but KL_tri 5.15
   (14 % lower). The bigger model is mode-narrower (H_ratio 0.92, H_gen
   2.62 vs 2.85) — it sharpens the field around fewer attractors. The
   default sits at the bottom of a shallow U-curve.

3. **The EqM-DFM gap survives backbone scaling.** DFM at d=1024 / 8L hits
   KL_bi=0.148. EqM at d=1536 / 12L (3.4× more parameters) is at 1.60 —
   still a 11× gap. **Backbone capacity is not the lever.**

This rules out the most plausible alternative explanation for Phase 8's
result (that EqM was just under-parameterised) and locks in the structural
diagnosis: **the gap is in how each architecture turns the encoder's output
into samples, not in what the encoder can represent.**

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

---

# Objective 3 — OOD detection & self-healing

A separate objective from generation: given a *frozen* full-text8 L256 `DirichletFM`
(`runs/sflm_bench_a100_20g_L256_d1280L14_full/DirichletFM/epoch_best.pt`, md5
`f1d809e8a42d`), can we (a) **detect** corrupted text and (b) **heal** it by
localize-then-inpaint? Everything below is the reproducible benchmark in
`bench_ood/` (detection) and `bench_heal/` (healing); each dir's `manifest.json`
records git SHA + `git_dirty` + a `repro.patch` + `new_scripts/` snapshot + the
exact per-arm CLI. Config: split=**test** (in-distribution; text8 train/val/test are
one distribution), n=256, fit_seqs=512, seed=42, full corruption-rate grid.

Detectors, each at its own best path-time: **NLL** (training-free denoiser surprise,
t=3), **BLR** (linear hinge energy on frozen features, t=4.5), **BGMM** (Bayesian
Dirichlet-process mixture density, t=7.5), **GPT2-SE** (spilled energy under frozen
GPT-2 — the external-LM baseline), plus supervised **full-dim heads**
(`BLR_FI`/`Logistic_*`) trained on specific corruptions. Corruptions:
**replace** (random char), **shuffle** (reorder), **falseinfo** (swap a word for a
random real same-length word), **plausible** (swap for the word that *minimises the
model's own NLL* — fluent but wrong).

> ✅ **All numbers below are from the full re-run of 2026-07-14** (`bench_ood/` 9 arms +
> `bench_heal/` 9 arms, single provenance record; `manifest.json` + `repro.patch` in each).
> Two corrections landed in this run and changed conclusions, so earlier drafts of this
> section are superseded:
> 1. The GPT-2 baseline was wrong twice — it computed a per-token **NLL** and called it
>    *spilled energy*, and it spread BPE scores uniformly onto characters. Both fixed;
>    the old comparators (0.847 / 0.817) are gone. See *"The external-LM baseline,
>    corrected"*.
> 2. Healing selected its operating point by cal-set **F1**, which over-weights recall and
>    picks the *damaging* end of the sweep. Now **F0.5** (precision-weighted; healing is
>    damage-averse). This flipped the insulin BGMM row from **−0.300 → +0.122**.

## Headline

1. **Geometric corruption (replace/shuffle): we win decisively, at the *fair* unit.**
   The training-free **NLL** reaches word-level AUROC **0.981 (replace)** / **0.963
   (shuffle)** at rate 0.15, against GPT-2's **0.738 / 0.746**. This is measured at the
   **word** level — the common unit, not our home turf — so it cannot be dismissed as a
   favourable choice of granularity. In GPT-2's *own* BPE units it does even worse
   (ΔE 0.480, NLL 0.557 ≈ chance): character noise **shatters its tokenization**, so it
   has no stable unit in which to localize.
2. **False-info: GPT-2 wins, and we should say so.** At word level **GPT2_NLL 0.853 vs
   our NLL 0.813** — an honest loss on exactly the axis where real language knowledge
   pays (which *word* belongs in this context). But the gap is small, and much smaller
   than our per-character number implied: pooling to words lifts our NLL **0.731 → 0.813**,
   because false-info *is* a word-level corruption and that is where its signal lives.
   Sequence-level, GPT-2 leads more clearly (0.893 vs 0.743).
3. **A supervised head is NOT needed for false-info** (this supersedes an earlier claim).
   `BLR_FI`, trained *on* false-info, reaches **0.800** at word level — it does **not**
   beat the zero-training NLL's **0.813**. The old "false-info needs a corruption-specific
   supervised head" conclusion was an artifact of measuring per-character.
4. **Plausible (model-fluent) swaps remain the real boundary — and here supervision *is*
   required.** The model's own readout **inverts**: a min-NLL swap is *less* surprising
   than the truth (swapped-word NLL **0.01 vs clean 0.13**), so NLL scores **0.264**
   (worse than chance) and density sits at chance. Only a head trained on plausible
   negatives detects it (**0.904 token / 0.808 seq**; **0.79 after frequency control**).
   GPT-2 is only partial (0.51 token / 0.73 seq).
5. **Healing recovers geometry, never meaning.** Localize-then-inpaint is net-positive on
   char noise (**+0.420** net/corrupt; **+0.374 on the real insulin article**) and
   net-negative on false-info at *every* operating point (−0.059 … −0.140) — a fluent
   wrong word is not recoverable from context.

## Detection — word level, the fair head-to-head (`bench_ood/RESULTS.md`)

Each model is scored in its **native unit** (flow-matching → character, GPT-2 → BPE
token) and both are pooled up to **words** for the comparison. A BPE score is never
attributed *down* onto characters — see *"The external-LM baseline, corrected"* below.

**WORD-level AUROC @ rate 0.15 (max-pool):**

| detector | replace | shuffle | falseinfo |
|---|---|---|---|
| **NLL** (ours, training-free) | **0.981** | **0.963** | 0.813 |
| BLR (replace-trained) | 0.935 | 0.872 | 0.744 |
| BLR_ADV (adversarial mix) | 0.929 | 0.857 | 0.779 |
| BLR_FI (supervised on falseinfo) | 0.818 | 0.762 | 0.800 |
| BGMM (DP density) | 0.888 | 0.886 | 0.707 |
| GPT2_SE (real cross-step ΔE) | 0.727 | 0.724 | 0.690 |
| **GPT2_NLL** (external LM) | 0.738 | 0.746 | **0.853** |

**Native units @ 0.15** (NOT cross-comparable — a char AUROC and a BPE AUROC count
different things; this is each model's localization in the units it actually has):

| detector | unit | replace | shuffle | falseinfo | seq (falseinfo) |
|---|---|---|---|---|---|
| NLL | char | **0.968** | **0.948** | 0.731 | 0.743 |
| BLR | char | 0.874 | 0.783 | 0.522† | 0.635 |
| BLR_FI | char | 0.756 | 0.675 | 0.745 | 0.733 |
| BGMM | char | 0.830 | 0.833 | 0.683 | 0.685 |
| GPT2_SE | bpe | 0.480 | 0.493 | 0.621 | 0.873 |
| GPT2_NLL | bpe | 0.557 | 0.581 | **0.768** | **0.893** |

† BLR trained on *replace* negatives fails to transfer to false-info (0.522 = chance) —
training on the wrong corruption is worse than the one-class density.

**Read the GPT-2 rows carefully.** Its near-chance BPE localization on replace/shuffle is
*not* only a weak score — the unit itself is broken there (char corruption re-tokenizes
the text; see the corrected-baseline section). Its **sequence** AUROC of ~1.0 on
replace/shuffle is correspondingly *cheap*: it detects "this is no longer English" via the
tokenization blow-up. Where GPT-2 is genuinely strong is **false-info**, and that is a real
result.

**SVGP does not help** — the t-SNE shows no nonlinear structure beyond the linear axis
(information limit, not capacity), and its predictive variance saturates to a constant in
1280-d (concentration of measure), which is why the repo retired `DirichletFMSvgp` for the
closed-form Laplace variance. The latent-split full-dim linear probe's **0.93** is
**in-sample** (`_probe_auroc` fits and scores the same tokens); held out it is ~0.75–0.87.

## The plausible boundary (`bench_ood/plausible*/`, n=64, held-out)

Same word slots, two swaps: **random** real word vs **model-min-NLL** ("plausible").
Detector, seq / token AUROC on **plausible**:

| detector | plausible (min-NLL) | plausible (freq-matched) |
|---|---|---|
| NLL | 0.448 / **0.264** (inverts) | 0.46 / 0.34 |
| BLR | 0.455 / 0.492 (chance) | chance |
| BGMM | 0.426 / 0.458 (chance) | chance |
| Logistic-fi (trained on *false-info*) | 0.342 / 0.473 (chance) | chance |
| **Logistic-plaus** (trained on plausible) | **0.808 / 0.904** | **0.66 / 0.79** |
| GPT2_SE (external LM) | 0.700 / 0.514 | 0.76 / 0.65 |
| GPT2_NLL (external LM) | 0.728 / 0.507 | — |

**The inversion is mechanical, not incidental.** The swap is chosen to *minimise the
model's own NLL*, so the planted word is **less surprising than the truth**: mean NLL on
the swapped slot is **0.01 (plausible) vs 0.13 (clean) vs 0.56 (random)**. A likelihood
readout must therefore score *below* chance — 0.264 is the signature of a detector being
used against its own objective, not of a weak signal.

### Cross-transfer: supervised heads are strictly corruption-specific

Both experiments use the **same word slots**, so the only thing that differs is *which*
word is planted (a random real word vs the model's own most-fluent candidate). Training a
full-dim linear head on one and evaluating on the other gives a clean diagonal
(**token AUROC** / *seq AUROC in italics*):

| trained on ↓ &nbsp; eval → | **false-info** | **plausible** |
|---|---|---|
| **false-info** (`Logistic_fi`) | **0.783** / *0.737* ✅ | 0.473 / *0.342* ❌ |
| **plausible** (`Logistic_plaus`) | 0.471 / *0.384* ❌ | **0.904** / *0.808* ✅ |

Each head works **only** on the corruption it was trained on, and the failure is not mere
chance — at sequence level both off-diagonal cells sit *below* 0.5 (0.342, 0.384), i.e.
the heads **anti-transfer**. That sign is informative, not noise:

- The false-info head learns *"this word is lexically out of place"*. A plausible swap is
  chosen to be maximally in-place, so the head scores it as **cleaner than clean**.
- The plausible head learns *"this word is suspiciously fluent for its slot"*. A random
  real word is not fluent, so it fails that test too — in the opposite direction.

They are detecting **orthogonal defects**, not the same defect at two difficulty levels.

### …but a UNIFIED head *is* possible (and the geometry says what shape it must be)

Specialist anti-transfer says nothing about whether *some other* head can span both — it
only says how those heads were *trained*. Two measurements settle it. In the standardised
feature space, the **paired** mean shifts (same positions, clean vs corrupt) genuinely
oppose, but the **learned** directions do not:

```
cos(d_falseinfo, d_plausible) = −0.49     ← mean shifts OPPOSE  (why specialists anti-fire)
cos(w_falseinfo, w_plausible) = +0.40     ← learned directions ALIGN  (so a shared w exists)
```

Clean therefore sits **between** the two corruption clusters. A *signed linear* head
(`w·z`) must pick a side; what the geometry asks for is a **shell** around clean — i.e. a
**sign-agnostic** score that measures *departure*, not direction. Training one head on the
**union** (replace+shuffle+falseinfo+plausible) across three hypothesis classes:

| head | → false-info | → plausible | |
|---|---|---|---|
| `Logistic_fi` (specialist) | **0.783** / 0.737 | 0.473 / 0.342 | anti-fires |
| `Logistic_plaus` (specialist) | 0.471 / 0.384 | **0.904** / 0.808 | anti-fires |
| `ALL_linear` (unified, signed) | 0.740 / 0.706 | 0.597 / 0.611 | ✅ both |
| **`ALL_quad`** (unified, rank-8 quadratic) | 0.727 / **0.740** | 0.683 / **0.704** | ✅ both |
| `ALL_mlp` (unified, MLP) | 0.706 / 0.654 | **0.710** / 0.686 | ✅ both |

*(held-out token / sequence AUROC)*

1. **Unification works.** Every unified head clears chance on *both*, where each specialist
   actively anti-fired on the other. There **is** a general "wrongness" head.
2. **Capacity buys plausible, monotonically** (0.597 → 0.683 → 0.710) — the signed
   hyperplane really was the wrong shape.
3. **The quadratic is the sweet spot, and that is the tell.** A rank-8 *sign-agnostic* form
   captures most of the MLP's gain and has the **best sequence AUROC of any head**
   (0.740/0.704), while a far higher-capacity MLP adds only ~0.03. The cross-corruption
   structure genuinely is a **shell around clean** — the geometry is the functional form,
   not a metaphor.
4. **A real trade-off survives.** No unified head reaches the plausible specialist's 0.904;
   the ceiling is ~0.71, a **−0.19 tax** paid to hold false-info at ~0.71–0.74. The MLP hit
   **0.99 training accuracy** yet only 0.710 held-out — so this is **not** a capacity limit
   but a **representation** limit: the two defects compete along shared directions.

5. **At a *deployable* threshold the quadratic wins outright** — and by far more than AUROC
   implies. Per-token F1 @5% FPR on **plausible**: `ALL_quad` **0.229** vs `ALL_mlp` 0.065
   vs `ALL_linear` 0.031 (**7× the linear head, 3.5× the MLP**), while also leading on
   false-info (0.267). AUROC ranks; F1 is what you actually flag. A *signed projection* has
   no natural scale, so its threshold is arbitrary; **distance-from-clean does**, which is
   why the sign-agnostic score is so much better behaved under thresholding. On AUROC alone
   the MLP looked competitive (0.710 vs 0.683) — at the operating point it is not.

**Deployment.** Know the failure mode → use the specialist. Don't → deploy the **quadratic**
unified head: best sequence AUROC of any head, best deployable F1 on both corruptions, rank-8
and cheap. The corruption-specificity result therefore stands as a statement about
*specialists*, not about what the representation can support.

The frequency-matched control (replacement drawn from the *original word's own
frequency band*) drops the plausible-trained head from 0.90 → **0.79 token** — so
**~0.11 of the raw signal was a genericness artifact** (min-NLL inserts common words)
and **the residual ~0.79 is genuine contextual-wrongness signal**. Net: plausible
errors are undetectable by the model's own unsupervised readouts, detectable only by
a **corruption-specific supervised head** (or an external LM), and true factual
verification still needs world knowledge / retrieval.

## Transfer to real out-of-domain text (`bench_ood/transfer/`)

The full-dim logistic false-info head trained on text8, evaluated on the **insulin
Wikipedia article** (out-of-domain for a text8 model):

| domain | seq AUROC | token AUROC |
|---|---|---|
| text8 (in-distribution) | 0.853 | 0.869 |
| insulin (out-of-domain) | 0.801 | 0.801 |

**Discrimination transfers** (~0.07 drop); it is **threshold calibration** that does
not — clean OOD scores shift, so a text8-calibrated operating point mis-fires. This
reconciles the AUROC-vs-recall gap: ranking survives OOD, calibrated recall does not.

## The external-LM baseline, corrected (`bench_ood/gpt2_se/`, `bench_ood/gpt2_nll/`)

Two bugs and one real methodological result. Full detail in `SESSION_OOD_HEALING.md`
Finding 5; validation in `scripts/validate_spilled_energy.py`.

**1. "Spilled energy" was a per-token NLL.** The real quantity (Minut, Dewidar & Masi,
*Spilled Energy in LLMs*, ICLR 2026, arXiv:2602.18671, Def 4.1/Eq. 8) is **cross-step** —
`ΔE_i = logsumexp(logits[i]) − logits[i-1][x_i]`: the logit energy is read at decoding
step `i-1`, the marginal energy at step `i`, and the residual that fails to cancel is the
signal. The repo computed both terms at the *same* step, which is just `−log p(x_i|x_<i)`.
Now matches the authors' reference implementation to **0.0 error**. (Sign follows their
code, not the paper's Eq. 8 prose, which has a sign typo — a flip inverts AUROC.)
The paper's ΔE≈0 property does hold, but only on text the LM models well: **−0.30 on
in-domain English** vs **+3.4 on clean text8** (GPT-2 finds text8 itself OOD).

**2. ΔE cannot localize — by construction.** It straddles two decoding steps, so it
cannot attribute blame to one: sequence AUROC ~1.0, BPE-level ~0.48 (chance). Our old
headline was therefore *our NLL vs GPT-2's NLL*, and beating ΔE at localization would be
a straw man. The bench now runs **both** `gpt2_se` (real ΔE) and `gpt2_nll` (same-step
NLL — the fair localization comparator).

**3. Evaluation protocol.** Our model scores CHARACTERS, GPT-2 scores BPE TOKENS.
Attributing a BPE score *down* onto characters is ill-posed; pooling *up* is exact. So
each model is scored in its native unit, and the comparison happens at a common one:

> **flow-matching → char · GPT-2 → BPE · both → WORD (the head-to-head)**

**4. Why word level is *required*, not just fairer.** Character corruption **shatters
GPT-2's tokenization**:

| | n BPE tokens | BPE prevalence |
|---|---|---|
| clean / falseinfo@0.05 | ~13,455 | 0.050 |
| **replace@0.15** | **22,801 (+70%)** | **0.399** |
| replace@0.5 | 33,014 (+145%) | 0.771 |
| falseinfo@0.15 | 13,683 (+1.7%) | 0.158 |

At 15% char corruption, **40% of BPE tokens touch a corrupted character** (not 15%). The
BPE unit is *itself a function of the corruption*, so "which BPE token is wrong" is not a
stable question for char-level noise. Word-level corruption (`falseinfo`) leaves the
tokenization intact (+1.7%) and prevalence tracks the rate exactly — so BPE numbers are
meaningful there and not on `replace`/`shuffle`. **Words are the only unit stable across
both tokenizations.**

Corrected numbers — seq | BPE (native) | word (max) | word (mean):

| corruption | GPT2_SE (ΔE) | GPT2_NLL |
|---|---|---|
| replace@0.15 | 1.000 \| 0.480 \| 0.727 \| 0.524 | 1.000 \| 0.557 \| 0.738 \| 0.596 |
| shuffle@0.15 | 1.000 \| 0.493 \| 0.724 \| 0.542 | 1.000 \| 0.581 \| 0.746 \| 0.623 |
| falseinfo@0.15 | 0.873 \| 0.621 \| 0.690 \| 0.672 | 0.893 \| **0.768** \| **0.853** \| 0.838 |

Two caveats worth stating openly:
- **GPT-2's 1.000 sequence AUROC on replace/shuffle is cheap** — it detects "this is no
  longer English" via the tokenization blow-up, not "this token is wrong".
- **GPT-2_NLL genuinely beats us on false-info** (word **0.853 vs our 0.813**). Resolved
  by the full re-run: it is a real loss, on exactly the axis where a pretrained LM's world
  knowledge pays. The gap is small, and much smaller than the per-character figure implied
  (pooling to words lifts our NLL 0.731 → 0.813). Report it as a competitive result.

## Healing benchmark (`bench_heal/RESULTS.md`, least-damaging operating point)

Localize (NLL / BLR / BGMM) → mask → Dirichlet-FM inpaint → score. Reported at the
**max-net/corrupt** operating point (best case; the GT-free deployable choice is
cal-set **F0.5** — precision-weighted, since healing is damage-averse: over-recall
breaks clean tokens and goes net-negative).

All rows below select **fpr = 0.02** (the least-damaging point of the sweep). `loc_*` is
how well the localizer *finds* the corrupt tokens; `fix`/`damage`/`net` is what the
inpainter then *does* with them.

**Synthetic text8** (n_demo=64, 3 corruption seeds):

| localizer | scheme | loc_P | loc_R | loc_F1 | fix | damage | **net/corrupt** |
|---|---|---|---|---|---|---|---|
| NLL | replace | 0.825 | 0.757 | 0.790 | 0.575 | 0.028 | **+0.420** |
| NLL | falseinfo | 0.347 | 0.089 | 0.142 | 0.021 | 0.020 | −0.140 |
| BLR | replace | 0.666 | 0.691 | 0.679 | 0.461 | 0.046 | +0.202 |
| BLR | falseinfo | 0.261 | 0.058 | 0.095 | 0.010 | 0.015 | −0.113 |
| BGMM | replace | 0.599 | 0.318 | 0.416 | 0.242 | 0.017 | +0.144 |
| BGMM | falseinfo | 0.185 | 0.032 | 0.055 | 0.003 | 0.008 | −0.059 |

**Real biomedical article (insulin)** — Track A = char noise, Track C = false info:

| localizer | track | loc_P | loc_R | loc_F1 | fix | damage | **net/corrupt** |
|---|---|---|---|---|---|---|---|
| NLL | A | 0.785 | 0.787 | 0.786 | 0.581 | 0.037 | **+0.374** |
| NLL | C | 0.513 | 0.196 | 0.283 | 0.040 | 0.023 | −0.142 |
| BGMM | A | 0.589 | 0.329 | 0.422 | 0.241 | 0.021 | **+0.122** |
| BGMM | C | 0.246 | 0.059 | 0.096 | 0.003 | 0.010 | −0.075 |
| GMM | A | 0.132 | 0.012 | 0.021 | 0.011 | 0.001 | +0.007 |
| GMM | C | 0.066 | 0.012 | 0.020 | 0.000 | 0.001 | −0.004 |

Char/geometric healing is net-positive **and transfers to the real article** (NLL
**+0.374**); false-info healing is net-negative at every operating point — the localizer
flags few words and the inpainter cannot restore the true one. **Detection ≠ healing.**

> **Corrected (2026-07-14).** The earlier claim that *"BGMM is a poor char-noise localizer
> (net-negative on insulin Track A, −0.30)"* was **an artifact of the operating point**, not
> a property of BGMM. `heal_protein_poc.py` selected by max cal-**F1**, which over-weights
> recall and lands on the damaging end (fpr = 0.10); it also recorded only that single
> point, so the benchmark could not re-select. With **F0.5** + a full FPR sweep, BGMM Track
> A is **+0.122** and GMM is **+0.007** (both previously reported negative). Healing is
> damage-averse — a false-positive edit *breaks a clean token*, and clean tokens vastly
> outnumber corrupt ones — so recall-weighted selection is simply the wrong rule here.

## Lessons (Objective 3)

1. **Training-free NLL is the best localizer for geometric corruption**, and it beats the
   external LM by a wide margin *at the common unit* (word: 0.981/0.963 vs 0.738/0.746) —
   a clean, cheap, positive result that does not depend on a favourable granularity.
2. **A pretrained LM still wins where world knowledge is the signal.** GPT2_NLL beats us on
   false-info (word 0.853 vs 0.813). Honest loss, small margin, entirely explicable — and
   worth reporting, because a reviewer will look for exactly this.
3. **The measurement unit is part of the claim.** Char corruption *shatters* a BPE
   tokenizer (+70% tokens at 15% noise), so BPE-level localization is degenerate there and
   BPE→char attribution is ill-posed in both directions. Score each model in its native
   unit; compare at the coarsest common one. Getting this wrong flattered *and* deflated
   different baselines at the same time.
4. **Supervision is needed for *plausible* errors — but not for false-info** (this is the
   corrected version of an earlier claim). At word level the zero-training NLL (0.813)
   matches a head supervised on false-info (0.800). But no unsupervised readout touches
   plausible swaps: the model's own likelihood **inverts** (0.264), because the swap is
   *constructed* to minimise it. Only a head trained on plausible negatives works (0.904;
   0.79 frequency-controlled).
5. **Specialists anti-transfer — but a unified head exists, and the geometry dictates its
   shape.** The specialist heads form a strict diagonal (false-info-trained → plausible
   0.473; plausible-trained → false-info 0.471; both *below* chance at sequence level, so
   they actively mislead each other). The reason is geometric: the **paired mean shifts
   oppose** (cos = −0.49), so clean sits *between* the corruption clusters and a signed
   hyperplane must pick a side. But the **learned directions align** (cos = +0.40), so a
   shared head does exist — and training one on the union confirms it (all unified heads
   clear chance on both). The right functional form is **sign-agnostic**: a rank-8
   *quadratic* (distance-from-clean in a learned subspace) beats the linear compromise on
   plausible (0.683 vs 0.597) and posts the best sequence AUROC of any head — while an MLP
   with vastly more capacity adds only ~0.03. **The catch:** unification costs ~0.19 AUROC
   on plausible versus its specialist, and the MLP's 0.99 *training* accuracy vs 0.710
   held-out shows this is a **representation** limit, not a capacity one. Know your failure
   mode → specialist; don't → quadratic unified head.
6. **Report held-out, not in-sample.** The latent-split 0.93 was an in-sample ceiling;
   deployed it is ~0.75–0.87. SVGP saturates (information + concentration-of-measure).
7. **Healing is damage-averse and geometry-bound.** A false-positive edit *breaks a clean
   token*, and clean tokens vastly outnumber corrupt ones — so select by precision
   (**F0.5**, not F1), stay at low FPR, and expect net gains only on corruption that leaves
   forensic evidence, never on fluent falsehood. Choosing the recall-weighted rule was
   enough to turn a **+0.12** result into a reported **−0.30**.

Reproduce any arm: `git checkout <manifest.git_sha> && git apply
bench_ood/repro.patch && cp bench_ood/new_scripts/* scripts/ && <arm cmd from
manifest.arms[]>`. Drivers: `scripts/run_bench_ood.py`, `scripts/run_bench_heal.py`;
aggregators `scripts/bench_aggregate.py`, `scripts/run_bench_heal.py --aggregate-only`.
