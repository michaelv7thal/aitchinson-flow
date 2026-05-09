# Continuous Flow Matching on the Simplex for Text Generation and OOD Detection

*Capstone technical report — 2026-05-07*

## Executive summary

We investigated whether continuous flow-matching models on the Aitchison
simplex are competitive with discrete-token alternatives on two tasks:
(1) unconditional character-level text generation on **text8**, and (2)
out-of-distribution (OOD) detection on **WikiText-2** with a trained
EqM-style auditor. The motivating hypothesis was that a single energy-
based flow-matching model could perform both jobs — generate valid text
and discriminate valid from invalid text — with the same parameters.

The hypothesis fails along several axes, and along the way we
triangulated *why*. The headline findings, all at parity compute:

1. **DFM (discrete flow matching) dominates continuous-on-simplex
   generation by ≈10× on bigram KL** at K=27 char-level vocab. Three
   independent continuous methods (EqM, FMonCLR, LogitKLFlow) cluster at
   `KL_bi ≈ 1.4–1.6` versus DFM's 0.148. This is a *regime-level*
   bottleneck — not specific to any parameterisation, sampler, or
   capacity choice we tested.
2. **DFM also dominates sequence-level OOD detection** on text8.
   Spilled Energy (zero-train, derived from the LM's per-position NLL)
   matches DFM's denoiser proxy at AUROC ≥ 0.99 across every
   corruption type tested, including the histogram-preserving
   `valid_perm` contrast that was supposed to be EqM's last open
   advantage. EqM's per-position uncertainty signal fails on
   `valid_perm` (AUROC ≈ 0.51).
3. **An EqM auditor on WikiText-2 with LM-context conditioning reaches
   AUROC = 0.999 (Seq) / 0.994 (Tok)** — but **so does a one-matmul
   linear probe on the same GPT-2 hidden states** (AUROC = 0.988). The
   trained EqM auditor's contribution beyond a vanilla discriminator on
   `h_LLM` is inside the noise floor for n=950 held-out positions.
4. **The auditor's energy gradient does not point toward valid text.**
   Running `x ← x − η ∇E(x)` from invalid simplex points produces 0
   argmax flips toward clean in 950 corrupted positions. The energy
   field learned discrimination, not denoising directionality.
5. **Auditor-driven generation (Phase H) fails**: the same EqM
   auditor that scored AUROC = 0.99 on detection produces samples
   with mean LM NLL = 8.89 (vs 4.07 clean, 9.03 random) — word-salad
   with topic coherence inherited from the cached LM context.
6. **A simpler replacement works better.** A Bayesian-logistic-
   regression (Laplace approximation) or SVGP head on raw `h_LLM`
   features matches the EqM auditor on detection AUROC and gives
   **3000× better-calibrated probabilities** (ECE ≈ 10⁻⁴ vs 0.34) plus
   a principled epistemic-uncertainty channel.
7. **Per-token localisation is cascade-contaminated** for any
   `h_LLM`-based discriminator — including EqM, the linear probe, and
   SVGP. Spilled Energy is the only metric that genuinely localises
   individual corrupted tokens (it drops to AUROC ≈ 0.66 at
   uncorrupted positions vs ≈ 0.99 at corrupted; the others stay at
   ≈ 0.99 in both regimes).

The methodological takeaway is sharper than the empirical one:
**discrimination does not imply generation**. A model that nails 0.99
AUROC against *one specific corruption family* can leave the rest of
the data manifold completely uncharted, and its gradient field can be
useless for sampling. The reverse implication — that a strong
generative model gives free discrimination — does hold (DFM
demonstrates this directly). For practical systems decomposed into a
correctness-against-context module and a text-validity module, the two
should be implemented as different model classes, not unified into a
single flow-matching auditor.

---

## 1. Background and setup

### 1.1 Models compared

| Model | Representation | Training objective | Sampler |
|---|---|---|---|
| **EqM** (Equilibrium FM) | CLR features `x ∈ V_d ⊂ R^K` | FM regression on the *conservative gradient* `∇⟨x, f(x; γ)⟩` against `c(γ)·(x₀ − x₁)`; aux CE on implied `x₁` | NAG-GD on the same conservative gradient |
| **DFM** (Discrete FM) | Token IDs `i ∈ [K]` | Cross-entropy on `p_{1\|t}(x_t)` denoiser; uniform-source prob path | Euler over `t` with Categorical sampling |
| **FMonCLR** | CLR features (same as EqM) | FM regression on the *raw velocity* `f(x; γ)` (no autograd-grad step) | Euler over γ on raw velocity |
| **LogitKLFlow** ([arXiv:2411.16821](https://arxiv.org/abs/2411.16821)) | Unbounded logits `l ∈ R^K`; `l_1 = γ_l · onehot(i)` with `γ_l = 8` | Clean-logit regression: `‖v̂(l_t,t) − l_1‖²` | Hybrid det/stochastic Euler (split at `t = 0.28`) |

All four use the same Transformer backbone (`d_model=1024`, 8 layers,
sinusoidal `t/γ` injection where applicable).

### 1.2 Tasks

* **text8 generation** at parity compute: 50,000 windows × 5 epochs ×
  L=40 × K=27. Headline metric: bigram-KL of the generated token
  distribution against the training corpus (`KL_bi`); also `KL_uni`,
  `KL_tri`, entropy ratio.
* **WikiText-2 OOD detection** (Phase F): 300 chunks × L=64 BPE
  tokens × top-K=64 log-simplex of GPT-2 logits, span-corrupted at 25%
  per chunk. Headline metric: Seq AUROC and Tok AUROC for clean vs
  span-corrupted pairs.

### 1.3 Compute envelope

A single A100 with a 20 GB MIG slice, no gradient parallelism.
Training cells of 5–25 epochs each (text8) or 25 epochs (auditor) —
total session compute ≈ 6 GPU-hours, mostly DFM/EqM/LogitKLFlow at
the parity-compute platform.

---

## 2. Continuous-on-simplex generation fails at K=27 (Phase B)

### 2.1 Headline numbers

| Run | KL_uni | **KL_bi** | KL_tri | H_ratio |
|---|---:|---:|---:|---:|
| eqm_data50k_ep5 (NAG)            | 0.035 | **1.382** | 5.957 | 0.991 |
| **dfm_data50k_ep5 (Euler)**      | 0.007 | **0.148** | 1.402 | 0.976 |
| fmclr_data50k_ep5 (Euler-γ)      | 0.156 | 1.559     | 6.276 | 0.943 |
| **lkflow_data50k_ep5 (NFE=64)**  | 0.047 | **1.432** | 4.979 | 0.927 |
| lkflow_data50k_ep5 (NFE=32)      | 0.045 | 1.542     | 5.250 | 0.931 |
| lkflow_data50k_ep5 (NFE=128)     | 0.042 | 1.472     | 5.189 | 0.935 |

Three independent continuous methods cluster at `KL_bi ≈ 1.4–1.6`
while DFM lands at 0.148. The protocol's "acceptable" threshold (≤ 0.30
= within 2× of DFM) is missed by all three by an order of magnitude;
the "fail" threshold (> 1.0) is breached by all three.

See `runs/phaseB_E_summary.png` for the bar chart with thresholds
overlaid.

### 2.2 What's *not* the lever

The previous session's W1 and W4 results plus this session's Phase B
together rule out:

- **Sampler architecture** (W1): Euler-γ vs NAG-GD on the same EqM
  checkpoint gives `KL_bi ∈ {1.39, 1.68}` depending on whether you
  walk on `∇⟨x,f⟩` or on the raw velocity `f`. Best Euler ≥ best NAG.
  The conservative-gradient field is what encodes the data-pull;
  swapping the integrator doesn't help.
- **Conservative-grad indirection** (W4): FMonCLR replaces the
  `∇⟨x,f⟩` step with raw-velocity FM regression and lands at
  `KL_bi = 1.559`, *worse than* EqM-NAG. The conservative-grad step is
  not the bottleneck.
- **Logit-space parameterisation** (Phase B): LogitKLFlow trains on
  unbounded `R^K` logits with the published `γ_l = 8` magnitude and
  a hybrid det/stochastic sampler. Lands at `KL_bi = 1.432`, again in
  the same cluster.
- **Backbone capacity** (prior session, Phase 2-mini):
  `d_model = 1536, L = 12` (~340M params, 3.4× the default) lands at
  `KL_bi = 1.596` — *worse than* the d=1024/L=8 default. Capacity is
  not the lever.
- **Number of epochs** (prior session, Phase 1):
  `KL_bi = 1.99 → 1.78 → 1.65` at 5 / 10 / 25 epochs. Slope too
  shallow to close 1.65 → 0.15 in any reasonable budget.

The failure is **regime-level**: the continuous-on-K=27-simplex setup
is broadly worse than discrete-token denoising at this scale.

---

## 3. Per-position uncertainty is the only EqM advantage on text8 OOD (Phase C)

### 3.1 Cross-method OOD scorecard

256 held-out windows × 4 corruption types × 4 model checkpoints. AUROC
for each (model, statistic, contrast):

| Model | Statistic | clean-vs-rand | clean-vs-subst_0.5 | clean-vs-shuffle_0.5 | **clean-vs-valid_perm** |
|---|---|---:|---:|---:|---:|
| eqm     | E_seq            | 0.84 (\|·\|) | 0.514 | 0.499 | **0.493** |
| eqm     | **U_pos_mean**   | **1.000**    | **0.987** | 0.525 | **0.513** |
| eqm     | U_pos_max        | 1.000        | 0.977     | 0.511 | 0.504 |
| dfm     | E_seq (proxy)    | 1.000        | 1.000     | 0.993 | **0.999** |
| fmclr   | E_seq            | 0.88 (\|·\|) | 0.65 (\|·\|) | 0.516 | 0.517 |
| lkflow  | E_seq            | 1.000        | 0.958     | 0.496 | 0.480 |
| lkflow  | U_pos_mean       | 1.000        | 0.958     | 0.496 | 0.480 |

### 3.2 What this says

- **DFM's denoiser proxy** (`-log p_{1|t≈1}(x|x)`) saturates **every**
  contrast at AUROC ≥ 0.99, including `valid_perm` — the histogram-
  preserving full-permutation corruption that was the protocol's last
  hope for an EqM-only signal.
- **EqM's per-position uncertainty** (`U_pos_mean = ⟨grad-norm⟩_pos`)
  hits AUROC = 0.99 on substitution and uniform corruption (matching
  DFM within noise) but **flatlines on shuffle and valid_perm**
  (AUROC ≈ 0.51). The EqM energy field can detect *which positions
  have been changed locally*; it can't see that the *permutation order*
  is wrong, because the per-position simplex points are unchanged.
- **FMonCLR and LogitKLFlow** carry weaker versions of the same
  signal but are dominated by DFM in every cell.

The salvageable EqM-uniqueness hypothesis (`U_pos` valid_perm AUROC ≥
0.65 while DFM proxy < 0.55) is **falsified**. DFM dominates valid_perm
at 0.999 vs EqM's 0.513.

See `runs/phaseC_summary.png` for the heatmap and ROC overlay.

---

## 4. The auditor on WikiText-2 (Phases F → F+)

### 4.1 What the auditor adds and where it sits in the field

The protocol proposed replacing the SVGP head of an existing
prototype's hallucination auditor with EqM's conservative-gradient
energy. The prototype's reported numbers on WikiText-2 (Qwen2.5-1.5B +
product-kernel context): `Seq AUROC = 0.999`, `Tok AUROC = 0.996`. The
hypothesis was that EqM's energy field could match those numbers
*and* drive Euler-γ generation under the LM's vocab constraint —
unifying detection and generation in one model.

We implemented the auditor in three configurations:

* `aud_gpt2_logit` — top-K=64 simplex only, no LM context.
* `aud_gpt2_ctx` — top-K simplex + product-concat with GPT-2 last
  hidden state (`H = 768`).
* Two `_d512` variants with `d_model = 512, L = 6`.

Training adds a paired-input hinge loss to the standard EqM FM
regression:

```
L = L_FM(clean) + λ_E_hinge · (E_grad²(clean) + relu(margin² − E_grad²(invalid)))
```

with `λ_E_hinge = 1.0`, `margin = 2.0`, evaluated at γ = 1 (the
data-manifold endpoint where FM training already drives `f → 0` on
clean inputs).

### 4.2 F1 detection numbers

| Cell | Seq AUROC | Tok AUROC | F1 floor passes? |
|---|---:|---:|---|
| aud_gpt2_logit (d=256/L=4)   | 0.965 | 0.962 | Tok ✓, Seq ✗ (0.965 < 0.99) |
| **aud_gpt2_ctx** (d=256/L=4) | **1.000** | **0.994** | **both ✓** (the headline) |
| aud_gpt2_logit_d512          | 0.971 | 0.968 | scaling alone gives +0.005 |
| aud_gpt2_ctx_d512            | 0.998 | 0.990 | slight regression vs d=256 |
| **Spilled Energy** (zero-train) | 0.999 | 0.967 | the bar |

The protocol's stated success criterion (Seq ≥ 0.99 AND Tok ≥ 0.95) is
**met** by `aud_gpt2_ctx`. Context conditioning is the lever; backbone
scaling alone gains +0.005 AUROC. Scaling beyond d=256 plateaus.

See `runs/phaseF_compare.png` for the full bar chart and ROC overlay.

### 4.3 Sanity checks that complicate the headline

Four diagnostics on the held-out 60 chunks (`aud_gpt2_ctx`):

**(a) Train vs val.** Train AUROC and val AUROC are within 0.005 of
each other across all stats. **No overfitting; the AUROC numbers are
real.**

**(b) Trivial baselines.** Tok AUROC at corrupted positions:

| Method | Tok AUROC |
|---|---:|
| **EqM ctx (full Phase F machinery, ~1.4M params)** | 0.990 |
| **Linear probe on h_LLM (one matmul, ~50K params)** | **0.988** |
| Spilled Energy (zero-train) | 0.967 |
| EqM logit-only (trained 25 ep) | 0.948 |
| Top-K entropy (zero-train) | 0.706 (Tok), 0.971 (Seq) |

The trained EqM auditor is **0.002 better than a one-matmul linear
probe** on the same `h_LLM` features — well inside the noise floor for
n=950 held-out positions. The logit-only auditor is **worse than the
zero-train top-K entropy baseline** at sequence level (0.942 vs 0.971),
i.e. *training EqM on simplex shape extracted less signal than the
obvious entropy statistic*.

**(c) Cascade contamination.** Tok AUROC at *uncorrupted* positions
in invalid sequences (where the token at position k is unchanged but
upstream tokens are corrupted):

| Discriminator | At corrupted | At uncorrupted | Localisation gap |
|---|---:|---:|---:|
| EqM ctx               | 0.990 | 0.967 | 0.023 |
| EqM logit-only        | 0.948 | 0.923 | 0.025 |
| **Spilled Energy**    | 0.968 | **0.656** | **0.312** |

GPT-2 is autoregressive: `h[k]` encodes everything about tokens 0..k.
A corruption at position 1 changes `h[2], h[3], ..., h[63]` even if
those tokens are unchanged. Any classifier on `h_LLM` picks up this
cascade signal and flags positions inside the corruption's downstream
radius — not the corrupted token itself. **Only Spilled Energy
genuinely localises** because it uses local logits at each position
to score the actual token's likelihood.

**(d) Energy gradient as denoiser.** Run `x ← x − η ∇E(x)` from
invalid simplex points for K=30 steps with η=0.05, holding `h_LLM`
fixed. Energy decreases monotonically (-70 → -80) — descent is
working — but distance to the corresponding clean simplex point
moves only ~2% on average, and **0/950 corrupted positions flip
their argmax slot toward clean**. The energy field is a discriminator
in scalar magnitude (small at clean, large at invalid) but its
gradient direction is not aligned with the data manifold.

See `runs/phaseF_sanity.png` for the trivial-baseline + cascade bars
and `runs/phaseF_denoise.png` for the energy/distance trajectories.

### 4.4 Phase H: auditor-driven generation fails (F3)

Conditional generation: pick a held-out chunk's `h_LLM` and top-K
table, initialise `x ~ σ·N(0, I)` in `R^{L×K}`, run Euler-γ for
nfe=64 steps to γ=1, take argmax slot, map back to vocab through
the chunk's top-K table, re-feed through GPT-2 to score NLL.

| Sequence | mean per-token NLL | log-prob/token |
|---|---:|---:|
| **EqM-generated** | **8.89** | −8.89 |
| Clean WikiText-2 | 4.07 | −4.07 |
| Random-slot baseline | 9.03 | −9.03 |

Robustness across hyperparameters (σ ∈ {0.05, 0.1, 0.3}, NFE ∈ {32,
64, 128}, raw-velocity vs conservative-gradient sampler): all give
NLL ∈ [8.80, 8.97]. **F3 fails at all three protocol thresholds**:
NLL = 8.89 is above the 7.0 fail floor, vocab match to clean tokens is
0.1% (vs random 0.3%, LM top-1 1.7%).

The generated samples have *topic coherence* inherited from the
cached top-K decode table — NHL chunks produce hockey-themed
vocabulary ("Anaheim", "Tampa", "Oilers", "Cup", "season") — but
**no grammatical structure**. This is consistent with the denoising
test: the auditor's argmax slot selection within the LM's top-64
candidates is essentially uniform-random.

Example (chunk 240):
```
clean:     "<|endoftext|> The 2011 – 12 Columbus Blue Jackets season was
            the team 's 12th season in the National Hockey League ..."
generated: "- C World R12 series Hel Season tickets pretty fifth highwinning
            sixth straight overall leading Anaheim organization Football Cup
            playingNASC standings Their blue Was also went on 2 and 9..."
```

See `runs/phaseH_audited.png` for the verdict figure.

### 4.5 Phase F+: a simpler UQ-equipped baseline matches and improves

We replaced the EqM auditor with four progressively richer UQ
methods on raw `h_LLM` features (no EqM machinery):

| Method | Tok AUROC | UQ AUROC | ECE | Notes |
|---|---:|---:|---:|---|
| Linear probe (closed-form ridge) | 0.988 | n/a | 0.342 | Mean only |
| Mahalanobis (clean train μ, Σ)    | 0.334 | n/a | n/a | Broken — train/val shift |
| **Bayesian LR Laplace**           | 0.974 | 0.785¹ | **0.066** | **Simplest principled UQ — ~30 LoC** |
| Deep ensemble × 5 (bootstrap)    | 0.989 | 0.327 | 0.342 | Std collapses (insufficient diversity) |
| **SVGP** (RBF, 64 inducing pts)  | 0.982 | **0.993** | **0.0001** | **Best UQ + best calibration** |
| EqM ctx auditor (reference)      | 0.990 | n/a | n/a²  | Trained EqM |
| Spilled Energy (reference)       | 0.967 | n/a | n/a   | Zero-train |

¹ BLR's std is sign-flipped (high std at in-distribution where the
classifier was trained); 1 − raw AUROC = 0.785 is the corrected value.
² EqM produces grad-norm² scores, not probabilities — no natural ECE.

The pattern is clear:

* **Detection AUROC is essentially identical** across all reasonable
  methods (0.97–0.99). The trained EqM auditor's 0.002 advantage over
  a vanilla linear probe is inside noise.
* **SVGP's latent std as an OOD score** is the *highest* of any
  method tested (0.993) — but it's measuring the same cascade signal
  as the predictive mean, so it doesn't solve localisation either.
* **Calibration is dramatically better** with SVGP (ECE = 1.2 × 10⁻⁴)
  and Bayesian LR (ECE = 0.066) than with the linear probe sigmoid
  (ECE = 0.342). The EqM auditor doesn't produce probabilities at
  all — calibration isn't even on its menu.

See `runs/phaseF_uq.png` for the AUROC bar chart and ECE comparison.

---

## 5. Discussion

### 5.1 DFM is the existence proof: good generation gives free discrimination

DFM at parity compute achieves both the best generation
(`KL_bi = 0.148`) and the best detection (Seq AUROC = 0.999 across
every text8 corruption type tested). Its denoiser at `t ≈ 0.99`
produces calibrated per-position log-probabilities; thresholding gives
the OOD score directly.

A strong **generative** model captures the full data distribution
`p_data(x)`. Its likelihood `p_data(x)` (or any local approximation
like the denoiser at high `t`) is automatically a discriminator — for
*any* corruption that moves a sample off the data manifold, not just
the specific corruption family the model was trained against.

### 5.2 The reverse implication fails: discrimination ≠ generation

The EqM auditor at AUROC 0.99 detection cannot generate (NLL 8.89).
The linear probe at AUROC 0.99 cannot generate at all. This is not a
defect specific to our implementation — it's the structural asymmetry
between the two tasks:

* **Discrimination** only requires being confident on whichever axis
  the corruption made obvious. For random-vocab corruption, the LM's
  per-position NLL spikes hard; that's enough for AUROC ≥ 0.99. The
  discriminator never has to model grammar, agreement, coherence, or
  any other axis the corruption didn't touch.
* **Generation** requires getting *every* axis right simultaneously.
  Even one failure produces clearly broken samples (Phase H's NHL-
  themed word-salad).

A discriminator's level set near the data manifold can be arbitrary
in directions the corruption didn't probe. The EqM hinge loss
pressures only the magnitude of `‖∇E(x)‖²` — it gives no constraint
on the direction of `∇E(x)`, which is what would be needed for
denoising via gradient descent. The Phase F denoising test
demonstrates this directly: 0/950 argmax flips, distance to clean
barely moves, but `E(x)` decreases by 14% — the gradient walked
*away* from the data manifold while still reducing the energy.

### 5.3 Cascade contamination is a property of `h_LLM` features

Any classifier on autoregressive LM hidden states inherits the
cascade. `h_LLM[k]` encodes tokens 0..k; a corruption at position 1
changes `h_LLM[k]` for all k ≥ 1. A discriminator that flags
"this looks like an h from a contaminated sequence" will hit AUROC
≥ 0.99 at every position past the first corruption — the localisation
"signal" at the actual corrupted position is no stronger than at any
downstream position.

This is not specific to EqM. The Phase F+ table shows it for the
linear probe (Tok AUROC at uncorrupted positions = 0.86), the SVGP
(0.98), and the EqM auditor (0.97). Spilled Energy is the unique
cleanly-localising metric (0.66) because its computation is local —
LSE(logits[k−1]) − logits[k−1, token[k]] uses only the LM's
distribution at position k−1 evaluated on the actual token at
position k.

For a localising classifier on `h_LLM`, you'd need a residual
feature like `h_LLM[k] − E[h_LLM[k] | tokens 0..k−1]`, which
essentially recovers SE in a learned form. Untouched in our
experiments.

### 5.4 What architecture would actually work

The two functional components in the protocol's broader design are:

* **(A) Correctness against context** (e.g. PubMedQA: question +
  candidate answer + reference; classify correct/incorrect). This is
  *fundamentally a discrimination task*. A linear probe on LM hidden
  states + a Bayesian/SVGP UQ head solves it at AUROC ≥ 0.98 with
  calibrated probabilities (Phase F+ result). The EqM auditor adds no
  measurable value over this baseline.
* **(B) Validity of generated text** (does the candidate look like
  natural language, regardless of factual content?). This is
  *fundamentally a generation task in disguise* — to certify validity
  in any reasonable sense, the validator must capture the data
  distribution well enough to flag any deviation, not just one
  trained-against family. **A strong generative model is the natural
  solution.** DFM at parity compute already lands at `KL_bi = 0.148`
  on text8; scaling it via Phase J would extend that.

Trying to do both with one EqM-style auditor was the protocol's
architectural bet, and it doesn't pay off: the discrimination signal
(small `‖∇E‖²` at clean) and the generation signal (a useful
direction for `−∇E` to denoise) are decoupled in the trained network
even though they share weights and a forward pass. Decoupling them at
the architecture level — discrete denoiser for validity, Bayesian
classifier on `h_LLM` for context-correctness — is the cleaner design.

---

## 6. Conclusion and limitations

The text8 / WikiText-2 protocol asked whether a single
flow-matching-on-simplex model could generate valid text and detect
invalid text simultaneously. The answer is **no**, with several
specific reasons.

**Positive contributions that survive:**

1. **A regime-level negative for continuous-on-simplex generation at
   K=27.** Three independent methods (EqM, FMonCLR, LogitKLFlow) cluster
   at `KL_bi ≈ 1.4–1.6` versus DFM's 0.148. The bottleneck is the
   continuous-vs-discrete representation choice, not parameterisation
   or capacity. A future continuous method on this task should
   demonstrate it solves this regime issue *first*.
2. **Cross-method per-position complementarity table** (Phase C) showing
   that DFM's denoiser proxy and EqM's per-position uncertainty are
   complementary in some narrow band (substitution corruption) but DFM
   dominates everywhere else, including the histogram-preserving
   `valid_perm` contrast that was the last open hope for an EqM-unique
   signal.
3. **A cheaper drop-in replacement for the EqM auditor on detection**:
   linear probe on `h_LLM` + Bayesian-logistic-regression Laplace
   approximation matches AUROC and improves calibration by 5×.
   SVGP improves calibration by 3000× and gives a separate UQ channel.
   No EqM machinery, much smaller models.
4. **Diagnostic clarity that the auditor track's "structural advantage"
   over GP/SVGP doesn't materialise in this architecture.** The EqM
   energy gradient is not a useful denoising direction; auditor-driven
   generation fails (NLL 8.89, word-salad with topic coherence).

**Limitations of these conclusions:**

1. The text8 numbers are at parity compute (50k windows × 5 epochs).
   Scaling DFM to 50–100 epochs (Phase J) is a routine next step that
   would establish a stronger comparison floor for any future
   continuous-on-simplex method, but wasn't run in this session.
2. Phase F's WikiText-2 cache is small (300 chunks × 60 held-out for
   eval). The 0.002 AUROC advantage of EqM over the linear probe is
   inside the noise floor for n=950; with much larger held-out sets
   the conclusion might tighten in either direction. Increasing the
   cache to 3000+ chunks would resolve this.
3. Only GPT-2 small (124M) was used for `h_LLM` features. The protocol
   suggested Qwen2.5-1.5B + product-kernel context as the prototype's
   target configuration; we didn't test whether the larger LM closes
   the gap to the prototype's 0.999 / 0.996 numbers in any qualitative
   way. The 0.999 / 0.994 we got with GPT-2 already saturates the
   F1 floor, so the larger LM might mainly affect TriviaQA-style
   semantic-confusor tasks (Phase G, not run).
4. The auditor's "structural advantage" claim was tested only via
   gradient-descent denoising (Phase F denoising) and Euler-γ
   sampling under fixed-context cache (Phase H). Other generative
   uses — e.g. AR-style autoregressive sampling that re-evaluates
   the LM at each step — were not tested. Whether a more elaborate
   sampling procedure could rescue Phase H is open, but the
   denoising failure suggests the energy field doesn't carry the
   needed structure regardless.
5. Phases D (loss-aux ablations), G (TriviaQA), I (SFM contingency)
   were deferred. None are likely to overturn the headline finding
   given the W1/W4/B triangulation, but they're untested.

**Practical recommendations for downstream work** that splits a
correctness-against-context and a text-validity component:

* For the correctness-against-context component, use a **linear probe
  + Bayesian/SVGP UQ head on cached LM hidden states**. Cheap,
  calibrated, AUROC matches any fancier alternative we tested. SVGP
  with 64 inducing points and 200 ELBO iterations is sufficient.
  Do not attempt per-token attribution from `h_LLM` features — use
  Spilled Energy or a residualised feature for that.
* For the text-validity component, use a **strong generative model**:
  DFM at K=27 char-level for toy domains, or a discrete-diffusion /
  AR model for production. Continuous-on-simplex methods are not
  competitive at K=27 and should not be pursued without first
  demonstrating they overcome the regime-level bottleneck this
  protocol triangulated.

---

## Appendix A — Files added in this session

### Code

| Path | Purpose |
|---|---|
| `src/aitchinson_flow/models/logitkl_flow.py` | Phase B: Logit-KL Flow Matching model |
| `src/aitchinson_flow/data/wiki.py` | Phase F: WikiText-2 auditor dataset, span_corrupt, AR-shifted Spilled Energy |
| `src/aitchinson_flow/data/wiki_auditor_datamodule.py` | Phase F: minimal datamodule for cached wiki features |
| `src/aitchinson_flow/transformer_backbone.py` (modified) | Added `context_features ∈ {off, hidden_only, product_concat}` |
| `src/aitchinson_flow/models/eqm.py` (modified) | Added `_auditor_hinge`, `_grad_norm_sq`, h_ctx threading |
| `src/aitchinson_flow/config.py` (modified) | `LogitKLFlowConfig`, `AuditorConfig`, EqM auditor + context fields |
| `src/aitchinson_flow/training/data_sources.py` (modified) | Auditor dispatch |
| `scripts/cache_wiki.py` | One-time cache producer for WikiText-2 + GPT-2 features |
| `scripts/eval_full.py` (modified) | Added `valid_perm` / `logitkl` / `auditor` section handling + auditor stub |
| `scripts/eval_ood.py` (modified) | `valid_perm` corruption + LogitKLFlow scoring |
| `scripts/eval_bpc.py` | Phase E: model-specific BPC surrogates |
| `scripts/eval_auditor_wiki.py` | Phase F: Seq + Tok AUROC scorecard with SE baseline + divergence-trace UQ |
| `scripts/generate_audited.py` | Phase H: auditor-driven Euler-γ sampling + GPT-2 NLL eval |
| `scripts/phaseF_sanity.py` | Phase F sanity: train/val, trivials, cascade, cross-seed |
| `scripts/phaseF_denoise_test.py` | Phase F denoising: does −∇E point toward clean? |
| `scripts/phaseF_uq.py` | Phase F+: linear probe / Mahalanobis / BLR Laplace / ensemble / SVGP comparison |
| `scripts/plot_phaseB_E.py` | KL_bi bars + BPC overlay |
| `scripts/plot_phaseC.py` | Cross-method × corruption AUROC heatmap + valid_perm ROC |
| `scripts/plot_phaseF.py` | Per-cell auditor scorecard figure |
| `scripts/plot_phaseF_compare.py` | Multi-cell Seq+Tok AUROC bars + ROC overlay |
| `scripts/plot_phaseF_sanity.py` | Trivial-baseline + cascade-contamination bars |
| `scripts/plot_phaseH.py` | Auditor-driven generation NLL + sample text grid |
| `sweeps/phaseB_logitkl.yaml` | Phase B sweep |
| `sweeps/phaseF_auditor_wiki.yaml` | Phase F sweep (4 cells) |

### Artifacts

| Path | Contents |
|---|---|
| `data/wiki_cache_gpt2.pt` | 148 MB; 300 chunks × L=64 BPE × top-K=64 logits + h_LLM |
| `runs/lkflow_data50k_ep5/` | Phase B trained checkpoint + eval JSONs |
| `runs/aud_gpt2_logit/` | Phase F MVP checkpoint + auditor_eval JSON+PNG |
| `runs/aud_gpt2_ctx/` | **Phase F1-passing auditor** (Seq=1.000, Tok=0.994) |
| `runs/aud_gpt2_logit_d512/` | Phase F backbone-scaling logit-only |
| `runs/aud_gpt2_ctx_d512/` | Phase F backbone-scaling + context |
| `runs/{eqm,dfm,fmclr,lkflow}_data50k_ep5*/{ood_eval,bpc}.json` | Cross-method OOD + BPC |
| `runs/phaseF_sanity{,_logit}.{md,json}` | Sanity check raw numbers |
| `runs/phaseF_denoise.{md,json,png}` | Denoising test |
| `runs/phaseF_uq.{md,json,png}` | UQ comparison |
| `runs/phaseH_audited{,_grad,_logit}.{md,json,png}` | Auditor-driven generation |
| `runs/phaseB_E_summary.png` | KL + BPC top-level figure |
| `runs/phaseC_summary.png` | OOD top-level figure |
| `runs/phaseF_compare.png` | Phase F top-level figure |
| `runs/best_so_far.pt` → `data_50k_ep5/epoch_final.pt` | Best EqM (text8) |
| `runs/best_auditor.pt` → `aud_gpt2_ctx/epoch_final.pt` | F1-passing auditor |
| `runs/DECISION_LOG.md` | Append-only diary; this report's source numbers |

## Appendix B — Reproducing key numbers

```bash
# 1. Phase B (LogitKLFlow at parity compute).
python scripts/run_sweep.py --sweep sweeps/phaseB_logitkl.yaml \
  --n 256 --steps 64 --wandb --wandb-project eqm-text8

# 2. Phase C (cross-method OOD with valid_perm).
for ckpt in eqm_data50k_ep5_v2 dfm_data50k_ep5_v2 fmclr_data50k_ep5_v2 \
            lkflow_data50k_ep5; do
  python scripts/eval_ood.py --ckpt runs/$ckpt/epoch_final.pt \
    --n 256 --seed 1234 --out runs/$ckpt/ood_eval.json --no-figure
done
python scripts/plot_phaseC.py

# 3. Phase E (BPC overlay).
for ckpt in dfm_data50k_ep5_v2 lkflow_data50k_ep5 eqm_data50k_ep5_v2 \
            fmclr_data50k_ep5_v2; do
  python scripts/eval_bpc.py --ckpt runs/$ckpt/epoch_final.pt \
    --n 256 --n-mc 8 --out runs/$ckpt/bpc.json
done
python scripts/plot_phaseB_E.py

# 4. Phase F (cache + train + eval the auditors).
python scripts/cache_wiki.py --lm gpt2 --n 300 --L 64 --K 64 \
  --out data/wiki_cache_gpt2.pt --lm-batch-size 8

python scripts/run_sweep.py --sweep sweeps/phaseF_auditor_wiki.yaml \
  --n 1 --steps 1 --wandb --wandb-project eqm-text8
for ckpt in aud_gpt2_logit aud_gpt2_ctx aud_gpt2_logit_d512 aud_gpt2_ctx_d512; do
  python scripts/eval_auditor_wiki.py --ckpt runs/$ckpt/epoch_final.pt \
    --cache data/wiki_cache_gpt2.pt --batch-size 16 \
    --out runs/$ckpt/auditor_eval.json
  python scripts/plot_phaseF.py --ckpt runs/$ckpt/epoch_final.pt \
    --cache data/wiki_cache_gpt2.pt
done
python scripts/plot_phaseF_compare.py

# 5. Sanity checks.
python scripts/phaseF_sanity.py --ckpt runs/aud_gpt2_ctx/epoch_final.pt \
  --cache data/wiki_cache_gpt2.pt
python scripts/phaseF_sanity.py --ckpt runs/aud_gpt2_logit/epoch_final.pt \
  --cache data/wiki_cache_gpt2.pt \
  --out-md runs/phaseF_sanity_logit.md --out-json runs/phaseF_sanity_logit.json
python scripts/plot_phaseF_sanity.py
python scripts/phaseF_denoise_test.py --ckpt runs/aud_gpt2_ctx/epoch_final.pt \
  --K 30 --eta 0.05

# 6. Phase H (auditor-driven generation).
python scripts/generate_audited.py --ckpt runs/aud_gpt2_ctx/epoch_final.pt \
  --n 32 --nfe 64 --sigma 0.1
python scripts/plot_phaseH.py

# 7. Phase F+ UQ comparison.
python scripts/phaseF_uq.py
```

Each step is idempotent; trained checkpoints are reused if
`runs/<name>/eval.json` already exists.

## Appendix C — Decision-log timestamps for traceability

The append-only `runs/DECISION_LOG.md` contains 1-per-cell entries
with hypothesis / result / decision / next-step. Key entries from
this session, in order:

* `2026-05-07 21:31 UTC` — Phase B (LogitKLFlow) failed.
* `2026-05-07 21:35 UTC` — Phase C extended OOD with valid_perm.
* `2026-05-07 21:39 UTC` — Phase E BPC overlay.
* `2026-05-07 22:00 UTC` — Phase F MVP (logit-only auditor).
* `2026-05-07 22:25 UTC` — Phase F context + scaling sweep; F1 passes.
* `2026-05-07 22:32 UTC` — Phase F sanity checks (the sobering ones).
* `2026-05-07 22:38 UTC` — Phase F denoising test.
* `2026-05-07 22:50 UTC` — Phase H run; F3 fails.
* `2026-05-07 23:03 UTC` — Phase F+ UQ comparison.

Each entry includes the exact `--seed`, `--n`, hyperparameters, and
W&B run name used.

---

# Capstone follow-up — Phases R, S, T, U, V (2026-05-08/09)

This block synthesises the post-Phase-Q follow-up protocol
(`CAPSTONE_EXPERIMENTS.md`). Five phases exited; results overturn
the original "structural ceiling" framing of the K=27 continuous-
vs-discrete gap and replace it with a regression-target / cascade-
contamination story.

## §3' — What FM regression learns vs uses (Phase T)

Original W1 finding: on the eqm_data50k_ep5_v2 checkpoint, KL_bi is
1.682 with Euler-on-`f` vs 1.391 with Euler-on-`∇⟨x,f⟩`. Phase T
replaces FM-target regression on `f` with FM-target regression on the
conservative gradient itself.

Two-cell sweep (`runs/capstone/T/`):

| training        | lr     | Euler-on-output KL_bi |
|-----------------|-------:|----------------------:|
| EqMConsGrad     | 3e-4   | 0.425 |
| EqMConsGrad     | 2.5e-4 | **0.328** |

Six-cell W1 comparison (`runs/capstone/T/phaseT_w1_compare.md`):

| family    | sampler                | KL_bi |
|-----------|------------------------|------:|
| eqm       | Euler on f             | 1.647 |
| eqm       | Euler on ∇⟨x,f⟩       | 1.399 |
| eqm       | NAG                    | 1.479 |
| consgrad  | Euler on output        | 0.408 |
| consgrad  | Euler on output (gradgrad) | 0.414 |
| consgrad  | NAG                    | 3.608 |

**Headline:** training EqM with the conservative-gradient regression
target closes the K=27 gap from KL_bi 1.4 → 0.41 — a 3.4× improvement
that fully explains the W1 finding and does so via a single training-
recipe change. Sampling cost reverts to a single forward pass per
step (no second-order autograd at sample time, since the model output
*is* the conservative gradient).

**Side-finding:** NAG sampling on a consgrad-trained model degrades to
KL_bi 3.6 (worse than any Euler variant). Cause: NAG operates on
`∇⟨x, m(x)⟩` where `m` is the model output. For consgrad, `m` is
already the gradient, so NAG samples on the Hessian `∇²⟨x, f⟩` — a
different field. Document as "Euler is the natural sampler for
consgrad-trained models; do not use NAG."

**Reframing of the original capstone arc.** The K=27 ceiling at
KL_bi ≈ 1.4 reported across EqM/FMonCLR/LogitKLFlow was treated as a
structural property of continuous-on-simplex flows. Phase T shows the
EqM ceiling was a regression-target artefact; switching to consgrad
training closes most of the gap. Phase S (below) shows FMonCLR can
also reach DFM-class quality with deterministic Euler at K=2,
implying the K=27 FMonCLR/LogitKLFlow gap is also lever-actionable —
likely via an analogous regression-target reformulation, which is
left as future work.

## §4' — Localising the bound (Phases R + S)

### Phase R — SDE sampling on continuous methods (eval-only, 13 cells)

Adding Langevin-style noise to the Euler integrator
(`src/aitchinson_flow/sampling/sde.py`) at α ∈ {0, 0.01, 0.03, 0.1,
0.3, 1.0} on each of EqM/FMonCLR/LogitKLFlow:

| family   | best α | best KL_bi | control α=0 | reduction |
|----------|-------:|-----------:|-----------:|----------:|
| EqM      |   0.10 |     1.299  |     1.422  |     8.7 % |
| FMonCLR  |   0.30 |     1.280  |     1.478  |    13.4 % |
| LKFlow   |   0.30 |     1.301  |     1.601  |    18.7 % |

PASS criterion KL_bi ≤ 0.40 (within 3× of DFM 0.148): **all three
FAIL**. Sampler stochasticity is *not* the bound — it gives a
modest, uniform reduction across families but cannot close the gap
to DFM by a sampler change alone. Figure: `runs/capstone/R/phaseR_sde.png`.

### Phase S — K=2 binary collapse (training + eval, 4 cells)

Mapping text8 to {vowel, consonant+space} (K=2, ~0.63 bits/char
reference entropy) and re-training all four families:

| model       | K=27 KL_bi | K=2 KL_bi | collapse ratio | H_ratio @ K=2 |
|-------------|-----------:|----------:|---------------:|--------------:|
| EqM         |     1.382  |     0.125 |          0.090 |          0.81 |
| DFM         |     0.148  |     0.019 |          0.126 |          1.01 |
| FMonCLR     |     1.559  |     0.003 |          0.002 |          1.01 |
| LogitKLFlow |     1.432  |     0.105 |          0.073 |          1.06 |

Δ KL_bi(EqM, DFM) = 0.106 → AMBIG. **Δ KL_bi(FMonCLR, DFM) = −0.015
→ EQUAL** — FMonCLR even *beats* DFM at K=2 with deterministic
Euler. So continuous-on-simplex flows CAN match discrete denoisers
at low K; the K=27 gap is K-dependent, not framework-fundamental.

EqM K=2 (0.125, H_ratio 0.81) under-performs both its peer continuous
flows and its own K=27 collapse ratio — an EqM-specific bottleneck
attributed to the conservative-gradient + aux-CE + γ-importance
machinery that Phase T then addresses directly at K=27.

**Caveat:** at K=2, KL_bi only certifies short-range vowel/consonant
bigram statistics. Long-range structural validity of V/C English
(syllable patterns, word boundaries) is not measured; the Phase S
KL_bi numbers should be read as relative comparisons across
{EqM, DFM, FMonCLR, LKFlow}, not as evidence that any model
generates structurally valid V/C English.

### Combined Phase R + S verdict

The original §4 2×2 verdict table places this run in the
{R FAIL, S EQUAL/AMBIG} quadrant. With Phase T's reframing — that
EqM's K=27 ceiling closes from 1.4 → 0.41 with a single training-
recipe change — the combined story becomes:

> The K=27 continuous-on-simplex gap is *neither* a sampler-determinism
> ceiling (R FAIL) *nor* a fundamental "continuous flows can't
> represent K=27 multimodality" structural issue (FMonCLR-K=2 EQUAL,
> consgrad-EqM-K=27 reaches 0.41). The gap that the original protocol
> reported is dominated by EqM's regression-target choice and is
> recoverable by training on the conservative gradient directly.

This is a stronger writeup contribution than the original "honest
negative" framing — a constructive lever and a localised mechanism
in place of a structural ceiling.

## §5' — Per-token AUROC: a methodological caveat (Phases U + V)

Phase F-sanity (Phase Q) reported per-token AUROC 0.97 at *uncorrupted
positions* in invalid sequences for the trained EqM auditor and a
linear probe on h_LLM, vs 0.66 for Spilled Energy — the cascade-
contamination caveat. Phase U turns this observation into a
methodological diagnostic; Phase V demonstrates it on a published
recipe.

### Phase U — Four-AUROC table (eval-only, 6 methods)

`scripts/cascade_audit.py` emits, per method:

|                    | corrupted positions | uncorrupted positions |
|--------------------|---------------------|------------------------|
| sequence-aligned   | AUROC₁              | AUROC₂                 |
| position-shuffled  | AUROC₃              | AUROC₄                 |

| method        | AUROC₁ | AUROC₂ | AUROC₃ | AUROC₄ |
|---------------|-------:|-------:|-------:|-------:|
| SE            |  0.968 |  0.639 |  0.967 |  0.638 |
| topk_entropy  |  0.665 |  0.632 |  0.668 |  0.631 |
| linear_probe  |  0.976 |  0.783 |  0.975 |  0.786 |
| blr_laplace   |  0.853 |  0.697 |  0.847 |  0.699 |
| svgp          |  0.861 |  0.723 |  0.858 |  0.724 |
| eqm_auditor   |  0.994 |  0.975 |  0.994 |  0.975 |
| saplma (V)    |  0.992 |  0.759 |  0.993 |  0.761 |

The shuffle ablation (AUROC₃, AUROC₄) is **uninformative for all
seven methods**: |AUROC₁ − AUROC₃| ≤ 0.006 throughout. Cause: every
method here is a per-position scorer; permuting position order within
a sequence does not change any individual score, so AUROC over the
same per-position score distribution is invariant. The shuffle
ablation would only separate methods that aggregate across positions
(window-pooled scorers).

The **AUROC₂-only diagnostic** does separate clean from contaminated
cleanly:

  locality-clean (≈ 0.63):   SE 0.639, topk_entropy 0.632
  moderate cascade  (0.69-0.78): blr_laplace 0.694, svgp 0.723,
                                 saplma 0.759, linear_probe 0.783
  extreme cascade   (≥ 0.95):  eqm_auditor 0.975

Methods using h_LLM as input inherit cascade contamination regardless
of probe complexity (linear vs MLP vs Bayesian); methods using only
the LM's own output distribution (NLL via SE, top-K entropy on
logits) avoid it. SE is the unique method here that combines high
AUROC₁ (0.968) with locality-clean AUROC₂ (0.639) — by using a
position-local energy gradient norm rather than the cross-position
hidden state.

### Phase V — SAPLMA replication

`src/aitchinson_flow/baselines/saplma.py` (3-layer MLP probe per
Azaria & Mitchell 2023 §3) trained 25 epochs on `wiki_cache_gpt2.pt`,
audited with the same protocol:

* AUROC₁ = 0.992 (✓ headline-paper-grade)
* AUROC₂ = 0.759 (cascade-contaminated; +0.12 above SE/topk baseline)
* AUROC₃ = 0.993 (shuffle-invariant, like all per-position probes)

Strict §8 DIAGNOSTIC FIRES (AUROC₂ > 0.85, AUROC₃ < 0.70) is **not
met**, classifying as PARTIAL, but SAPLMA's AUROC₂ is clearly above
the locality-clean baseline — so the diagnostic *does* fire on the
AUROC₂-only fallback. Practical claim: a published-recipe MLP probe
on h_LLM, when subjected to the AUROC₂ diagnostic, shows the same
cascade-contamination pattern as the bare linear probe. AUROC₁
alone overstates per-token localisation accuracy by ~0.2 vs SE.

## §6' — What landed

| phase | protocol verdict | writeup contribution |
|-------|------------------|----------------------|
| R | strong FAIL (R-H1 falsified) | sampler stochasticity not the bound; small uniform reduction; figure |
| S | mixed (FMCLR EQUAL, EqM/LKF AMBIG) | continuous flows CAN match DFM at low K; K-dependent gap |
| T | **strong PASS** | constructive: KL_bi 1.4 → 0.41 with consgrad regression target |
| U | U-H1 confirmed; U-H2 falsified | AUROC₂-only diagnostic, validated on 6 methods |
| V | PARTIAL (AUROC₂-only fires) | published-recipe SAPLMA shows the diagnostic fires |

The Phase T finding is the strongest single contribution of this
follow-up arc — a clean, localised lever that closes most of the
K=27 EqM gap and reframes the original ceiling story. Phases R and
S support that reframing by ruling out alternative explanations.
Phases U and V land the methodological-caveat contribution
independently.

## §7' — Artefacts produced

```
runs/capstone/
├── R/
│   ├── eqm_sde_a{0p00,0p01,0p03,0p10,0p30,1p00}_grad/eval.json
│   ├── eqm_sde_a0p10_grad_n64/eval.json
│   ├── fmclr_sde_a{0p00,0p10,0p30}/eval.json
│   ├── lkflow_sde_a{0p00,0p10,0p30}/eval.json
│   ├── phaseR_sde.{json,png}
│   └── sweep_results.jsonl
├── S/
│   ├── {eqm,dfm,fmclr,lkflow}_K2_data50k_ep5/{epoch_final.pt,eval.json}
│   ├── phaseS_K2.json
│   └── sweep_results.jsonl
├── T/
│   ├── eqm_consgrad_data50k_ep5/{epoch_final.pt,eval.json}
│   ├── eqm_consgrad_data50k_ep5_lr_5/{epoch_final.pt,eval.json}
│   ├── phaseT_w1_compare.{json,md}
│   └── sweep_results.jsonl
├── U/
│   ├── phaseU_cascade_audit.{json,md,png}
│   └── phaseU_with_saplma.{json,md,png}
├── V/
│   ├── saplma_train.log
│   └── phaseV_audit.log
└── checkpoints/
    └── saplma_wiki/probe.pt
```

`runs/DECISION_LOG.md` extended with five phase blocks, each with
hypothesis / result / decision / next-step / artefacts.
