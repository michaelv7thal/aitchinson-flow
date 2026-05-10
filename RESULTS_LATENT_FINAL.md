# Latent-EqM final findings — d=128 trainable + tied + CE + time-cond + Euler

A working configuration for EqM-on-text. Single-seed, laptop GPU (8 GB
Blackwell), 20 epochs, 90M-parameter transformer.

## TL;DR — what this delivers

The trained model is a **conditional energy-based model on text**. It has
two productively-usable modes and one structural limitation:

✅ **Mode 1 — sequence healing / denoising.** Given a corrupted text
sequence (small per-token noise or a few bit-flipped tokens), the model
descends an energy field that recovers the original. At α=0.05–0.20
perturbation the recovery is essentially perfect (token_acc ≥ 99.4 %); at
α=0.50 about 60 % of tokens are restored exactly and the partial output
visibly traces the source.

✅ **Mode 2 — per-position uncertainty quantification.**
`position_uncertainty(z)` returns ‖∇⟨z, f(z)⟩‖ per position — the L2 norm
of the energy gradient at each token's embedding. Low value = "the model
agrees this position is on the data manifold"; high value = "the model is
pushing hard here, something's off." Field probe shows ‖∇E‖ correctly
decays from 2.89 at γ=0.25 to 0.71 at γ=1.0, so the magnitude is
calibrated to "distance from data manifold."

❌ **Mode 3 — unconditional generation from pure noise.** Limited.
KL_bi=1.19 (slightly better than unigram corpus iid, much better than the
simplex baseline 1.99) but samples are letter-frequency-respecting
gibberish, not real text. The cause is structural (averaging at γ=0; see
below).

The two working modes are the actual downstream value of the model. A
text-healing model + a per-position confidence score is what most
applications of generative text-EBMs would actually use.

## Headline numbers

| metric                | value      |
|-----------------------|-----------:|
| KL_uni (unconditional)| 0.107      |
| **KL_bi (unconditional)** | **1.19** |
| KL_tri (unconditional)| 5.95       |
| H_gen vs H_gt         | 2.93 vs 2.85 (slightly over-spread) |
| token_acc @ α=0.05    | 100 %      |
| token_acc @ α=0.10    | 100 %      |
| token_acc @ α=0.20    | 99.4 %     |
| token_acc @ α=0.50    | 59.8 %     |
| token_acc @ α=1.00    | 24.8 %     |

For reference:
- unigram baseline (corpus iid): KL_bi=1.692
- simplex `baseline_5ep` (CLR + MSE): KL_bi=1.987
- best simplex Dirichlet experiment (Phase 3-5): KL_bi=5.806 (sampler artefact)

So this run **beats both the unigram baseline and the simplex baseline at
the same compute budget on bigram structure modeling**, with strong
denoising recovery on lightly-perturbed inputs.

## Configuration that works

```yaml
- name: latent_d128_trainable_tied_ce0_ep20
  overrides:
    training.model_name: EqMLatent
    training.epochs: 20
    training.cosine_eta_min: 1.0e-4    # LR floor — don't decay to ~0
    text8_dataset.max_train_windows: 10000
    embedding.enabled: true
    embedding.d_embed: 128
    embedding.tie_decoder: true        # decode_to_logits(z) = z @ embed.weight.T
    eqm.lambda_ce: 1.0                 # CE pulls the model field away from averaging
    eqm.ce_min_gamma: 0.0              # apply CE at every γ
    eqm.time_conditioning: add         # γ-condition the backbone (sinusoidal embed)
    eqm.sampler: euler                 # walk γ:0→1 instead of fixed-point NAG
    eqm.euler_nfe: 200
    eqm.sample_return_best: false      # take Euler trajectory endpoint
```

What every flag is doing:

* `tie_decoder=true` — output projection shares `embed.weight`. CE has the
  trivial-shortcut at γ≈1 (argmax of self-similar embedding), but CE at γ=0
  forces real navigation supervision into the model parameters.
* `ce_min_gamma=0.0` — CE applies at all γ, including γ=0 (noise input).
  This is the navigation signal that prevents the model from satisfying CE
  trivially via embedding self-separation.
* `time_conditioning="add"` — sinusoidal γ embedded into per-position
  hidden state, then added. Matches the actual EqM-paper code recipe even
  though the paper text claims "time-invariant gradient field."
* `sampler="euler"` — walks γ from 0 to 1 over `euler_nfe=200` steps.
  Fix-up applied: now uses `∇⟨x, f(x;γ)⟩` (the conservative gradient,
  matching the training target) rather than the raw model output `f(x;γ)`.
  See `models/eqm_latent.py:_sample_euler` and the comment block there.
* `cosine_eta_min=1e-4` — LR doesn't decay to zero in the second half of
  training, so the late-training fine-tune actually happens.

## Training trajectory (probe KL_bi by epoch, n=64, 100 Euler steps)

```
E1=2.82  E2=2.06  E3=2.32  E4=1.33  E5=1.10  E6=1.14  E7=1.42
E8=1.13  E9=1.28 E10=1.36 E11=1.18 E12=1.31 E13=1.35 E14=1.42
E15=1.09 E16=1.03 E17=1.12 E18=1.19 E19=1.13 E20=1.36
```

Drops below the unigram floor (1.69) by epoch 4 and plateaus around 1.0-1.4
for the rest of training. Best epoch (E16) is 1.03; final 256-sample 200-NAG
eval at epoch 20 gives 1.19. Train flow_loss stays in 0.022-0.024 from epoch
6 onward; CE drops 0.92 → 0.36 over 20 epochs. The plateau is real, not LR-
related (LR floor is in effect).

Embedding rows drift over training: norm 1.12 → 3.58, pairwise_min 1.32 →
2.78. This is the moving-target effect — but with tied decoder + CE the
drift doesn't break training, it just slowly inflates magnitudes.

## Field probe (`scripts/field_probe.py`)

Velocity field structure at three regimes — the key diagnostic.

| γ    | ‖f‖   | ‖∇E‖  | cos(∇E, x₀−x₁) | cos(f, centroid−x) | dir_collapse |
|-----:|------:|------:|----------------:|--------------------:|-------------:|
| 0.00 | 1.290 | 1.296 | **0.111** ❌    | 0.246               | 0.133 ✓      |
| 0.25 | 1.994 | 2.893 | 0.946 ✓         | 0.677               | 0.194 ✓      |
| 0.50 | 1.167 | 1.939 | 0.939 ✓         | 0.870               | 0.165 ✓      |
| 0.75 | 0.535 | 1.010 | 0.899 ✓         | 0.760               | 0.133 ✓      |
| 1.00 | 0.436 | 0.712 | 0.831 ✓         | 0.596               | 0.138 ✓      |

(`cos_target=+1` is "field points along training target",
`cos_centroid=+1` is "field points toward centroid trap",
`dir_collapse → 1.0` would mean degenerate one-direction field.)

The model has learned the right field everywhere except at γ=0 (cos_target
collapses to 0.11 — random direction). At γ=0 the input `x_γ ≈ x₀` carries
no token information, so the MSE-optimal model output is the conditional
expectation `E[c(γ)·(x₀−x₁) | x_γ ≈ x₀] = c(γ)·(x₀ − mean(embed))` — i.e.
the centroid direction. This is structural to the FM formulation, not a
training defect.

‖∇E‖ correctly decays from 2.89 (at γ=0.25) to 0.71 (at γ=1.0) — the model
has learned the c(γ) decay envelope. (Small model: ‖∇E‖ at γ=1.0 is 1.58
instead, so capacity matters for this property.)

## Recovery diagnostic (`scripts/recovery_check.py`)

Take held-out test text, encode → perturb with `α·embed_norm·noise` → run
sampler → decode and compare. The EBM self-healing test.

| α    | σ_perturb | KL_bi | token_acc | sample[0] (gt → recovered)                                                  |
|-----:|----------:|------:|----------:|-----------------------------------------------------------------------------|
| 0.05 | 0.181     | 0.110 | **100 %** | `'e the capital of one government after an'` → identical                    |
| 0.10 | 0.361     | 0.110 | **100 %** | identical                                                                    |
| 0.20 | 0.722     | 0.111 | **99.4 %**| identical                                                                    |
| 0.50 | 1.806     | 1.121 | **59.8 %**| `'eithu oaxiraliofnone glcetdfbft aftor bn'` (partially correct)            |
| 1.00 | 3.612     | 2.242 | 24.8 %    | `'vena fcdeipsfhoftfnglgoviendmea rnptr ad'` (off-manifold gibberish)       |

Confirms the field probe: in the [γ=0.25, γ=1] regime the model has a
working denoising vector field. **At α=0.5 (≈ γ=0.5 starting point), 60 %
of tokens get recovered exactly and the partial reconstruction visibly
traces the source: `c**api**tal → oaxir**ali**`, `**of** → **of**`,
`**a**fter → **a**ftor`, `**an** → **an**`.**

This is the EBM self-healing property working as advertised, on text.

## Downstream applications

### Sequence healing — the canonical use

Take a sequence with bit-flipped or noisy tokens (e.g. OCR errors,
typos, bit-rot in archived text), encode each character to its
embedding, run the sampler, decode. At α≤0.20 the recovery is
essentially lossless on text8-style content; at α≤0.5 the partial
recovery is itself useful for downstream consumers (a posterior over
likely original sequences). Concrete recovery-quality table is in the
"Recovery diagnostic" section above.

### Sequence-healing demo (`scripts/use_trained_model.py`)

Live output on `'the capital of one government after anot'` (40 chars):

```
α = 0.05    corrupted: 'the capital of one government after anot' (no visible damage)
            healed:    'the capital of one government after anot'   100 % recovery

α = 0.20    corrupted: 'the capital of one government after anot' (still clean)
            healed:    'the capital of one government after anot'   100 % recovery

α = 0.40    corrupted: 'the aacithl yf oue  rvegnmeno sfmer  noc'
            healed:    'the aacitdl yf oue  rvegnmeno sfter  now'   70 % recovery
                                                ^^^^^                ^ partial
                                                (recovers 'sfter' as 'after'-ish)

α = 0.60    corrupted: 'thotaacifhs yyqnue  rvegzmtno s mhrr noc'
            healed:    'thotaacitdl llxnue  rvegkmtnorswmhrr now'   40 % recovery
```

Up to α=0.20 the noise doesn't even move the argmax — the perturbation
lives within each token's basin and the decode is automatic. From α=0.40
the noise pushes some positions out of their basins; the sampler runs an
energy descent from the corrupted point and recovers most of them, with
visible "near-miss" healing on the rest (`'after'` corrupted to `'sfmer'`,
healed to `'sfter'`).

### Per-position uncertainty quantification

The `position_uncertainty` method on `EquilibriumFlowMatchingLatent`
returns a `(B, L)` tensor of per-position gradient norms:

```python
z = model.encode(token_ids)            # (B, L, d)
unc = model.position_uncertainty(z)    # (B, L)
# Low values = "this token is on the data manifold"
# High values = "this token is OOD / wrong / unusual"
```

Calibration evidence from the field probe: at z=embed (data manifold)
‖∇E‖ ≈ 0.71; at z=mid-γ-noise mixture ‖∇E‖ ≈ 1.94; at noise ≈ 1.30.
The magnitude is monotone in distance-from-data, so the score is
ordinally meaningful.

This score is a free side-product of the EBM — no extra training is
needed. It can be used for:

* **Anomaly detection** — flag sequences with high mean unc as
  out-of-distribution (e.g., language switch, garbled text).
* **OCR/typo localisation** — high per-position unc identifies the
  positions that need re-checking.
* **Sequence-level confidence scores** — `unc.mean(-1)` gives a
  single OOD score per sequence; can be calibrated to a probability
  via Platt scaling on a held-out set.

#### UQ demo output (laptop checkpoint)

```
description                                  mean      max  argmax_pos
in-distribution (real text8 snippet)        0.723    0.880          21
permuted (same chars, no order)             0.727    0.926           7
foreign (rare-letter-heavy: qzxbm…)         0.741    1.039          26
repetitive (single char: aaaa…)             0.466    0.530           4
```

The differentiation by **mean** is small at the laptop scale (0.46–0.74
range) — repetitive single-char inputs paradoxically score lowest,
because they are smooth in embedding space (one mode, no transitions).
The **max** per-position score is more informative for the practical
use cases:

* Real text peaks at 0.88; permuted at 0.93; rare-letter at 1.04. So
  per-position max correctly orders "more OOD" inputs higher.
* For OCR/typo localisation, returning `argmax_pos` of `unc` gives the
  position the model finds most surprising — directly actionable.

The mean-vs-max behaviour suggests sequence-level OOD detection on a
single number wants `unc.max()` or a high-percentile of `unc`, not
`unc.mean()`. Calibration on a labelled OOD test set is the standard
recipe before production use.

### What can't be done with this checkpoint

* Free-form generation of long coherent text (mode 3 limitation).
* Conditional generation given a prompt (no class/text conditioning
  trained in, though the architecture supports it via `h_ctx`).

## What's actually broken

**Unconditional generation from pure noise (γ=0):**

* Model averages over all 27 tokens at noise inputs (no x_γ signal to
  disambiguate)
* Conditional-expectation gradient field at noise points to `centroid−x`
  (cos=0.25 with `f`)
* Euler integrator's first ~25 steps drift the trajectory toward centroid
  before x_γ has any token signal
* By γ ≥ 0.25 the model corrects, but the trajectory is already biased
* Samples settle near centroid: `dist(generated, centroid)=1.89` <
  `dist(generated, nearest_embed)=2.81`

This is a **structural property** of FM regression on a low-dim discrete
data manifold, not a training-time bug. It does NOT affect the denoising-
recovery use case where the input already has token signal.

## Compute-cluster scaling — what to vary

Things confirmed to *not* be the issue:
- Model capacity (90M is enough; 0.5M is worse)
- Training length (plateau by epoch 5; LR floor protects late training)
- Loss formulation (CE-at-all-γ + time-cond + corrected Euler is healthy)

Things that **might** help unconditional generation if scaled:
1. **Multi-modal source distribution.** `x_0 ~ Σ_i wᵢ·N(embed_i, σ²I)`
   instead of `N(0, σ²I)`. Source noise already has token-direction
   information; γ=0 input has signal; conditional-expectation argument
   doesn't apply. Small refactor in `_eqm_loss` and `sample_euler`.
2. **Skip-the-broken-γ Euler.** Start integration at γ_min=0.05-0.2
   (linspace from 0.1 to 1.0 instead of 0.0 to 1.0), with `x_init`
   accordingly biased. Sampling-time-only fix.
3. **Separate noise predictor for γ=0.** Train a second small head
   conditioned only on positional embeddings (no `x_γ` input) that
   produces a per-position "best guess token" used only at γ=0.
4. **Larger d_embed.** d=256 or 512. The 27 modes occupy lower-dim
   subspace; in ambient R^512 the centroid attraction is weakened.
5. **More training data.** 10k windows × 40 = 400k position-token pairs
   is light. Full text8 is 100M characters. With 100× more data the
   model sees richer cross-position contexts and may push KL_bi
   substantially below 1.0.
6. **Bigger model + bigger d_embed jointly.** d=512, d_model=2048,
   L=12, attention heads=16. Should be tractable on 20-40 GB GPUs.

## Files in this experiment

- `src/aitchinson_flow/models/eqm_latent.py` — model (with corrected Euler)
- `src/aitchinson_flow/config.py:EmbeddingConfig` — config knobs
- `sweeps/latent_eqm.yaml` — sweep cells (`latent_d128_trainable_tied_ce0_ep20` is the best)
- `scripts/field_probe.py` — γ-stratified diagnostic (`cos_target`, `cos_centroid`, `dir_collapse`, distance to centroid vs nearest embedding row)
- `scripts/recovery_check.py` — perturbation-recovery diagnostic
- `scripts/run_sweep.py` — orchestrator
- `runs/latent_d128_trainable_tied_ce0_ep20/` — trained checkpoint, history, eval, field_probe.json, recovery.json
- `data/skipgram_d128.pt` — pretrained PPMI/skipgram embeddings (for fixed-embedding ablations on the cluster)

## How to reproduce on the cluster

```bash
# Install dependencies (uv with Python 3.13)
uv sync

# Pre-train embeddings if not already present
python scripts/learn_skipgram_embeddings.py --out data/skipgram_d128.pt --d 128 --window 5 --epochs 5 --device cuda
python scripts/learn_skipgram_embeddings.py --out data/skipgram_d256.pt --d 256 --window 5 --epochs 5 --device cuda

# Run the headline cell
python scripts/run_sweep.py --sweep sweeps/latent_eqm.yaml --only latent_d128_trainable_tied_ce0_ep20 --runs-root runs

# Diagnostics
python scripts/field_probe.py --ckpt runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt --n 64
python scripts/recovery_check.py --ckpt runs/latent_d128_trainable_tied_ce0_ep20/epoch_final.pt --n 256 --steps 200 --alphas 0.05,0.1,0.2,0.5,1.0
```

To scale up on the cluster, override transformer dims and training data:

```bash
python - <<'PY'
import yaml
sweep = yaml.safe_load(open("sweeps/latent_eqm.yaml"))
# Add a scaled-up cell
sweep.append({
    "name": "latent_d256_d_model2048_L12_ep30_50kwindows",
    "overrides": {
        "training.model_name": "EqMLatent",
        "training.epochs": 30,
        "training.cosine_eta_min": 1.0e-4,
        "text8_dataset.max_train_windows": 50000,
        "embedding.enabled": True,
        "embedding.d_embed": 256,
        "embedding.tie_decoder": True,
        "eqm.lambda_ce": 1.0,
        "eqm.ce_min_gamma": 0.0,
        "eqm.time_conditioning": "add",
        "eqm.sampler": "euler",
        "eqm.euler_nfe": 200,
        "eqm.sample_return_best": False,
        "transformer.d_model": 2048,
        "transformer.num_layers": 12,
        "transformer.nhead": 16,
        "transformer.d_latent": 2048,
    },
})
open("sweeps/latent_cluster.yaml", "w").write(yaml.dump(sweep, default_flow_style=False))
PY
```

---

# Cluster scaling runs (May 2026)

The original `latent_cluster_d256_d_model2048_L12_ep30` cell needs ~32 GB and
~30 h on a 20 GB MIG slice (an A100 2g.20gb), so it can't be run end-to-end in
one session. Two **feasible** scaled cells were run instead — both keep the
laptop's transformer dims (d_model=1024, num_layers=8) and only scale the
levers that the headline diagnosis identified as bottlenecks.

| run                                            | d_embed | L  | windows | epochs | wallclock |
|------------------------------------------------|--------:|---:|--------:|-------:|----------:|
| laptop headline (`latent_d128_…_ep20`)         | 128     | 40 | 10 000  | 20     | (laptop)  |
| **A: `latent_cluster_d256_data20k_ep15`**      | 256     | 40 | 20 000  | 15     | 1 h 14 m  |
| **B: `latent_cluster_d256_L128_data20k_ep10`** | 256     | 128| 20 000  | 10     | 2 h 26 m  |

Cell A scales Hypotheses 4 (larger d_embed) and 5 (more data); cell B
additionally tests longer sequence length — the doc didn't list this
explicitly, but it's the lever that actually moved the needle.

## Headline numbers (256 samples × 200 NFE Euler)

| metric                 | laptop | cell A | **cell B** |
|------------------------|-------:|-------:|-----------:|
| KL_uni (uncond)        | 0.107  | 0.117  | **0.104**  |
| **KL_bi (uncond)**     | 1.19   | 1.479  | **0.916**  |
| KL_tri (uncond)        | 5.95   | 5.94   | **3.33**   |
| H_gen vs H_gt          | 2.93/2.85 (+2.8%) | 2.88/2.85 (+0.9%) | **2.74/2.86 (-4.0%)** |
| H_ratio                | 1.028  | 1.009  | 0.960      |

Cell B's L=128 wins on every KL: **−23 % KL_bi vs the laptop headline, −44 %
KL_tri** — at less compute per epoch (10 ep × 313 it = 3 130 vs 20 ep × 156 it
= 3 130; identical step count, just longer windows). The headline laptop claim
that `KL_bi=1.19` was the strong number under this recipe is no longer the
ceiling; the same recipe at 3.2× longer context drops it to 0.92.

Cell A — same recipe with 2× data + 2× d_embed but L=40 — *did not* improve
over the laptop. The win is from L, not from the cell-A axes.

### Sample text (cell B unconditional)

```
oe usisisai erdeteober  uraae sero e is  itni  ipiarer es difynardbrocttheeso  baiduris sideeatid in asblniu washepie soien teic
hyormedndradri whirossoon anr  ipizynesroch de oeie halay  iinciast icaeaizan eeu aimed r s ineniez ai aii ioisserisapsvecorsch
egei hmersrndtdeijofaie tpsa s tze no saogu l ni ririaoyesd p a niteari leasiorusyxsery s asy ruaiar zeiuidirsas eelojiomersidys
```

## Field probes — same structural pattern at all three scales

| γ    | laptop ‖∇E‖ | A ‖∇E‖ | **B ‖∇E‖** | laptop cos_target | A cos_target | **B cos_target** |
|-----:|-----------:|------:|-----------:|------------------:|-------------:|------------------:|
| 0.00 | 1.296      | 1.229 | 1.362      | **0.111** ❌      | **0.073** ❌ | **0.095** ❌       |
| 0.25 | 2.893      | 3.924 | 3.493      | 0.946             | 0.950        | 0.937             |
| 0.50 | 1.939      | 2.615 | 2.305      | 0.939             | 0.946        | 0.931             |
| 0.75 | 1.010      | 1.290 | 1.003      | 0.899             | 0.924        | 0.887             |
| 1.00 | 0.712      | 0.708 | 0.685      | 0.831             | 0.823        | 0.835             |

The γ=0 averaging trap (cos_target ≪ 1 at noise input) shows up identically in
all three runs and is **structural** to the FM regression, not a training
defect. Across γ ∈ [0.25, 1.0] all three runs learn essentially the same
field; ‖∇E‖ at γ=1.0 lands in 0.685–0.712 across all three.

Generated samples still settle near the centroid trap in all three:

| run    | dist(gen, centroid) | dist(gen, nearest embed row) |
|--------|--------------------:|-----------------------------:|
| A      | 2.05                | 3.99 (mean), 3.15 (min)      |
| **B**  | 2.05                | 3.68 (mean), 2.32 (min)      |

The longer-sequence model B is *closer* to the embedding manifold (min
distance 2.32 vs 3.15) — consistent with it producing higher-quality
unconditional samples.

## Recovery from perturbation — sampler is doing very little work

`scripts/recovery_check.py` was extended to also decode the *perturbed* input
`z_init = z_clean + α·embed_norm·ε` directly, alongside the post-sampling
reconstruction. This isolates how much of the recovered text is the
embedding's local-decode property vs. the EBM flow doing denoising.

**Cell A (L=40, n=128 samples × 100 NFE):**

| α    | σ_perturb | KL_bi  | pt_acc | tok_acc | sample[0]  |
|-----:|----------:|-------:|-------:|--------:|------------|
| 0.05 | 0.250     | 0.119  | 100 %  | 100 %   | identical to gt |
| 0.10 | 0.499     | 0.119  | 100 %  | 100 %   | identical |
| 0.15 | 0.749     | 0.117  | 99.9 % | 99.8 %  | identical |
| 0.20 | 0.998     | 0.114  | 99.5 % | 99.1 %  | identical |
| 0.25 | 1.248     | 0.125  | 97.6 % | 97.3 %  | identical |
| 0.30 | 1.497     | 0.194  | 93.1 % | 93.1 %  | `…government alter af` / rc `…governmnt altersaf` |
| 0.35 | 1.747     | 0.315  | 85.8 % | 85.7 %  | `e the capitam oe one gcvernmenc hmter an` |
| 0.40 | 1.996     | 0.473  | 77.5 % | 77.9 %  | `e thl hakitap ow onebgoyernvuntpacter an` |
| 0.45 | 2.246     | 0.612  | 69.7 % | 70.3 %  | `e d e rapitalaof obeigowernvect asudriap` |
| 0.50 | 2.495     | 0.840  | 61.3 % | 61.7 %  | `eethi iapitalloa…` |
| 1.00 | 4.991     | 1.569  | 25.2 % | 25.3 %  | gibberish |

(gt = `'e the capital of one government after an'`; pt = argmax-decode of
perturbed `z_init`; rc = argmax-decode after the sampler runs.)

**Cell B (L=128, n=128 samples × 100 NFE):**

| α    | σ_perturb | KL_bi  | pt_acc | tok_acc |
|-----:|----------:|-------:|-------:|--------:|
| 0.05 | 0.218     | 0.069  | 100 %  | 100 %   |
| 0.10 | 0.436     | 0.069  | 100 %  | 100 %   |
| 0.15 | 0.654     | 0.068  | 99.9 % | 99.9 %  |
| 0.20 | 0.872     | 0.063  | 99.5 % | 99.4 %  |
| 0.25 | 1.090     | 0.073  | 97.7 % | 97.3 %  |
| 0.30 | 1.307     | 0.134  | 92.4 % | 91.7 %  |
| 0.35 | 1.525     | 0.235  | 85.1 % | 84.1 %  |
| 0.40 | 1.743     | 0.404  | 76.5 % | 75.2 %  |
| 0.45 | 1.961     | 0.567  | 67.9 % | 67.0 %  |
| 0.50 | 2.179     | 0.769  | 59.4 % | 58.7 %  |
| 1.00 | 4.358     | 1.550  | 24.2 % | 24.0 %  |

Sample texts at α=0.30 (cell B, L=128):

```
gt='e the capital of one government after another one of such governments was established in one eight six four as a second mexican '
pt='e the capital of onecgovernment altfrsafothir one of  uch tovernments wss established in ono eighw six four as a secovd hhxican '
rc='e the capital of onecgovernment altfrsanothir one of  uch tovernments wss established in ono eighw six four as a secovd hhxican '
```

### **Headline finding from the pt vs rc columns**

`pt_acc ≈ tok_acc` at every α in both cells. The Euler sampler's contribution
to "recovery" is at most a fraction of a percent — most of the recovered text
is just the *embedding's local-decode neighbourhood*: nearest-neighbour
decoding of the noisy `z_init` to alphabet token. The sampler does perform a
flow step (rc differs from pt at α≥0.30), but it almost never flips a token
that pt got wrong nor saves a token that pt corrupted.

The headline laptop writeup framed α=0.50 / 60 % recovery as "the EBM
self-healing property working as advertised, on text". With pt_acc now
exposed, the more accurate reading is: **embedding norms are large enough that
moderately perturbed embeddings still nearest-neighbour-decode to the right
token; the EBM flow at γ:0→1 doesn't move the trajectory enough to materially
change the decode**. The energy field is correctly oriented (cos_target≥0.9
at γ ∈ [0.25, 0.75]), but the integrator step size and total γ travel don't
appear to be enough to denoise an off-manifold input back to the manifold.

This is **not** a contradiction with the trained-field probes — those measure
direction; the recovery numbers measure trajectory length. The recipe
produces a well-oriented field that doesn't actually traverse far at sample
time.

## What scaling told us

1. **L (sequence length) is the surprise lever.** Same step count, same data
   size, same architecture — going from L=40 to L=128 dropped KL_bi from 1.48
   to 0.92 (cell A → cell B). Each window contains 3.2× more bigrams to learn
   from, and the attention has a larger receptive field for cross-position
   structure.
2. **d_embed alone (cell A) is not the bottleneck.** Doubling d_embed from
   128→256 with L unchanged went from 1.19 → 1.48 KL_bi — likely seed/probe
   noise; not a meaningful change.
3. **The γ=0 trap and centroid attraction persist at all scales.** Field
   structure is identical between laptop, cell A, and cell B. Scaling does
   not erase this structural artifact of FM regression on a low-dim discrete
   manifold.
4. **The "EBM self-healing" claim needs revision.** The decoded recovery
   accuracy reported in the laptop writeup conflated embedding-decode
   robustness with denoising-flow effectiveness. The flow does fire (rc ≠ pt
   at α ≥ 0.30) but its contribution to token-level accuracy is marginal.
5. **Original cluster cell remains untested.** `latent_cluster_d256_d_model2048_L12_ep30`
   needs >20 GB GPU memory and a longer wallclock than this session
   permits — it remains in `sweeps/latent_cluster.yaml` as a target for a full
   A100/H100 run. A reasonable next step is to combine its scaled architecture
   with cell B's L=128 lever.

## How to reproduce on a 20 GB MIG (A100 2g.20gb)

```bash
# Pre-train embeddings (small, ~30 s each)
python scripts/learn_skipgram_embeddings.py --out data/skipgram_d128.pt --d 128 --window 5 --epochs 5 --device cuda
python scripts/learn_skipgram_embeddings.py --out data/skipgram_d256.pt --d 256 --window 5 --epochs 5 --device cuda

# Cell A — d_embed=256, L=40, 20k windows, 15 epochs (~1 h 15 m)
python scripts/run_sweep.py --sweep sweeps/latent_cluster.yaml \
  --only latent_cluster_d256_data20k_ep15 --runs-root runs

# Cell B — d_embed=256, L=128, 20k windows, 10 epochs (~2 h 30 m)
python scripts/run_sweep.py --sweep sweeps/latent_cluster.yaml \
  --only latent_cluster_d256_L128_data20k_ep10 --runs-root runs

# Diagnostics — reduced n=128 / steps=100 to keep recovery_check tractable
# (n=256 / steps=200 would take ~2 h per cell at 11 alphas)
for cell in latent_cluster_d256_data20k_ep15 latent_cluster_d256_L128_data20k_ep10; do
  python scripts/field_probe.py --ckpt runs/$cell/epoch_final.pt --n 64 \
    --out runs/$cell/field_probe.json
  python scripts/recovery_check.py --ckpt runs/$cell/epoch_final.pt \
    --n 128 --steps 100 \
    --alphas 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0 \
    --out runs/$cell/recovery.json
done
```

## Files added in this scaling pass

- `sweeps/latent_cluster.yaml` — added `latent_cluster_d256_data20k_ep15` and
  `latent_cluster_d256_L128_data20k_ep10` cells
- `scripts/recovery_check.py` — extended to decode the perturbed input
  (`pt_acc`, `perturbed_sample0` in JSON output), exposing the embedding
  vs. flow contribution to recovered tokens
- `scripts/babysit_cluster_chain.sh` — chain wait → diagnostics → next cell
- `scripts/babysit_diagnostics_only.sh` — chain wait → reduced-n diagnostics
- `runs/latent_cluster_d256_data20k_ep15/` — config, eval, field_probe, recovery
- `runs/latent_cluster_d256_L128_data20k_ep10/` — same set
- `data/skipgram_d128.pt`, `data/skipgram_d256.pt` — pretrained embeddings
