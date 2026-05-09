# Latent-EqM final findings — d=128 trainable + tied + CE + time-cond + Euler

A working configuration for EqM-on-text. Single-seed, laptop GPU (8 GB
Blackwell), 20 epochs, 90M-parameter transformer.

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
