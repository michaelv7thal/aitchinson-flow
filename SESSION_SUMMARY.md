# Aitchison Flow — Session Summary

## 1. Starting state

**Goal**: train a continuous flow-matching model on text8 (K=27, L=40) so that
unconditional generation produces character distributions resembling English.

**Symptoms**:
- With Hilbert / soft-Hilbert loss, training stalled at small γ (~0.5–1.2 MSE)
  and inference output collapsed to 2–3 characters per sequence.
- After swapping to MSE the optimization stabilised but inference collapsed to
  ~85 % space (`' '`), text8's most frequent unigram. Generated text was rows
  of `· · · ·`. KL(gen ‖ gt) ≈ 0.42 nats dominated by that single peak.
- Reconstruction BPD (perturb GT → run NAG-GD) was effectively zero, so the
  model had learned the data manifold's local minima — only **unconditional
  sampling from noise** was broken.

## 2. Diagnosis

### 2.1 Why the Hilbert loss broke training

`hilbert_distance(x, y) = max_i(d_i) − min_i(d_i)` only depends on the argmax
and argmin of the residual. Its (sub)gradient is therefore non-zero on exactly
**2 of K=27 coordinates** per token. Soft-Hilbert at α=5 with CLR residuals of
magnitude ~12 (label-smoothed log-1-hot has a single +12 peak) collapses LSE
back to a hard max, so the gradient stays effectively 2-sparse. Hilbert is the
right *metric* on the simplex (Nielsen & Sun's k-means / NN application) but a
weak L∞-like *regression objective*.

### 2.2 Why MSE alone produced "all space"

Conditional flow matching trains the model to predict
`E[c(γ)·(x0 − x1) | x_γ]`. At γ → 0 the input `x_γ ≈ x0` carries **no
information** about `x1`, so the variance-minimising regressor is the
**marginal** target `−E[x1]` — i.e. the unigram log-prob direction. NAG-GD
descent from noise then descends straight into the deepest mode of that field,
which for text8 is `' '` (~17 % of the corpus). Dense gradients fixed the
optimisation but didn't change *what minimum* the energy field had at
unconditional initialisation.

### 2.3 Other findings during the comprehensive review

| # | Bug / gap | Where | Impact |
|---|-----------|-------|--------|
| A | Sampler descended raw `f(x)` instead of `∇⟨x, f(x)⟩` | `eqm.py:_compute_grad` | training & inference using different vector fields |
| B | Train σ=0.1, sample σ=0.01 (10× mismatch) | `eqm.py:45` vs `eqm.py:161` | inference x0 OOD; deterministic NAG-GD → identical outputs |
| C | No CE / classification anchor at γ ≈ 1 | `_eqm_loss` | energy field has only marginal-mode minima |
| D | Uniform γ ~ U(0,1) | `eqm.py:49` | irreducible-variance noise endpoint dominates loss |
| E | `eval_plots.py` and `compare_gt_vs_gen.py` fed `(B,L,K-1)` ILR tensors into a `(B,L,K)` CLR model | both scripts | scripts un-runnable since the 118d572 refactor |
| F | `main.py:_set_config` builds `set_cfg` (with `model_name="bayesian_auditor_stage1"`) and discards it | `main.py:78-83` | every CLI flag except `--out-dir` was a no-op |
| G | `lambda_vol`, `lambda_mse`, `alpha`, `volume_penalty` configured but never wired | `eqm.py`, `geometry.py` | half-implemented features, easy to misread |
| H | `lr=1e-5` for a 100M-param Transformer; `_clamp_unit_open` checks `if not (0.0, x <= 1.0)` (always False); scheduler builder silently returns None for any name except `cosine` | `optim.py` | slow training, broken validation, undocumented gaps |
| I | Argmax-only decoding | eval scripts | even diverse logits → identical text |
| J | No coverage / unigram-KL during training | `runner.py` | mode collapse invisible in loss curves |
| K | `--smoke` / `--lm-key` / `--top-k` / `--seq-length` / `--char-window-length` / `--corrupt-rate` / `--no-renormalize` parsed but unread | `main.py` | dead code, misleading help |
| L | Final checkpoint saved optimizer state (~830 MB → ~385 MB after fix) | `runner.py:148` | wasted disk |
| M | Reconstruction BPD always ≈ 0 from perturbed GT | `eqm.py:bpd` + `evaluate_eqm.py:A` | tautological metric, not a model-quality signal |

The string `bayesian_auditor_stage1` in `main.py:61` is fossil from commit
`118d572` ("Refactor code structure for improved readability and
maintainability"), which deleted the entire Bayesian-auditor pipeline
(`bayesian_auditor*.py`, `bayesian_generator.py`, `equilibrium.py`, `benchmarks/`,
`ARCHITECTURE.md`, etc.) and added the new `eqm.py` / `dfm.py`. The model class
no longer exists; if `set_cfg` were ever consumed `build_model` would raise.

## 3. Fixes applied (all 11 implemented this session)

| # | Change | File |
|---|--------|------|
| 1 | `_compute_grad` now computes `∇_x ⟨x, f(x)⟩` via autograd at sample time | `eqm.py` |
| 2 | Single `cfg.eqm.source_sigma = 0.1` used by both training and `model.sample()` | `eqm.py`, `config.py` |
| 3 | Aux CE loss on the implied-`x1` reconstruction (`x1 ≈ x_γ − λ·grad_g` for linear decay) at γ ≥ `ce_min_gamma=0.5`, weight `lambda_ce=0.5` | `eqm.py`, `config.py` |
| 4 | `gamma = U(0,1)**gamma_power` with `gamma_power=0.5` (Beta-style upweight near γ=1) | `eqm.py`, `config.py` |
| 5 | `_unigram_kl_probe` runs after each epoch, samples 64 sequences via `model.sample`, logs KL / H_gen / H_gt | `runner.py`, `config.py` |
| 6 | `evaluate_eqm.py` panel G uses `top-k` decode (k=3, T=1.0); also rewritten samplers to use the conservative field | `scripts/evaluate_eqm.py` |
| 7 | `lr=3e-4`, `grad_clip_norm=1.0` | `config.py` |
| 8 | `main.py` reduced to a real entry point: `--out-dir`, `--epochs`, `--lr`. Dead `_set_config` and unread flags removed | `main.py` |
| 9 | `eval_plots.py` and `compare_gt_vs_gen.py` rewritten in CLR (K-dim) throughout; ILR used only for distance reporting | `scripts/eval_plots.py`, `scripts/compare_gt_vs_gen.py` |
| 10 | `_clamp_unit_open` condition fixed; `build_scheduler` raises on unknown name | `optim.py` |
| 11 | Final checkpoint saved with `optimizer=None` (385 MB instead of 830 MB) | `runner.py` |

`cfg.loss.mode = "mse"` was kept from the prior step.

## 4. Results — same 5-epoch budget, all-fixes config

The training-loop probe samples 64 sequences via `model.sample()` (the fixed
NAG-GD on the conservative field) at the end of every epoch.

| epoch | train | flow_loss | ce | g<.33 | g<.66 | g<1 | unigram_kl | H_gen | H_gt |
|------:|------:|----------:|----:|------:|------:|----:|----------:|------:|------:|
| 1 | 0.2106 | 0.2103 | 0.0006 | 0.691 | 0.188 | 0.120 | **0.802** | 2.595 | 2.848 |
| 2 | 0.0165 | 0.0165 | 0.0001 | 0.106 | 0.007 | 0.006 | **0.377** | 2.366 | 2.848 |
| 3 | 0.0153 | 0.0153 | 0.0001 | 0.115 | 0.005 | 0.003 | **0.211** | 2.802 | 2.848 |
| 4 | 0.0064 | 0.0064 | 0.0001 | 0.042 | 0.003 | 0.001 | **0.224** | 2.589 | 2.848 |
| 5 | 0.0070 | 0.0069 | 0.0001 | 0.048 | 0.003 | 0.001 | **0.065** | 2.677 | 2.848 |

Validation at epoch 5 matches training (`val_total_loss = 0.0058`,
`val_ce = 0.0001`) — no overfitting in 5 epochs.

### What changed quantitatively

| Metric | Hilbert (epoch 5) | MSE-only (epoch 5) | All-fixes (epoch 5) |
|--------|------------------:|-------------------:|--------------------:|
| Output entropy `H_gen` | <1 nat (2–3 chars) | ~0.7 nats (mostly `' '`) | **2.68 nats** |
| Unigram KL | n/a (degenerate) | 0.42 (single-peak) | **0.065** |
| Train loss | hard to interpret (Hilbert units) | 0.18 | 0.007 (MSE+CE) |
| Mid/high-γ MSE | unstable | 0.025 | <0.005 |

`H_gen / H_gt = 0.94` — generated unigram entropy is 94 % of the text8 corpus
entropy. Output uses the full vocabulary, not a single mode.

### Why each fix mattered

- **Conservative gradient at sample time (#1)**: training regresses
  `f + (J_f^T)x` to the FM target, so descending raw `f` was descending the
  wrong field. Re-aligning train ↔ inference is the difference between "model
  learned an energy minimum here" and "sampler can find it".
- **Same σ in train and sample (#2)**: training saw `x0 ~ 0.1·N(0,I)`; sampler
  used `0.01·N(0,I)`, so inference x0 was 10× outside the training input
  distribution. The model treated test-time noise as an OOD point.
- **Aux CE (#3)**: this is the structural fix. The energy field's
  unconditional minima had no token-specific terminus, so they collapsed onto
  the marginal mode. Anchoring `predicted_x1 = x_γ − λ·grad_g` against the
  ground-truth tokens at γ ≥ 0.5 installs per-token attractors. CE drops to
  ~0.0001 by epoch 2 — the regression *can* recover x1 from x_γ at high γ; it
  just wasn't being asked to.
- **γ importance sampling (#4)**: with `gamma_power=0.5`, samples concentrate
  in `γ ∈ [0.5, 1]`, where the regression target carries the most information.
  Reduces wasted gradient on the irreducible-variance noise endpoint.
- **Unigram-KL probe (#5)**: caught the collapse story epoch by epoch instead
  of having to read it off the eval script after the fact.
- **3e-4 LR + grad clip (#7)**: at 1e-5 the model was barely moving in 5
  epochs; 30× higher LR with clipping was the difference between "loss
  plateaus around 0.2" and "loss drops 10× per epoch".

## 5. Open questions

### 5.1 Generated text is not yet English

Even with the unigram and (partly) bigram statistics close to GT, the panel-G
samples after 5 epochs are character soup, not words. Examples:

```
" qiuint  th  ix  ettin tit i tttt fff utwa"
" tx tetia th  in tttin tit i tttt fff wkwa"
```

The model has learned **per-position character distributions** but not
**multi-character coherence**. Likely contributors:

- Only 5 epochs at L=40, B=64, ~10k training windows.
- Aux CE is per-position; nothing in the loss explicitly penalises invalid
  digraphs/trigrams.
- The transformer has only positional info, no character-level inductive bias
  for n-gram structure.

**Likely path forward**: longer training (20-50 epochs), larger window count,
and / or an n-gram likelihood term computed on `predicted_x1`. Worth comparing
against the DFM baseline already in the repo (`models/dfm.py`) — DFM with the
same compute should already produce digraph-level structure.

### 5.2 The visual eval shown earlier is stale

`evaluate_eqm.py`'s internal `_sample` / `_sample_snapshots` /
`_sample_grad_norms` originally called `model(x)` directly (raw `f`, the
non-conservative field) and used `σ=0.01`. The image saved at
`scripts/evaluate_eqm.png` was therefore a worst-case run of the new model
(KL=1.124 in panel C, vs. 0.065 from the in-training probe that uses
`model.sample()`). Those samplers have now been patched to use the
conservative gradient and the trained σ, but the figure has not been
regenerated. **Re-running `python scripts/evaluate_eqm.py` would produce a
fairer picture**.

### 5.3 BPD metric is tautological

`evaluate_eqm.py` panel A and `model.bpd()` both start from `x1 + 0.1·randn`
and run NAG-GD. With η=0.05 and 5 steps the position has barely moved, yet
argmax recovers `x1` exactly. BPD is `0.0000` for every step count, every
model. It does not measure model quality.

A more meaningful BPD would either: (a) integrate the conditional likelihood
along the flow path; (b) start from a fixed σ-noise initialisation and measure
NLL on a held-out set; or (c) be replaced entirely with the unigram /
bigram / longer-range coherence metrics already in the eval.

### 5.4 Hilbert vs MSE — does it still matter?

With aux CE anchoring per-token attractors and importance-sampled γ pushing
mass toward γ ≈ 1, Hilbert's 2-sparse subgradient should be much less
crippling. An ablation (same fixes, swap loss back to `hilbert_soft` with
real α-annealing) would tell us whether the geometry-aware metric still pays
off when the optimisation can converge. Anchoring the simplex direction with
CE may have eliminated Hilbert's main weakness as a regression loss.

### 5.5 Reconstruction is perfect; unconditional generation isn't English

Reconstruction-from-perturbed-GT works and `H_gen` matches GT, but
unconditional samples don't read as English. This means: the data manifold's
attractors exist locally, but the **basin geometry around random
initialisations** doesn't carry a sample to a coherent point — it carries it
to a *typical-looking* but not *valid* point. The DFM-style time conditioning
(γ as input) might help here, since right now the model has to infer "where
am I in the flow" from the structure of x alone.

### 5.6 Code/architecture leftovers

- `volume_penalty` in `geometry.py` is fully implemented but not consumed.
- `lambda_vol`, `lambda_mse`, `alpha` in `EqM` config are unused.
- `Text8DataConfig.enabled` flag is never gated against.
- Position embedding is `Embedding(L, d)` — the model can't generate longer
  than the trained length without retraining.

These are hygiene items, not correctness blockers.

## 6. Concrete next experiments

In rough priority order:

1. **Re-run `evaluate_eqm.py` with the patched samplers** to get a fair
   visual scorecard.
2. **Train 25–50 epochs** on the same config — extrapolate whether
   bigram-level coherence emerges.
3. **DFM head-to-head** at the same compute budget — both models are in the
   repo; an `nfe_comparison.py`-style sweep would be the capstone's
   "continuous-flow generates discrete tokens" headline.
4. **Drop CE, ablate**: how much of the recovery is the CE term and how much
   is the noise-scale + conservative-gradient fix?
5. **Hilbert + CE**: does Hilbert's L∞ shape still hurt once CE anchors the
   target direction?
6. **Replace BPD** with a held-out unigram/bigram NLL metric so eval has a
   real number for "how much of text8 did we capture".

## 7. File-by-file recap of edits

- `src/aitchinson_flow/config.py` — `lr=3e-4`, `grad_clip_norm=1.0`,
  `loss.mode="mse"`, EqM gets `lambda_ce`, `ce_min_gamma`, `gamma_power`,
  `source_sigma`, plus `sample_eval_*` knobs in TrainingConfigs.
- `src/aitchinson_flow/models/eqm.py` — autograd `_compute_grad`,
  σ-from-config, γ importance, aux CE branch, single `forward` /
  `training_step` / `eval_step` block.
- `src/aitchinson_flow/training/runner.py` — `_unigram_kl_probe` per epoch,
  final checkpoint saved without optimizer state.
- `src/aitchinson_flow/training/loops.py` — postfix shows `flow_loss` / `ce`.
- `src/aitchinson_flow/training/optim.py` — `build_scheduler` raises on
  unknown name; `_clamp_unit_open` no longer always-false.
- `main.py` — dead code removed; only `--out-dir`, `--epochs`, `--lr`.
- `scripts/evaluate_eqm.py` — internal samplers use conservative gradient
  and `cfg.eqm.source_sigma`; panel G has `top-k` / `sample` decode.
- `scripts/eval_plots.py`, `scripts/compare_gt_vs_gen.py` — rewritten in CLR;
  ILR only for distance reporting.

Final EqM checkpoint: `checkpoints/epoch_final.pt` (385 MB, model only).
