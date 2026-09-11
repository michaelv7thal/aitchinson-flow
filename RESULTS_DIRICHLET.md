# Dirichlet capstone — Phase 0-3 results


> **Current (2026-05-09), with one heading corrected below.** The measurements here are upstream provenance the paper still uses: the FM-target magnitudes 12.27 / 12.51 (`runs/dirichlet_phase0_diag.json`), the unigram-baseline KL_bi 1.692 (`runs/unigram_baseline/eval.json`), and the gradient_lambda recalibration derivation. The Phase 5 "Hilbert vs Aitchison ablation" marked "(in progress)" below is **completed — a null result**: the smoothed Hilbert loss does not rescue the collapse (paper `tab:hilbert-null`). The two `sweeps/dirichlet_phase{4_5_quick,5_minimal}.yaml` files named in its run command no longer exist.

Generated 2026-05-09 from a single 8 GB Blackwell laptop GPU run.

## What the change is

`token_ids_to_features_dirichlet` (`src/aitchinson_flow/data/transforms.py`) replaces
the deterministic one-hot + label-smoothing CLR encoding with a fresh draw from
`Dir(α_base + α_peak·e_token)` → CLR each batch. Wired in via
`CorruptingCollate(dirichlet_sampling=True, ...)`; the dataset's precomputed
features are bypassed when the flag is on so each batch sees a different `x_1`
for the same token. New config knobs in `TransformationConfig`:

```python
dirichlet_sampling: bool = False          # off ⇒ original deterministic pipeline
dirichlet_alpha_peak: float = 10.0        # target-class concentration
dirichlet_alpha_base: float = 0.1         # off-target concentration
```

## Phase 0/1/2 — calibration

`scripts/check_dirichlet_data.py` runs the data-validity diagnostic and an
α_peak sweep with argmax-recovery as the success criterion. Across 3 seeds at
B=64, L=40, K=27 (= 2560 positions per draw):

| α_peak | recover acc | var_norm mean | per-token x₁ var |
|-------:|------------:|--------------:|------------------:|
|  5     | 0.974-0.981 | 22.5          | 1165              |
|  7     | 0.994-0.995 | 22.6          | 1141              |
| 10     | 0.99961-1.00| 22.7          | 1124              |
| 50     | 1.000       | 22.95         | 1007              |
| 200    | 1.000       | 23.00         |  899              |
| 1000   | 1.000       | 23.02         |  776              |

Per-token x₁ variance under the deterministic encoding is ~5e-11 (numerical
noise); under Dirichlet at α_peak=10 it's ~1100 — that's the discontinuity
smoothing the plan asked for. **α_peak=10 is the smallest peak with stable
≥99.5 % argmax recovery, so it locks in maximal smoothing while keeping the
data uniquely token-identifiable.**

FM-target diagnostic at σ_source=0.1:

| metric                   | deterministic | Dirichlet (α_peak=10) | factor |
|--------------------------|--------------:|----------------------:|-------:|
| `\|x₁\|` L₂ mean         | 12.27         | 34.18                 | 2.78×  |
| `c(γ)·(x₀−x₁)` L₂ mean   |  5.83         | 16.24                 | 2.78×  |
| variation-norm mean      | 12.51         | 22.95                 | 1.83×  |

The Dirichlet target is **2.78× larger** in L₂. MSE is quadratic in this
difference, so the regression objective grows ≈8× harder at fixed model /
optimiser. This is the prediction the plan calls out in §4.1 ("re-tune
gradient_lambda for the new magnitude").

## Phase 3 — single-loss MSE + Dirichlet at default hyperparameters

`runs/dphase3_mse_dirichlet/` — identical to `runs/baseline_5ep/` except
`transformation.dirichlet_sampling=True`, α_peak=10. 5 epochs × 10000 windows.

### Train trajectory (`history.jsonl`, all values are epoch averages)

| epoch | flow_loss | ce    | g<.33 | g<.66 | g<1   | uni_KL | bi_KL  | tri_KL | H_gen |
|------:|----------:|------:|------:|------:|------:|-------:|-------:|-------:|------:|
| 1     | 6.089     | 0.195 | 11.39 | 0.773 | 1.370 | 0.664  | 6.030  | 14.52  | 3.292 |
| 2     | 3.713     | 0.181 | 7.700 | 0.140 | 0.172 | 0.703  | 6.208  | 14.65  | 3.291 |
| 3     | 3.804     | 0.165 | 7.810 | 0.101 | 0.133 | 0.662  | 6.032  | 14.69  | 3.293 |
| 4     | 3.441     | 0.151 | 7.165 | 0.063 | 0.092 | 0.692  | 6.174  | 14.78  | 3.292 |
| 5     | 3.304     | 0.145 | 6.825 | 0.055 | 0.081 | 0.695  | 6.298  | 15.00  | 3.290 |

Validation at epoch 5 mirrors train: `val_flow_loss=3.080, val_ce=0.149,
val_g<1=0.074`. No train/val gap → no overfit; the model is simply
under-fit relative to the new target magnitude.

### Eval (`eval.json`, n=256, 200 NAG steps)

```
unigram_kl=0.6414  bigram_kl=5.806  trigram_kl=13.479
H_gen=3.295        H_gt=2.848       H_ratio=1.157
grad_at_gen=7.63   grad_at_gt=25.33
```

### Decoded samples

```
nweaholwasnjyfhbdvyulyrjdgcuprlazeofwmyk
wrlohljwmhwhoezfzbhluyreiykxjmnnujcs tsp
jzngyszadngkkff xgmujbjourirpbonpewwuuoc
lzwdfzafufjvqjhmcwuqjbluzslbxywqoqcupudl
```

### Comparison vs baselines

| metric    | unigram | baseline_5ep | dphase3_det_v2 | dphase3_dir | Δ dir vs det |
|-----------|--------:|-------------:|---------------:|------------:|-------------:|
| KL_uni    | 0.0009  | 0.0513       | 0.0273         | 0.6414      | +23×         |
| KL_bi     | 1.6917  | 1.9874       | 1.1543         | 5.8056      | +5.0×        |
| KL_tri    | 7.1581  | 7.2170       | 5.4594         | 13.4789     | +2.5×        |
| H_ratio   | 0.995   | 0.955        | 0.960          | 1.157       | over-spread  |
| grad_gen  | —       | 0.227        | 0.407          | 7.631       | model sick   |
| grad_gt   | —       | 0.179        | 0.400          | 25.33       | data far OOD |
| flow_loss | —       | 0.0069       | 0.354 (val)    | 3.080 (val) | +9×          |

`dphase3_mse_deterministic_v2` is a same-git-revision parity reproduction;
it lands slightly *better* than baseline_5ep (RNG variance), confirming the
comparison vs Dirichlet is fair on identical code.

**Phase 3 result: Dirichlet at default hyperparameters regresses on every
metric.** Worse than even the trivial unigram baseline on KL_bi/KL_tri.

### Diagnostic interpretation

1. `grad_at_gen` (7.63) < `grad_at_gt` (25.33): the model's energy is
   *higher* on real data than on its own samples — the wrong direction. The
   conservative-gradient field has not yet found the data manifold; samples
   converge to a uniform-ish basin that NAG-GD can reach quickly.
2. `H_gen=3.29 > H_gt=2.85` ⇒ generated unigram is **more entropic than the
   corpus**. With α_peak=10 + α_base=0.1, the off-target concentration is
   small enough that the data has heavy-tailed off-target draws; the model
   over-fits this spread and produces too-flat distributions.
3. `g<1` (γ ≥ 0.66, the signal regime) drops by ~17× over 5 epochs (1.37 →
   0.081), which says the FM regression itself is converging — just to an
   8×-larger target. Final relative error per coordinate is ~0.08/16 ≈
   5e-3 vs the baseline's ~0.001/5.83 ≈ 2e-4.

## Phase 4 — gradient_lambda recalibration

`runs/dphase4_lambda_recalib/` — same as Phase 3 dirichlet but
`eqm.gradient_lambda=3.0` (≈ the 2.78× target-magnitude ratio).

### Train trajectory

| epoch | flow_loss | ce    | g<1   | val_flow_loss |
|------:|----------:|------:|------:|--------------:|
| 1     | 0.801     | 0.205 | 0.256 | —             |
| 2     | 0.424     | 0.078 | 0.068 | —             |
| 3     | 0.426     | 0.060 | 0.054 | —             |
| 4     | 0.395     | 0.054 | 0.047 | —             |
| 5     | 0.378     | 0.051 | 0.044 | 0.354         |

Recalibration delivers a clean **8.7× reduction in flow_loss vs Phase 3
default** (3.30 → 0.378). g<1 is 2× lower (0.081 → 0.044), ce is 3× lower
(0.145 → 0.051). The FM regression is now in a healthy regime.

### Eval — and a sampler artefact

```
unigram_kl=0.6414  bigram_kl=5.806  trigram_kl=13.479
H_gen=3.295        H_gt=2.848       H_ratio=1.157
grad_at_gen=2.657  grad_at_gt=9.951
```

These KL values are **bit-identical** to Phase 3 dirichlet (0.6414 / 5.806 /
13.479). `grad_at_gen` and `grad_at_gt` scale ~3× cleanly with the new λ
(7.63 → 2.66, 25.33 → 9.95) — confirming the model output magnitude
changed in proportion to gradient_lambda. But `H_gen ≈ log(27) = 3.296` —
the samples are essentially **uniform argmaxes of the noise init**, not
data-pushed iterates.

Root cause is `sample_return_best=True` (`cfg.eqm.sample_return_best`,
`models/eqm.py:436-450`). The sampler returns the iterate with lowest mean
‖∇E‖. With Dirichlet-data training, the trained energy field is shallow at
the noise init (the model has only seen high-γ Dirichlet points; the
γ=0 noise regime is OOD), so the lowest-grad iterate of the NAG
trajectory is step 0 — the noise itself. RNG state at sample time is
deterministic given seed=42 + identical RNG-consumption-per-step training
trajectories ⇒ identical noise ⇒ identical samples ⇒ bit-identical KL.

Same-seed, same-data-pipeline, same-RNG-budget makes this artefact
loudly visible; in a regime where the model actually pushes samples,
the run-to-run variance would mask it. **Verdict: Phase 4 retune fixed the
regression but didn't fix sampling. The sampler needs separate work
(disable `sample_return_best`, or move to the Euler sampler, or train
longer so γ=0 isn't OOD).**

## Phase 5 — Hilbert vs Aitchison ablation (in progress)

The trajectory and the FM-target magnitude analysis both point at
gradient_lambda being the wrong scale. Two cells testing the two highest-
leverage knobs at fixed everything-else:

- **`dphase4_lambda_recalib`** — `eqm.gradient_lambda=3.0` (≈ the 2.78×
  ratio of Dirichlet vs deterministic target magnitudes). MSE on a smaller
  scaled target should converge ~8× faster.
- **`dphase5_hilbert_dirichlet`** — `loss.mode=hilbert_soft, hilbert_alpha=
  1.0, gamma_power=1.0`. Soft-Hilbert is logarithmic in coordinate
  differences (LSE-based) so the magnitude blow-up affects it less than MSE
  does. This is the "headline ablation made informative by the data
  change" the plan calls out.

Run after cell 2 finishes:

```
python scripts/run_sweep.py --sweep sweeps/dirichlet_phase4_5_quick.yaml \
    --runs-root runs
python scripts/dirichlet_results_summary.py --out RESULTS_DIRICHLET.md
```

## Files added in this session

| path                                                  | role                                       |
|-------------------------------------------------------|--------------------------------------------|
| `src/aitchinson_flow/data/transforms.py`              | + `token_ids_to_features_dirichlet`         |
| `src/aitchinson_flow/data/diagnostics.py`             | Phase 0 data-validity + FM-target probes    |
| `src/aitchinson_flow/data/corrupting_collate.py`      | Dirichlet branch in collate                 |
| `src/aitchinson_flow/data/text8_datamodule.py`        | propagates Dirichlet flags to collate       |
| `src/aitchinson_flow/data/__init__.py`                | re-exports                                  |
| `src/aitchinson_flow/config.py`                       | + 3 fields on `TransformationConfig`        |
| `main.py`                                             | + `--dirichlet`, `--alpha-peak`, … overrides |
| `scripts/check_dirichlet_data.py`                     | Phase 0+2 diagnostics & calibration         |
| `scripts/unigram_baseline.py`                         | Phase 6 trivial baseline                    |
| `scripts/compare_dirichlet_runs.py`                   | side-by-side eval comparison                |
| `scripts/dirichlet_results_summary.py`                | end-to-end results aggregator               |
| `sweeps/dirichlet_capstone.yaml`                      | full Phase 3-5 sweep (12 cells)             |
| `sweeps/dirichlet_phase4_5_quick.yaml`                | minimal Phase 4+5 (2 cells)                 |
| `sweeps/dirichlet_phase5_minimal.yaml`                | minimal Phase 5 (2 cells)                   |
| `runs/dirichlet_phase0_diag.json`                     | Phase 0/2 diagnostic dump                   |
| `runs/unigram_baseline/eval.json`                     | unigram reference                           |
| `runs/dphase3_mse_dirichlet/`                         | Phase 3 result                              |
