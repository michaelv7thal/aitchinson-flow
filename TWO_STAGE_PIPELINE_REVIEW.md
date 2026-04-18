# Two-Stage Training Pipeline: Code Review (Round 2)

**Date:** 2026-04-18  
**Scope:** Working-tree changes since last commit — tracked files (`git diff HEAD`) plus untracked stage files and tests.

---

## Status of Previous Findings

| ID | Description | Status |
|----|-------------|--------|
| C1 | Token projection mismatch in Stage 2 | ✅ Fixed |
| C2 | Contrastive detach (asymmetric gradients) | ✅ Fixed — but see **N1** below |
| C3 | Helmert matrix scaling incorrect | ✅ Fixed |
| C4 | Missing loss key crashes training | ✅ Fixed |
| C5 | Fused checkpoint carries wrong epoch metadata | ✅ Fixed |
| C6 | Tests: random seed never fixed | ✅ Fixed |
| C7 | Composed `BayesianAuditor.training_step` untested | ✅ Fixed |
| M1 | `d_model % nhead == 0` not validated | ❌ Not fixed |
| M2 | KL divided by batch size, not dataset size | ✅ Fixed |
| M3 | Per-token KL normalisation inconsistency | ✅ Fixed |
| M4 | Dropout config setting silently ignored | ✅ Fixed |
| M5 | No L2 regularisation on GP hyperparameters | ✅ Fixed |
| M6 | Margin and lambda hyperparameters unvalidated | ✅ Fixed |
| M7 | Unexpected keys silently dropped at composition | ✅ Fixed |
| M8 | Epoch count not validated | ✅ Fixed |
| M9 | Batch schema not validated in Stage 2 | ✅ Fixed |
| M10 | Cosine scheduler `T_max` not clamped in warmup branch | ✅ Fixed |
| M11 | Tests: only `soft_hilbert` velocity loss tested | ❌ Not fixed |
| M12 | Tests: sign-convention assertion is conditional | ✅ Fixed |
| M13 | Variance loss commented out, config params still exposed | ✅ Fixed |

---

## New Issues Introduced by the Fixes

### N1 — VelocityHead Mean-Subtraction Wrong for ILR Mode (New Critical)
**File:** [transformer_backbone.py:119-120](src/aitchinson_flow/transformer_backbone.py#L119)

The refactored `VelocityHead.forward` now subtracts the feature-dimension mean from the projected output:
```python
return v - v.mean(dim=-1, keepdim=True)
```
The docstring justifies this as "projecting onto the Aitchison tangent space (sum-to-zero constraint)." This is only correct for **CLR mode** (`K`-dimensional coordinates). In **ILR mode** (the default, `K-1`-dimensional), the ILR transform is designed to *remove* the sum-to-zero constraint — ILR coordinates live in unconstrained R^(K-1). Subtracting the mean imposes that constraint back, silently reducing the effective rank of the velocity field from `K-1` to `K-2` and introducing a systematic bias. The original `VelocityHead` had no mean subtraction.

The docstring also contains a mathematical error: it calls R^(K-1) "the ILR subspace of R^(K-1)"; ILR is not a subspace of itself, it IS R^(K-1). `V_K` (the Aitchison tangent space) lives in R^K with the sum-to-zero constraint — the ILR transform maps `V_K → R^(K-1)` *to remove that constraint*.

**Fix:** Gate the mean subtraction on `transform_mode`:
```python
v = self.proj(h)
if self.cfg.hf_dataset.transform_mode == "clr":
    v = v - v.mean(dim=-1, keepdim=True)
return v
```

---

### N2 — Stage 2 Docstring Still Says `mean_v_detach` After C2 Fix (New Minor)
**File:** [bayesian_auditor_stage2.py:67](src/aitchinson_flow/models/bayesian_auditor_stage2.py#L67)

The class docstring pseudo-code block still reads:
```
contrastive = relu(margin_E - (mean_r - mean_v_detach)).mean()
```
`mean_v_detach` no longer exists — the detach was intentionally removed as part of the C2 fix. A reader of the docstring would believe the old (broken) behaviour is still in effect.

---

### N3 — `kl_normalizer` Falls Back to Batch Size Without Emitting a Warning at Test Time (New Minor)
**File:** [models/base.py:14-41](src/aitchinson_flow/models/base.py#L14)

`kl_normalizer` warns once per model-id when `_kl_normalizer` is absent, using the `id(model)` as a dedup key. During tests, multiple test methods construct fresh model instances; because each instance has a unique `id`, the warning fires for every test that exercises the KL path, flooding the test output. A per-test-run or module-level dedup (e.g. keying on `type(model).__name__`) would be less noisy.

---

## Remaining Open Issues (Carried from Round 1)

### R1 — `d_model % nhead == 0` Not Validated (was M1)
**Files:** [config.py:16-19](src/aitchinson_flow/config.py#L16), [transformer_backbone.py:47-48](src/aitchinson_flow/transformer_backbone.py#L47)

`TransformerConfig` has no `__post_init__`. A misconfigured `(d_model, nhead)` pair produces a confusing runtime error inside PyTorch rather than a clear validation message at config time.

**Suggested fix:** Add to `TransformerConfig.__post_init__`:
```python
if self.d_model % self.nhead != 0:
    raise ValueError(
        f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
    )
```

---

### R2 — Velocity Loss Types `hard_hilbert`, `clr_mse`, `ilr_mse` Have No Test Coverage (was M11)
**File:** [tests/test_two_stage_auditor.py:40](tests/test_two_stage_auditor.py#L40)

Stage 1 is still only parameterised with `soft_hilbert`. Bugs in the other three `_HILBERT_FAMILY` members would not be caught.

---

### R3 — GP Noise Variance Not Configurable; Softplus Can Underflow (was m3)
**File:** [gp/gp.py:71](src/aitchinson_flow/gp/gp.py#L71)

`log_noise_var` is hardcoded to `log(0.1)` with no config override. Training can push this very negative, causing `softplus` to underflow toward zero and breaking the NLL.

---

### R4 — Time Conditioning Unused in FlowMatchingAuditor (was m5)
**File:** [models/flow_matching.py:30,45](src/aitchinson_flow/models/flow_matching.py#L30)

Backbone is constructed with `time_conditioned=True` but the training target `u_tgt = log_x1 - log_x0` is time-independent. The time embedding is wasted model capacity.

---

## Summary

| Category | Count |
|----------|-------|
| Fixes verified | 17 / 20 |
| New issues introduced | 3 (1 critical, 2 minor) |
| Remaining unfixed from round 1 | 3 |

**Top priority:** Fix **N1** (`VelocityHead` ILR/CLR branch) before any training run — it silently corrupts velocity predictions in the default ILR configuration by constraining them to a lower-dimensional hyperplane. The other two new items (N2 stale docstring, N3 warning noise) are cosmetic.
