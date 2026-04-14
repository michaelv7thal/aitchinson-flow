# Architecture Overview

This document describes how the major components of the `aitchinson_flow` codebase fit together.

---

## High-level picture

```
Config
  │
  ├─► LM (HFCausalLMInference)
  │       │
  │       ▼
  │   CausalLMTeacher  ──► CausalLMTeacherDataModule
  │                                │
  │                                ▼ batches: {log_x, token_ids?, logits?}
  │
  ├─► build_model(cfg)  ──► fit(cfg, datamodule, model)  ──► trained model
  │                                                               │
  └─► BenchmarkTask.run(model, datamodule, cfg)  ◄───────────────┘
            │
            ▼
      results: dict[scale_tag, dict[metric, float]]
```

---

## 1. Config

**File:** [`src/aitchinson_flow/config.py`](src/aitchinson_flow/config.py)

All configuration lives in a tree of plain `@dataclass` objects rooted at `Config`:

| Sub-config | Purpose |
|---|---|
| `DatasetConfig` | Vocabulary size `K`, sequence length `L`, dataset name |
| `TransformerConfig` | `d_model`, `nhead`, `num_layers`, `d_latent`, dropout |
| `TrainingConfig` | `model_name`, batch size `B`, epochs, LR, scheduler, device, `use_tqdm` |
| `BenchmarkConfig` | Scale grid, task name, LM key, `n_batches`, corruption, `train_before_eval`, `train_epochs` |
| `GPConfig` | Inducing point count, KL/variance margins for Bayesian models |
| `TeacherConfig` | HF model ID, dtype, device, caching for the teacher LM |
| `HFDatasetConfig` | Hub paths, splits, streaming settings |
| `EquilibriumFlowConfig` | Interpolation endpoints, ODE generation steps |

`Config()` with no arguments gives working defaults for every field.
Scale sweeps call `apply_transformer_scale(cfg, ...)` which `deepcopy`s and `replace`s the relevant fields without mutating the original.

---

## 2. Registries

Three independent registries follow the same pattern — a dict populated by `@register("key")` decorators on factory functions, and a `build_X(key, cfg)` lookup function.

| Registry | File | Key type |
|---|---|---|
| **Models** | [`src/aitchinson_flow/models/factory.py`](src/aitchinson_flow/models/factory.py) | `cfg.training.model_name` |
| **Tasks** | [`benchmarks/tasks/registry.py`](benchmarks/tasks/registry.py) | `cfg.benchmark.task_name` |
| **LMs** | [`src/aitchinson_flow/llms/registry.py`](src/aitchinson_flow/llms/registry.py) | `cfg.benchmark.lm_key` |

Registries are populated as a side-effect of importing the relevant module (e.g. `import aitchinson_flow.models` at the top of the benchmark runner). Any new model, task, or LM only needs to call `@register("name")` on its factory function to be discoverable.

---

## 3. Data pipeline

**Files:**
- [`src/aitchinson_flow/data/teachers/causal_lm.py`](src/aitchinson_flow/data/teachers/causal_lm.py)
- [`src/aitchinson_flow/training/datamodule.py`](src/aitchinson_flow/training/datamodule.py)

```
HFCausalLMInference
  │  .generate_ids(batch_size, max_new_tokens, prompt_ids?)
  │  .forward_logits(input_ids)
  ▼
CausalLMTeacher
  │  .sample_log_x()           → {log_x: (B,L,K)}
  │  .sample_with_logits()     → {log_x, token_ids, logits}
  ▼
CausalLMTeacherDataset   (IterableDataset, streams n_batches batches)
  ▼
CausalLMTeacherDataModule  implements DataModule protocol
  │  .train_dataloader()       → DataLoader (batch_size=None)
  │  .val_dataloader()         → None
  └  .test_dataloader()        → None
```

Each `__iter__` of the dataset calls the teacher to generate a fresh batch on the fly. `emit_logits=True` (set when `compute_spilled_energy=True`) switches from `sample_log_x` to `sample_with_logits`, adding `token_ids` and raw `logits` to every batch. Those extra fields are what the spilled-energy computation requires.

**Text8 prompts:** when `use_text8_prompts=True`, [`benchmarks/text8_prompts.py`](benchmarks/text8_prompts.py) loads real text8 chunks, tokenizes them, and passes them to the teacher as conditioning prompts so generated sequences are grounded in natural-language context rather than BOS tokens.

---

## 4. Models

**Base protocols:** [`src/aitchinson_flow/models/base.py`](src/aitchinson_flow/models/base.py)

```
GenerativeTrainingModel (Protocol)
  training_step(batch, step) → LossDict   # must include key "loss"
  eval_step(batch)           → LossDict

AuditorModel(GenerativeTrainingModel)
  audit(batch)               → LossDict   # inference-time scoring
```

Registered concrete models:

| Key | File | What it does |
|---|---|---|
| `flow_matching` | [`models/flow_matching.py`](src/aitchinson_flow/models/flow_matching.py) | Linear interpolation flow on the log-simplex. `training_step` supervises a velocity field; `audit` returns MSE velocity error. |
| `bayesian_generator` | [`models/bayesian_generator.py`](src/aitchinson_flow/models/bayesian_generator.py) | TransformerBackbone → latent `z` → SparseGP energy. Velocity = −∇E. Adds KL and volume penalties. |
| `bayesian_auditor` | [`models/bayesian_auditor.py`](src/aitchinson_flow/models/bayesian_auditor.py) | Extends `bayesian_generator` with contrastive valid/invalid hinge losses. Uses GP variance as OOD score. |
| `per_token_bayesian_auditor` | [`models/per_token_bayesian_auditor.py`](src/aitchinson_flow/models/per_token_bayesian_auditor.py) | Maintains a separate GP per sequence position; optionally conditions on teacher hidden states. |

All models consume `batch["log_x"]` as the primary input. Bayesian auditor variants also expect `batch["log_x_invalid"]` (corrupted samples) during training.

---

## 5. Training runner

**Files:**
- [`src/aitchinson_flow/training/runner.py`](src/aitchinson_flow/training/runner.py)
- [`src/aitchinson_flow/training/loops.py`](src/aitchinson_flow/training/loops.py)

```
fit(cfg, datamodule, model=None, resume_from=None) → nn.Module
  │
  ├─ build_model(cfg)  if model not provided
  ├─ build_optimizer(model, cfg)
  ├─ build_scheduler(optimizer, cfg)
  ├─ optional: load_checkpoint(resume_from, ...)
  │
  └─ epoch loop  [tqdm: "epochs", leave=True]
       │
       ├─ train_epoch(model, train_loader, optimizer, ...)
       │    └─ batch loop  [tqdm: "train epoch N", leave=False]
       │         model.training_step(batch, step) → loss → backward → step
       │         running_average(agg, counts, out)
       │
       ├─ evaluate(model, val_loader, ...)   every eval_every epochs
       │    └─ batch loop  [tqdm: "validation", leave=False]
       │         model.eval_step(batch) → metrics
       │
       ├─ epoch_pbar.set_postfix(train=..., val=..., lr=...)
       ├─ scheduler.step()
       └─ save_checkpoint(...)   every checkpoint_every epochs
```

**tqdm nesting:** during a run you see three live bars — epochs (persistent) above train-batch (transient) and optionally val-batch (transient). The epoch bar's postfix updates with the epoch-average train loss, val loss, and current LR.

Metric accumulation uses [`src/aitchinson_flow/training/metrics.py`](src/aitchinson_flow/training/metrics.py): `running_average` keeps a running sum + count dict; `finalize_averages` divides at the end of each loop.

---

## 6. Benchmark runner

**Files:**
- [`benchmarks/runner.py`](benchmarks/runner.py)
- [`benchmarks/tasks/text_audit.py`](benchmarks/tasks/text_audit.py)
- [`benchmarks/corruption.py`](benchmarks/corruption.py)

```
run_benchmark(cfg) → dict[scale_tag, dict[metric, float]]
  │
  ├─ build_lm(cfg.benchmark.lm_key, cfg.teacher)
  ├─ CausalLMTeacher(lm, cfg)
  ├─ optional: load_text8_prompts(lm, n_prompts, prompt_length)
  ├─ CausalLMTeacherDataModule(teacher, cfg, emit_logits=compute_spilled_energy)
  ├─ build_task(cfg.benchmark.task_name)
  │
  └─ scale loop  [tqdm: "benchmark scales"]
       │
       ├─ apply_transformer_scale(cfg, d_model, num_layers, nhead)
       ├─ override epochs if benchmark.train_epochs is set
       ├─ build_model(scfg)
       ├─ fit(scfg, datamodule, model)   if train_before_eval=True
       └─ task.run(model, datamodule, scfg) → dict[metric, float]
```

`TextAuditTask.run` iterates the dataloader once in `model.eval()` mode:
1. Calls `build_invalid_batch` on each batch to generate corrupted counterparts.
2. Calls `model.audit(batch)` — the trained model's detection metrics.
3. If `compute_spilled_energy`, calls `_spilled_metrics` for the training-free baseline.
4. All metrics are accumulated via `running_average` and returned as floats.

The result dict has two non-overlapping metric namespaces, so trained-model and spilled-energy scores can be directly compared:

```
results["d_model128_nhead8_num_layers4"] = {
    # model metrics (from audit())
    "loss":            0.31,
    "velocity_loss":   0.31,

    # spilled-energy baseline (training-free)
    "spilled_mean_valid":    -4.21,
    "spilled_mean_invalid":  -3.87,
    "spilled_separation":     0.34,
    "spilled_anomaly_valid":  4.21,
    "spilled_anomaly_invalid": 3.87,
    "spilled_token_mean":    -4.21,
    "spilled_token_std":      0.62,
}
```

---

## 7. Spilled energy

**File:** [`src/aitchinson_flow/metrics/spilled_energy.py`](src/aitchinson_flow/metrics/spilled_energy.py)

**Reference:** Minut, Dewidar & Masi, "Spilled Energy in Large Language Models", ICLR 2026.

The per-token spilled energy measures how much probability mass the LM "wasted" on tokens it did not actually predict:

```
ΔE(x_i) = −logsumexp(logits[i−1])   ← marginal energy at step i−1
         + logits[i−1, token_id[i]]  ← logit of the token that actually followed
```

A low (more negative) value means the LM confidently placed mass on the correct token — normal behaviour. A high (less negative) value means the actual token was assigned low probability — anomalous.

The sequence-level anomaly score is `−mean(ΔE)`: higher = more anomalous.

In the benchmark, spilled energy is compared against the trained auditor's own anomaly scores to evaluate whether learned detection adds value beyond this training-free baseline.

---

## 8. Entry points

| Script | Purpose |
|---|---|
| `benchmarks/runner.py` | Scale sweep: train each model variant and run the audit benchmark. Results saved to `cfg.benchmark.results_dir/benchmark_latest.json`. |
| `scripts/scale_sweep.py` | Lighter training-only sweep (calls `fit` without a benchmark task). |

Both accept a `Config()` built with defaults. Override fields directly or pass `--config-out path.json` to the benchmark runner to dump the resolved config for inspection.
